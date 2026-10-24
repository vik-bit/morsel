// Hash join: a build pipeline materializes the build side into a chained hash
// table; probe operators in a later pipeline look up every input row.

#include <algorithm>
#include <atomic>
#include <numeric>
#include <stdexcept>

#include "internal.h"

namespace morsel::detail {

namespace {

constexpr std::uint32_t kEmpty = 0xFFFFFFFFu;

// Build columns stored next to each key, row by row, widest first so every
// field is naturally aligned.
struct PayloadLayout {
  std::vector<int> columns;  // in the build schema
  std::vector<std::size_t> width, offset;
  std::size_t stride = 0;

  static PayloadLayout make(const Schema& schema, const std::vector<int>& columns) {
    PayloadLayout l;
    l.columns = columns;
    l.width.resize(columns.size());
    l.offset.resize(columns.size());
    std::vector<std::size_t> order(columns.size());
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t i = 0; i < columns.size(); ++i) {
      l.width[i] = type_width(schema.types[static_cast<std::size_t>(columns[i])].id);
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return l.width[a] > l.width[b]; });
    for (std::size_t i : order) {
      l.offset[i] = l.stride;
      l.stride += l.width[i];
    }
    l.stride = (l.stride + 7) / 8 * 8;
    if (l.stride == 0) l.stride = 8;
    return l;
  }
};

void copy_field(std::uint8_t* dst, const std::uint8_t* src, std::size_t w) {
  switch (w) {
    case 1:
      *dst = *src;
      break;
    case 4:
      std::memcpy(dst, src, 4);
      break;
    case 8:
      std::memcpy(dst, src, 8);
      break;
    default:
      std::memcpy(dst, src, 16);
  }
}

template <class K>
struct BuildLocal final : SinkState {
  std::vector<K> keys;
  Buffer payload;
  std::size_t rows = 0;
  std::vector<K> scratch = std::vector<K>(kVectorSize);
};

class JoinBuildBase : public Sink {
 public:
  JoinBuildBase(KeyLayout layout, PayloadLayout payload)
      : layout_(std::move(layout)), payload_(std::move(payload)) {}
  const PayloadLayout& payload() const { return payload_; }

 protected:
  KeyLayout layout_;
  PayloadLayout payload_;
};

template <class K>
class JoinBuild final : public JoinBuildBase {
 public:
  using JoinBuildBase::JoinBuildBase;

  std::unique_ptr<SinkState> make_local() const override { return std::make_unique<BuildLocal<K>>(); }

  void consume(Chunk& chunk, SinkState& state) const override {
    auto& local = static_cast<BuildLocal<K>&>(state);
    pack_keys(layout_, chunk, chunk.sel, local.scratch.data());
    for_each_selected(chunk.sel, [&](std::uint32_t i) { local.keys.push_back(local.scratch[i]); });
    const std::size_t first = local.rows;
    local.rows += chunk.sel.count;
    local.payload.resize(local.rows * payload_.stride);
    for (std::size_t c = 0; c < payload_.columns.size(); ++c) {
      const auto* src = static_cast<const std::uint8_t*>(chunk.cols[static_cast<std::size_t>(payload_.columns[c])]);
      const std::size_t w = payload_.width[c];
      std::uint8_t* dst = local.payload.data() + first * payload_.stride + payload_.offset[c];
      for_each_selected(chunk.sel, [&](std::uint32_t i) {
        copy_field(dst, src + i * w, w);
        dst += payload_.stride;
      });
    }
  }

  void finalize(std::vector<std::unique_ptr<SinkState>>& locals, ThreadPool& pool) override {
    std::vector<std::size_t> start(locals.size() + 1, 0);
    for (std::size_t l = 0; l < locals.size(); ++l) {
      start[l + 1] = start[l] + static_cast<BuildLocal<K>&>(*locals[l]).rows;
    }
    const std::size_t n = start.back();
    if (n >= kEmpty) throw std::runtime_error("hash join build side too large");
    keys_.resize(n);
    next_.resize(n);
    payload_rows_.resize(n * payload_.stride);
    pool.parallel_for(locals.size(), [&](std::size_t l) {
      auto& local = static_cast<BuildLocal<K>&>(*locals[l]);
      std::copy(local.keys.begin(), local.keys.end(), keys_.begin() + static_cast<std::ptrdiff_t>(start[l]));
      if (local.rows) {
        std::memcpy(payload_rows_.data() + start[l] * payload_.stride, local.payload.data(),
                    local.rows * payload_.stride);
      }
      local = BuildLocal<K>();  // free the local copy early
    });

    // Directory of chain heads, at least twice the row count so chains stay
    // short. Rows are pushed onto their bucket's chain with an atomic
    // exchange, so all workers insert at once without locks.
    const std::size_t buckets = next_pow2(std::max<std::size_t>(2 * n, 64));
    mask_ = buckets - 1;
    heads_.reset(new std::atomic<std::uint32_t>[buckets]);
    constexpr std::size_t kBlock = 1 << 16;
    pool.parallel_for((buckets + kBlock - 1) / kBlock, [&](std::size_t b) {
      const std::size_t end = std::min(buckets, (b + 1) * kBlock);
      for (std::size_t i = b * kBlock; i < end; ++i) heads_[i].store(kEmpty, std::memory_order_relaxed);
    });
    pool.parallel_for((n + kBlock - 1) / kBlock, [&](std::size_t b) {
      const std::size_t end = std::min(n, (b + 1) * kBlock);
      for (std::size_t i = b * kBlock; i < end; ++i) {
        const std::size_t bucket = hash_key(keys_[i]) & mask_;
        next_[i] = heads_[bucket].exchange(static_cast<std::uint32_t>(i), std::memory_order_relaxed);
      }
    });
  }

  std::uint32_t head(std::uint64_t hash) const { return heads_[hash & mask_].load(std::memory_order_relaxed); }
  const std::atomic<std::uint32_t>* head_ptr(std::uint64_t hash) const { return &heads_[hash & mask_]; }
  const K& key(std::uint32_t row) const { return keys_[row]; }
  std::uint32_t next(std::uint32_t row) const { return next_[row]; }
  const std::uint8_t* payload_row(std::uint32_t row) const { return payload_rows_.data() + row * payload_.stride; }

 private:
  std::vector<K> keys_;
  std::vector<std::uint32_t> next_;
  Buffer payload_rows_;
  std::unique_ptr<std::atomic<std::uint32_t>[]> heads_;
  std::size_t mask_ = 0;
};

template <class K>
class JoinProbe final : public Operator {
 public:
  JoinProbe(const PlanNode& join, std::shared_ptr<JoinBuild<K>> build)
      : build_(std::move(build)), type_(join.join_type) {
    const Schema& probe_schema = join.children[0]->schema;
    layout_ = KeyLayout::make(probe_schema, join.probe_keys);
    for (const auto& t : probe_schema.types) probe_widths_.push_back(type_width(t.id));
  }

  struct State final : OpState {
    std::vector<K> keys = std::vector<K>(kVectorSize);
    std::vector<std::uint64_t> hashes = std::vector<std::uint64_t>(kVectorSize);
    std::vector<std::uint32_t> pos = std::vector<std::uint32_t>(kVectorSize);
    std::vector<std::uint32_t> row = std::vector<std::uint32_t>(kVectorSize);
    Buffer sel{kSelCapacity * sizeof(std::uint32_t)};
    std::vector<Buffer> columns;
    Chunk out;
  };

  std::unique_ptr<OpState> make_state() const override {
    auto st = std::make_unique<State>();
    if (type_ == JoinType::Inner) {
      const PayloadLayout& pl = build_->payload();
      for (std::size_t w : probe_widths_) st->columns.emplace_back(kVectorSize * w);
      for (std::size_t w : pl.width) st->columns.emplace_back(kVectorSize * w);
      st->out.cols.resize(st->columns.size());
      for (std::size_t c = 0; c < st->columns.size(); ++c) st->out.cols[c] = st->columns[c].data();
    }
    return st;
  }

  void execute(Chunk& in, OpState& state, Emitter& next) const override {
    auto& st = static_cast<State&>(state);
    pack_keys(layout_, in, in.sel, st.keys.data());
    // Hash every key and prefetch its bucket first, so the cache misses of a
    // large table overlap instead of being taken one at a time.
    for_each_selected(in.sel, [&](std::uint32_t i) {
      st.hashes[i] = hash_key(st.keys[i]);
      __builtin_prefetch(build_->head_ptr(st.hashes[i]));
    });

    if (type_ != JoinType::Inner) {
      const bool want = type_ == JoinType::Semi;
      auto* sel = st.sel.template as<std::uint32_t>();
      std::uint32_t k = 0;
      for_each_selected(in.sel, [&](std::uint32_t i) {
        bool found = false;
        for (std::uint32_t r = build_->head(st.hashes[i]); r != kEmpty && !found; r = build_->next(r)) {
          found = build_->key(r) == st.keys[i];
        }
        sel[k] = i;
        k += found == want;
      });
      if (k == 0) return;
      in.sel = {sel, k};
      next.emit(in);
      return;
    }

    std::uint32_t m = 0;
    for_each_selected(in.sel, [&](std::uint32_t i) {
      for (std::uint32_t r = build_->head(st.hashes[i]); r != kEmpty; r = build_->next(r)) {
        if (!(build_->key(r) == st.keys[i])) continue;
        st.pos[m] = i;
        st.row[m] = r;
        if (++m == kVectorSize) {
          flush(in, st, m, next);
          m = 0;
        }
      }
    });
    if (m) flush(in, st, m, next);
  }

 private:
  void flush(const Chunk& in, State& st, std::uint32_t m, Emitter& next) const {
    const std::size_t np = probe_widths_.size();
    for (std::size_t c = 0; c < np; ++c) {
      gather(in.cols[c], probe_widths_[c], st.pos.data(), m, st.columns[c].data());
    }
    const PayloadLayout& pl = build_->payload();
    for (std::size_t c = 0; c < pl.columns.size(); ++c) {
      const std::size_t w = pl.width[c], off = pl.offset[c];
      std::uint8_t* dst = st.columns[np + c].data();
      for (std::uint32_t j = 0; j < m; ++j) copy_field(dst + j * w, build_->payload_row(st.row[j]) + off, w);
    }
    st.out.size = m;
    st.out.sel = {nullptr, m};
    next.emit(st.out);
  }

  std::shared_ptr<JoinBuild<K>> build_;
  JoinType type_;
  KeyLayout layout_;
  std::vector<std::size_t> probe_widths_;
};

}  // namespace

std::shared_ptr<Sink> make_join_build(const PlanNode& join) {
  const Schema& bs = join.children[1]->schema;
  KeyLayout layout = KeyLayout::make(bs, join.build_keys);
  PayloadLayout payload = PayloadLayout::make(bs, join.build_payload);
  if (layout.wide) return std::make_shared<JoinBuild<Key128>>(std::move(layout), std::move(payload));
  return std::make_shared<JoinBuild<std::uint64_t>>(std::move(layout), std::move(payload));
}

std::unique_ptr<Operator> make_join_probe(const PlanNode& join, const std::shared_ptr<Sink>& build) {
  if (auto b = std::dynamic_pointer_cast<JoinBuild<Key128>>(build)) {
    return std::make_unique<JoinProbe<Key128>>(join, b);
  }
  return std::make_unique<JoinProbe<std::uint64_t>>(join, std::dynamic_pointer_cast<JoinBuild<std::uint64_t>>(build));
}

}  // namespace morsel::detail
