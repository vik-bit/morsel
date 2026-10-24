// Hash aggregation. Every worker aggregates into its own hash table with no
// synchronisation. At the end the groups are split into partitions by hash,
// and each partition is merged by one worker, so the merge runs in parallel
// too (the scheme of the morsel-driven paper, without its spilling).

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>

#include "internal.h"

namespace morsel::detail {

namespace {

constexpr int kPartitionBits = 6;
constexpr std::size_t kPartitions = std::size_t{1} << kPartitionBits;
// Below this many groups in total, one thread merges everything: the
// partitioning pass would cost more than it saves.
constexpr std::size_t kParallelMergeGroups = 1 << 14;

// Each aggregate gets a 16-byte aligned slot in the group's state row.
struct AggLayout {
  std::vector<BoundAgg> aggs;
  std::vector<std::size_t> offset;
  std::size_t stride = 0;

  explicit AggLayout(const std::vector<BoundAgg>& a) : aggs(a) {
    for (const auto& agg : aggs) {
      offset.push_back(stride);
      const bool wide_avg = agg.kind == AggKind::Avg && agg.input_type.id != TypeId::Double;
      stride += wide_avg ? 32 : 16;
    }
    if (stride == 0) stride = 16;
  }
};

template <class T>
T& at(std::uint8_t* p) {
  return *reinterpret_cast<T*>(p);
}
template <class T>
const T& at(const std::uint8_t* p) {
  return *reinterpret_cast<const T*>(p);
}

void init_state(const AggLayout& layout, std::uint8_t* st) {
  std::memset(st, 0, layout.stride);
  for (std::size_t a = 0; a < layout.aggs.size(); ++a) {
    const BoundAgg& agg = layout.aggs[a];
    std::uint8_t* p = st + layout.offset[a];
    const bool is_double = agg.input_type.id == TypeId::Double;
    if (agg.kind == AggKind::Min) {
      if (is_double) at<double>(p) = std::numeric_limits<double>::infinity();
      else at<std::int64_t>(p) = std::numeric_limits<std::int64_t>::max();
    } else if (agg.kind == AggKind::Max) {
      if (is_double) at<double>(p) = -std::numeric_limits<double>::infinity();
      else at<std::int64_t>(p) = std::numeric_limits<std::int64_t>::min();
    }
  }
}

void merge_state(const AggLayout& layout, std::uint8_t* dst, const std::uint8_t* src) {
  for (std::size_t a = 0; a < layout.aggs.size(); ++a) {
    const BoundAgg& agg = layout.aggs[a];
    std::uint8_t* d = dst + layout.offset[a];
    const std::uint8_t* s = src + layout.offset[a];
    const bool is_double = agg.input_type.id == TypeId::Double;
    switch (agg.kind) {
      case AggKind::Count:
        at<std::int64_t>(d) += at<std::int64_t>(s);
        break;
      case AggKind::Sum:
        if (is_double) at<double>(d) += at<double>(s);
        else at<int128>(d) += at<int128>(s);
        break;
      case AggKind::Avg:
        if (is_double) {
          at<double>(d) += at<double>(s);
          at<std::int64_t>(d + 8) += at<std::int64_t>(s + 8);
        } else {
          at<int128>(d) += at<int128>(s);
          at<std::int64_t>(d + 16) += at<std::int64_t>(s + 16);
        }
        break;
      case AggKind::Min:
        if (is_double) at<double>(d) = std::min(at<double>(d), at<double>(s));
        else at<std::int64_t>(d) = std::min(at<std::int64_t>(d), at<std::int64_t>(s));
        break;
      case AggKind::Max:
        if (is_double) at<double>(d) = std::max(at<double>(d), at<double>(s));
        else at<std::int64_t>(d) = std::max(at<std::int64_t>(d), at<std::int64_t>(s));
        break;
      case AggKind::Any:
        std::memcpy(d, s, 16);
        break;
    }
  }
}

// Open addressing with linear probing over group indices. Keys and hashes
// live in arrays next to the state rows, so growing rehashes only the small
// slot array. Group-by keys of at most two dictionary columns (16 bits)
// index a direct table instead and never hash on the hot path.
template <class K>
class AggTable {
 public:
  AggTable(const AggLayout* layout, bool direct, std::size_t direct_bits)
      : layout_(layout), direct_(direct) {
    slots_.assign(direct ? (std::size_t{1} << direct_bits) : 1024, 0);
    mask_ = slots_.size() - 1;
  }

  std::uint32_t find_or_insert(const K& key) {
    if (direct_) {
      std::uint32_t& s = slots_[static_cast<std::size_t>(key_word(key, 0))];
      if (s == 0) s = add_group(key, hash_key(key)) + 1;
      return s - 1;
    }
    return find_or_insert(key, hash_key(key));
  }

  std::uint32_t find_or_insert(const K& key, std::uint64_t hash) {
    if (direct_) return find_or_insert(key);
    for (std::size_t i = hash & mask_;; i = (i + 1) & mask_) {
      const std::uint32_t s = slots_[i];
      if (s == 0) {
        const std::uint32_t g = add_group(key, hash);
        slots_[i] = g + 1;
        if (2 * groups() > slots_.size()) grow();
        return g;
      }
      if (hashes_[s - 1] == hash && keys_[s - 1] == key) return s - 1;
    }
  }

  std::size_t groups() const { return keys_.size(); }
  const K& key(std::uint32_t g) const { return keys_[g]; }
  std::uint64_t hash(std::uint32_t g) const { return hashes_[g]; }
  std::uint8_t* state(std::uint32_t g) { return states_.data() + g * layout_->stride; }
  const std::uint8_t* state(std::uint32_t g) const { return states_.data() + g * layout_->stride; }

 private:
  std::uint32_t add_group(const K& key, std::uint64_t hash) {
    const auto g = static_cast<std::uint32_t>(keys_.size());
    keys_.push_back(key);
    hashes_.push_back(hash);
    states_.resize((g + 1) * layout_->stride);
    init_state(*layout_, state(g));
    return g;
  }

  void grow() {
    slots_.assign(slots_.size() * 2, 0);
    mask_ = slots_.size() - 1;
    for (std::uint32_t g = 0; g < keys_.size(); ++g) {
      std::size_t i = hashes_[g] & mask_;
      while (slots_[i] != 0) i = (i + 1) & mask_;
      slots_[i] = g + 1;
    }
  }

  const AggLayout* layout_;
  bool direct_;
  std::vector<std::uint32_t> slots_;
  std::size_t mask_ = 0;
  std::vector<K> keys_;
  std::vector<std::uint64_t> hashes_;
  Buffer states_;
};

template <class K>
struct AggLocal final : SinkState {
  AggLocal(const AggLayout* layout, bool direct, std::size_t bits) : table(layout, direct, bits) {}
  AggTable<K> table;
  std::vector<K> keys = std::vector<K>(kVectorSize);
  std::vector<std::uint32_t> gids = std::vector<std::uint32_t>(kVectorSize);
  std::vector<std::vector<std::uint32_t>> partitions;
};

// Calls f(state_slot, value) for every selected row; gids are in selection order.
template <class T, class F>
inline void update_each(std::uint8_t* states, std::size_t stride, std::size_t off, const Sel& sel,
                        const std::uint32_t* gids, const T* values, F&& f) {
  std::uint32_t j = 0;
  for_each_selected(sel, [&](std::uint32_t i) { f(states + gids[j++] * stride + off, values[i]); });
}

// Integer inputs widened to int64 for min/max state.
template <class F>
void with_int_input(const Type& t, const void* col, F&& f) {
  switch (t.id) {
    case TypeId::Int64:
      f(static_cast<const std::int64_t*>(col));
      break;
    case TypeId::Dict:
      f(static_cast<const std::uint8_t*>(col));
      break;
    case TypeId::Int128:
      f(static_cast<const int128*>(col));
      break;
    default:
      f(static_cast<const std::int32_t*>(col));
  }
}

template <class K>
class AggregateSink final : public Sink {
 public:
  explicit AggregateSink(const PlanNode& node) : layout_(node.aggs), schema_(node.schema) {
    const Schema& in = node.children[0]->schema;
    keys_ = KeyLayout::make(in, node.group_cols);
    direct_ = keys_.parts.size() <= 2;
    for (const auto& p : keys_.parts) direct_ = direct_ && p.type == TypeId::Dict;
    direct_bits_ = 8 * keys_.parts.size();
  }

  std::unique_ptr<SinkState> make_local() const override {
    return std::make_unique<AggLocal<K>>(&layout_, direct_, direct_bits_);
  }

  void consume(Chunk& chunk, SinkState& state) const override {
    auto& local = static_cast<AggLocal<K>&>(state);
    AggTable<K>& table = local.table;
    const Sel& sel = chunk.sel;
    std::uint32_t* gids = local.gids.data();
    if (keys_.parts.empty()) {
      const std::uint32_t g = table.find_or_insert(K{});
      std::fill(gids, gids + sel.count, g);
    } else {
      pack_keys(keys_, chunk, sel, local.keys.data());
      std::uint32_t j = 0;
      for_each_selected(sel, [&](std::uint32_t i) { gids[j++] = table.find_or_insert(local.keys[i]); });
    }
    if (table.groups() == 0) return;
    std::uint8_t* states = table.state(0);
    const std::size_t stride = layout_.stride;

    for (std::size_t a = 0; a < layout_.aggs.size(); ++a) {
      const BoundAgg& agg = layout_.aggs[a];
      const std::size_t off = layout_.offset[a];
      const void* col = agg.input >= 0 ? chunk.cols[static_cast<std::size_t>(agg.input)] : nullptr;
      const bool is_double = agg.input_type.id == TypeId::Double;
      switch (agg.kind) {
        case AggKind::Count: {
          std::uint32_t j = 0;
          for (std::uint32_t k = 0; k < sel.count; ++k) at<std::int64_t>(states + gids[j++] * stride + off) += 1;
          break;
        }
        case AggKind::Sum:
        case AggKind::Avg: {
          const bool avg = agg.kind == AggKind::Avg;
          if (is_double) {
            update_each(states, stride, off, sel, gids, static_cast<const double*>(col),
                        [avg](std::uint8_t* p, double v) {
                          at<double>(p) += v;
                          if (avg) at<std::int64_t>(p + 8) += 1;
                        });
          } else {
            with_int_input(agg.input_type, col, [&](const auto* v) {
              update_each(states, stride, off, sel, gids, v, [avg](std::uint8_t* p, auto x) {
                at<int128>(p) += static_cast<int128>(x);
                if (avg) at<std::int64_t>(p + 16) += 1;
              });
            });
          }
          break;
        }
        case AggKind::Min:
        case AggKind::Max: {
          const bool is_min = agg.kind == AggKind::Min;
          if (is_double) {
            update_each(states, stride, off, sel, gids, static_cast<const double*>(col),
                        [is_min](std::uint8_t* p, double v) {
                          double& s = at<double>(p);
                          s = is_min ? std::min(s, v) : std::max(s, v);
                        });
          } else {
            with_int_input(agg.input_type, col, [&](const auto* v) {
              update_each(states, stride, off, sel, gids, v, [is_min](std::uint8_t* p, auto x) {
                std::int64_t& s = at<std::int64_t>(p);
                const auto y = static_cast<std::int64_t>(x);
                s = is_min ? std::min(s, y) : std::max(s, y);
              });
            });
          }
          break;
        }
        case AggKind::Any: {
          const std::size_t w = type_width(agg.input_type.id);
          const auto* src = static_cast<const std::uint8_t*>(col);
          std::uint32_t j = 0;
          for_each_selected(sel, [&](std::uint32_t i) {
            std::memcpy(states + gids[j++] * stride + off, src + i * w, w);
          });
          break;
        }
      }
    }
  }

  void finalize(std::vector<std::unique_ptr<SinkState>>& locals, ThreadPool& pool) override {
    std::vector<AggTable<K>*> tables;
    std::size_t total = 0;
    for (auto& l : locals) {
      tables.push_back(&static_cast<AggLocal<K>&>(*l).table);
      total += tables.back()->groups();
    }
    // An aggregate without GROUP BY returns one row even for empty input.
    if (keys_.parts.empty()) tables[0]->find_or_insert(K{});

    if (tables.size() == 1 || total < kParallelMergeGroups) {
      AggTable<K>& dst = *tables[0];
      for (std::size_t t = 1; t < tables.size(); ++t) {
        AggTable<K>& src = *tables[t];
        for (std::uint32_t g = 0; g < src.groups(); ++g) {
          merge_state(layout_, dst.state(dst.find_or_insert(src.key(g), src.hash(g))), src.state(g));
        }
      }
      allocate_result(dst.groups());
      write_rows(dst, 0);
      return;
    }

    // Phase 1: each worker sorts its own groups into partitions by hash.
    pool.parallel_for(locals.size(), [&](std::size_t l) {
      auto& local = static_cast<AggLocal<K>&>(*locals[l]);
      local.partitions.assign(kPartitions, {});
      for (std::uint32_t g = 0; g < local.table.groups(); ++g) {
        local.partitions[local.table.hash(g) >> (64 - kPartitionBits)].push_back(g);
      }
    });
    // Phase 2: one worker per partition merges it from every local table.
    std::vector<std::unique_ptr<AggTable<K>>> merged(kPartitions);
    pool.parallel_for(kPartitions, [&](std::size_t p) {
      auto dst = std::make_unique<AggTable<K>>(&layout_, direct_, direct_bits_);
      for (auto& l : locals) {
        auto& local = static_cast<AggLocal<K>&>(*l);
        for (std::uint32_t g : local.partitions[p]) {
          const std::uint32_t d = dst->find_or_insert(local.table.key(g), local.table.hash(g));
          merge_state(layout_, dst->state(d), local.table.state(g));
        }
      }
      merged[p] = std::move(dst);
    });
    // Phase 3: partitions are written to consecutive row ranges in parallel.
    std::vector<std::size_t> start(kPartitions + 1, 0);
    for (std::size_t p = 0; p < kPartitions; ++p) start[p + 1] = start[p] + merged[p]->groups();
    allocate_result(start.back());
    pool.parallel_for(kPartitions, [&](std::size_t p) { write_rows(*merged[p], start[p]); });
  }

 private:
  void allocate_result(std::size_t rows) {
    result.names = schema_.names;
    result.types = schema_.types;
    result.rows = rows;
    result.columns.clear();
    for (const auto& t : schema_.types) result.columns.emplace_back(rows * type_width(t.id));
  }

  void write_rows(const AggTable<K>& table, std::size_t first) {
    const std::size_t nkeys = keys_.parts.size();
    for (std::uint32_t g = 0; g < table.groups(); ++g) {
      const std::size_t row = first + g;
      for (std::size_t k = 0; k < nkeys; ++k) unpack_key_part(keys_.parts[k], table.key(g), result.columns[k].data(), row);
      const std::uint8_t* st = table.state(g);
      for (std::size_t a = 0; a < layout_.aggs.size(); ++a) {
        const BoundAgg& agg = layout_.aggs[a];
        const std::uint8_t* p = st + layout_.offset[a];
        std::uint8_t* out = result.columns[nkeys + a].data();
        const bool is_double = agg.input_type.id == TypeId::Double;
        switch (agg.kind) {
          case AggKind::Count:
            reinterpret_cast<std::int64_t*>(out)[row] = at<std::int64_t>(p);
            break;
          case AggKind::Sum:
            if (is_double) reinterpret_cast<double*>(out)[row] = at<double>(p);
            else reinterpret_cast<int128*>(out)[row] = at<int128>(p);
            break;
          case AggKind::Avg: {
            double v;
            if (is_double) {
              v = at<double>(p) / static_cast<double>(at<std::int64_t>(p + 8));
            } else {
              v = static_cast<double>(at<int128>(p)) / static_cast<double>(at<std::int64_t>(p + 16)) /
                  static_cast<double>(pow10_i64(agg.input_type.scale));
            }
            reinterpret_cast<double*>(out)[row] = v;
            break;
          }
          case AggKind::Min:
          case AggKind::Max:
            if (is_double) {
              reinterpret_cast<double*>(out)[row] = at<double>(p);
            } else {
              const std::int64_t v = at<std::int64_t>(p);
              switch (agg.output_type.id) {
                case TypeId::Int64:
                  reinterpret_cast<std::int64_t*>(out)[row] = v;
                  break;
                case TypeId::Dict:
                  out[row] = static_cast<std::uint8_t>(v);
                  break;
                default:
                  reinterpret_cast<std::int32_t*>(out)[row] = static_cast<std::int32_t>(v);
              }
            }
            break;
          case AggKind::Any: {
            const std::size_t w = type_width(agg.output_type.id);
            std::memcpy(out + row * w, p, w);
            break;
          }
        }
      }
    }
  }

  AggLayout layout_;
  Schema schema_;
  KeyLayout keys_;
  bool direct_ = false;
  std::size_t direct_bits_ = 0;
};

}  // namespace

std::shared_ptr<Sink> make_aggregate(const PlanNode& node) {
  const KeyLayout keys = KeyLayout::make(node.children[0]->schema, node.group_cols);
  if (keys.wide) return std::make_shared<AggregateSink<Key128>>(node);
  return std::make_shared<AggregateSink<std::uint64_t>>(node);
}

}  // namespace morsel::detail
