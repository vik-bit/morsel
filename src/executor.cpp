// Pipeline construction and morsel-driven execution, plus the simple
// operators: filter, projection, result collection and sort.

#include "morsel/executor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <stdexcept>

#include "internal.h"

namespace morsel {

namespace detail {

KeyLayout KeyLayout::make(const Schema& schema, const std::vector<int>& columns) {
  KeyLayout l;
  std::vector<std::size_t> order(columns.size());
  std::iota(order.begin(), order.end(), 0);
  auto width = [&](std::size_t k) { return type_width(schema.types[static_cast<std::size_t>(columns[k])].id); };
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return width(a) > width(b); });
  l.parts.resize(columns.size());
  std::size_t offset = 0;
  for (std::size_t k : order) {
    l.parts[k] = {columns[k], schema.types[static_cast<std::size_t>(columns[k])].id,
                  static_cast<int>(offset / 8), static_cast<int>((offset % 8) * 8)};
    offset += width(k);
  }
  if (offset > 16) throw std::invalid_argument("keys wider than 16 bytes");
  l.wide = offset > 8;
  return l;
}

template <class T>
void gather_typed(const void* src, const std::uint32_t* idx, std::uint32_t n, void* dst) {
  const T* s = static_cast<const T*>(src);
  T* d = static_cast<T*>(dst);
  for (std::uint32_t j = 0; j < n; ++j) d[j] = s[idx[j]];
}

struct Bytes16 {
  std::uint64_t a, b;
};

void gather(const void* src, std::size_t width, const std::uint32_t* idx, std::uint32_t n, void* dst) {
  switch (width) {
    case 1:
      gather_typed<std::uint8_t>(src, idx, n, dst);
      break;
    case 4:
      gather_typed<std::uint32_t>(src, idx, n, dst);
      break;
    case 8:
      gather_typed<std::uint64_t>(src, idx, n, dst);
      break;
    default:
      gather_typed<Bytes16>(src, idx, n, dst);
  }
}

void append_selected(Buffer& dst, std::size_t rows, const void* src, std::size_t width, const Sel& sel) {
  dst.resize((rows + sel.count) * width);
  std::uint8_t* out = dst.data() + rows * width;
  if (!sel.idx) {
    std::memcpy(out, src, sel.count * width);
    return;
  }
  gather(src, width, sel.idx, sel.count, out);
}

namespace {

class FilterOp final : public Operator {
 public:
  explicit FilterOp(const PlanNode& node) : pred_(node.pred.get()) {}
  struct State final : OpState {
    std::unique_ptr<PredState> pred;
    Buffer sel{kSelCapacity * sizeof(std::uint32_t)};
  };
  std::unique_ptr<OpState> make_state() const override {
    auto st = std::make_unique<State>();
    st->pred = pred_->make_state();
    return st;
  }
  void execute(Chunk& in, OpState& state, Emitter& next) const override {
    auto& st = static_cast<State&>(state);
    auto* sel = st.sel.as<std::uint32_t>();
    const std::uint32_t n = pred_->select(in, in.sel, sel, *st.pred);
    if (n == 0) return;
    in.sel = {sel, n};
    next.emit(in);
  }

 private:
  const BoundPred* pred_;
};

class ProjectOp final : public Operator {
 public:
  explicit ProjectOp(const PlanNode& node) {
    for (const auto& e : node.exprs) exprs_.push_back(e.get());
  }
  struct State final : OpState {
    std::vector<std::unique_ptr<ExprState>> exprs;
    Chunk out;
  };
  std::unique_ptr<OpState> make_state() const override {
    auto st = std::make_unique<State>();
    for (const auto* e : exprs_) st->exprs.push_back(e->make_state());
    st->out.cols.resize(exprs_.size());
    return st;
  }
  void execute(Chunk& in, OpState& state, Emitter& next) const override {
    auto& st = static_cast<State&>(state);
    st.out.size = in.size;
    st.out.sel = in.sel;
    for (std::size_t e = 0; e < exprs_.size(); ++e) st.out.cols[e] = exprs_[e]->eval(in, in.sel, *st.exprs[e]);
    next.emit(st.out);
  }

 private:
  std::vector<const BoundExpr*> exprs_;
};

struct CollectLocal final : SinkState {
  std::vector<Buffer> columns;
  std::size_t rows = 0;
};

class CollectSink : public Sink {
 public:
  explicit CollectSink(const Schema& schema) : schema_(schema) {
    for (const auto& t : schema.types) widths_.push_back(type_width(t.id));
  }
  std::unique_ptr<SinkState> make_local() const override {
    auto local = std::make_unique<CollectLocal>();
    local->columns.resize(widths_.size());
    return local;
  }
  void consume(Chunk& chunk, SinkState& state) const override {
    auto& local = static_cast<CollectLocal&>(state);
    for (std::size_t c = 0; c < widths_.size(); ++c) {
      append_selected(local.columns[c], local.rows, chunk.cols[c], widths_[c], chunk.sel);
    }
    local.rows += chunk.sel.count;
  }
  void finalize(std::vector<std::unique_ptr<SinkState>>& locals, ThreadPool&) override {
    result.names = schema_.names;
    result.types = schema_.types;
    result.rows = 0;
    for (auto& l : locals) result.rows += static_cast<CollectLocal&>(*l).rows;
    result.columns.clear();
    for (std::size_t c = 0; c < widths_.size(); ++c) {
      result.columns.emplace_back(result.rows * widths_[c]);
      std::size_t row = 0;
      for (auto& l : locals) {
        auto& local = static_cast<CollectLocal&>(*l);
        if (local.rows) std::memcpy(result.columns[c].data() + row * widths_[c], local.columns[c].data(), local.rows * widths_[c]);
        row += local.rows;
      }
    }
  }

 protected:
  Schema schema_;
  std::vector<std::size_t> widths_;
};

template <class T>
int three_way(const void* col, std::size_t a, std::size_t b) {
  const T x = static_cast<const T*>(col)[a], y = static_cast<const T*>(col)[b];
  return x < y ? -1 : (y < x ? 1 : 0);
}

// Collects every row, then sorts a permutation; with a LIMIT only the first
// `limit` positions are ordered (std::partial_sort, a heap-based top-N).
class SortSink final : public CollectSink {
 public:
  explicit SortSink(const PlanNode& node)
      : CollectSink(node.schema), cols_(node.sort_cols), desc_(node.sort_desc), limit_(node.limit) {}

  void finalize(std::vector<std::unique_ptr<SinkState>>& locals, ThreadPool& pool) override {
    CollectSink::finalize(locals, pool);
    MaterializedTable all = std::move(result);
    std::vector<std::uint32_t> perm(all.rows);
    std::iota(perm.begin(), perm.end(), 0u);
    auto less = [&](std::uint32_t a, std::uint32_t b) {
      for (std::size_t k = 0; k < cols_.size(); ++k) {
        const auto c = static_cast<std::size_t>(cols_[k]);
        const void* col = all.columns[c].data();
        int r = 0;
        switch (all.types[c].id) {
          case TypeId::Int32:
          case TypeId::Date:
            r = three_way<std::int32_t>(col, a, b);
            break;
          case TypeId::Int64:
            r = three_way<std::int64_t>(col, a, b);
            break;
          case TypeId::Int128:
            r = three_way<int128>(col, a, b);
            break;
          case TypeId::Double:
            r = three_way<double>(col, a, b);
            break;
          case TypeId::Dict:  // sorted dictionary: code order is string order
            r = three_way<std::uint8_t>(col, a, b);
            break;
          case TypeId::String:
            r = three_way<std::string_view>(col, a, b);
            break;
        }
        if (r != 0) return desc_[k] ? r > 0 : r < 0;
      }
      return a < b;  // stable for equal keys
    };
    const std::size_t n = limit_ ? std::min(limit_, all.rows) : all.rows;
    if (n < all.rows) {
      std::partial_sort(perm.begin(), perm.begin() + static_cast<std::ptrdiff_t>(n), perm.end(), less);
    } else {
      std::sort(perm.begin(), perm.end(), less);
    }
    result.names = all.names;
    result.types = all.types;
    result.rows = n;
    result.columns.clear();
    for (std::size_t c = 0; c < all.columns.size(); ++c) {
      result.columns.emplace_back(n * widths_[c]);
      gather(all.columns[c].data(), widths_[c], perm.data(), static_cast<std::uint32_t>(n), result.columns[c].data());
    }
  }

 private:
  std::vector<int> cols_;
  std::vector<bool> desc_;
  std::size_t limit_;
};

}  // namespace

std::unique_ptr<Operator> make_filter(const PlanNode& node) { return std::make_unique<FilterOp>(node); }
std::unique_ptr<Operator> make_project(const PlanNode& node) { return std::make_unique<ProjectOp>(node); }
std::shared_ptr<Sink> make_collect(const Schema& schema) { return std::make_shared<CollectSink>(schema); }
std::shared_ptr<Sink> make_sort(const PlanNode& node) { return std::make_shared<SortSink>(node); }

}  // namespace detail

namespace {

using namespace detail;

struct Pipeline {
  std::string name;
  std::function<Relation()> source;  // called when the pipeline starts
  std::vector<std::unique_ptr<Operator>> ops;
  std::shared_ptr<Sink> sink;
};

// Cuts the plan into pipelines. Pipelines are appended to `done` in an order
// where every pipeline comes after the ones it depends on.
class Compiler {
 public:
  std::vector<std::unique_ptr<Pipeline>> done;

  std::unique_ptr<Pipeline> compile(const PlanNode& n) {
    switch (n.kind) {
      case PlanNode::Scan: {
        auto p = std::make_unique<Pipeline>();
        p->name = n.label;
        p->source = [rel = n.relation] { return rel; };
        return p;
      }
      case PlanNode::Filter: {
        auto p = compile(*n.children[0]);
        p->ops.push_back(make_filter(n));
        return p;
      }
      case PlanNode::Project: {
        auto p = compile(*n.children[0]);
        p->ops.push_back(make_project(n));
        return p;
      }
      case PlanNode::Join: {
        auto build = compile(*n.children[1]);
        build->sink = make_join_build(n);
        build->name += " -> hash build";
        const auto sink = build->sink;
        done.push_back(std::move(build));
        auto p = compile(*n.children[0]);
        p->ops.push_back(make_join_probe(n, sink));
        p->name += " -> probe";
        return p;
      }
      case PlanNode::Aggregate:
      case PlanNode::OrderBy: {
        const bool agg = n.kind == PlanNode::Aggregate;
        auto input = compile(*n.children[0]);
        input->sink = agg ? make_aggregate(n) : make_sort(n);
        input->name += agg ? " -> aggregate" : " -> sort";
        const auto sink = input->sink;
        done.push_back(std::move(input));
        auto p = std::make_unique<Pipeline>();
        p->name = agg ? "aggregate result" : "sorted result";
        p->source = [sink] { return sink->result.view(); };
        p->sink = sink;  // remembered so the root can take its result
        return p;
      }
    }
    throw std::logic_error("unhandled plan node");
  }
};

class OpStage final : public Emitter {
 public:
  OpStage(const Operator* op, OpState* state, Emitter* next) : op_(op), state_(state), next_(next) {}
  void emit(Chunk& chunk) override { op_->execute(chunk, *state_, *next_); }

 private:
  const Operator* op_;
  OpState* state_;
  Emitter* next_;
};

class SinkStage final : public Emitter {
 public:
  SinkStage(const Sink* sink, SinkState* local) : sink_(sink), local_(local) {}
  void emit(Chunk& chunk) override {
    if (chunk.sel.count) sink_->consume(chunk, *local_);
  }

 private:
  const Sink* sink_;
  SinkState* local_;
};

void run_pipeline(Pipeline& p, ThreadPool& pool) {
  const Relation rel = p.source();
  std::vector<std::size_t> widths;
  for (const auto& t : rel.types) widths.push_back(type_width(t.id));
  std::vector<std::unique_ptr<SinkState>> locals;
  for (int t = 0; t < pool.size(); ++t) locals.push_back(p.sink->make_local());

  std::atomic<std::size_t> next_morsel{0};
  pool.run([&](int worker) {
    std::vector<std::unique_ptr<OpState>> states;
    for (const auto& op : p.ops) states.push_back(op->make_state());
    std::vector<std::unique_ptr<Emitter>> stages;
    stages.push_back(std::make_unique<SinkStage>(p.sink.get(), locals[static_cast<std::size_t>(worker)].get()));
    for (std::size_t i = p.ops.size(); i-- > 0;) {
      stages.push_back(std::make_unique<OpStage>(p.ops[i].get(), states[i].get(), stages.back().get()));
    }
    Emitter& first = *stages.back();

    Chunk chunk;
    chunk.cols.resize(rel.data.size());
    for (;;) {
      const std::size_t begin = next_morsel.fetch_add(kMorselRows, std::memory_order_relaxed);
      if (begin >= rel.rows) break;
      const std::size_t end = std::min(rel.rows, begin + kMorselRows);
      for (std::size_t v = begin; v < end; v += kVectorSize) {
        const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(kVectorSize, end - v));
        for (std::size_t c = 0; c < rel.data.size(); ++c) {
          chunk.cols[c] = static_cast<const std::uint8_t*>(rel.data[c]) + v * widths[c];
        }
        chunk.size = n;
        chunk.sel = {nullptr, n};
        first.emit(chunk);
      }
    }
  });
  p.sink->finalize(locals, pool);
}

}  // namespace

Executor::Executor(int threads)
    : pool_(threads > 0 ? threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))) {}

MaterializedTable Executor::run(const Plan& plan) {
  Compiler compiler;
  auto root = compiler.compile(*plan.node);
  std::shared_ptr<Sink> result_sink;
  if (root->ops.empty() && root->sink) {
    result_sink = root->sink;  // the plan ends in an aggregation or sort
  } else {
    root->sink = make_collect(plan.schema());
    root->name += " -> collect";
    result_sink = root->sink;
    compiler.done.push_back(std::move(root));
  }
  profile_.clear();
  for (auto& p : compiler.done) {
    const auto t0 = std::chrono::steady_clock::now();
    run_pipeline(*p, pool_);
    const auto t1 = std::chrono::steady_clock::now();
    profile_.push_back({p->name, std::chrono::duration<double, std::milli>(t1 - t0).count()});
  }
  return std::move(result_sink->result);
}

}  // namespace morsel
