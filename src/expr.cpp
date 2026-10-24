#include "morsel/expr.h"

#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

namespace morsel {

int Schema::index_of(const std::string& name) const {
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) return static_cast<int>(i);
  }
  return -1;
}

int Schema::require(const std::string& name) const {
  const int i = index_of(name);
  if (i < 0) {
    std::string known;
    for (const auto& n : names) known += (known.empty() ? "" : ", ") + n;
    throw std::invalid_argument("unknown column '" + name + "' (have: " + known + ")");
  }
  return i;
}

// ---------------------------------------------------------------------------
// DSL

namespace {

std::shared_ptr<ExprAst> make_expr(ExprAst::Kind kind) {
  auto e = std::make_shared<ExprAst>();
  e->kind = kind;
  return e;
}

E binary(ExprAst::Kind kind, const E& a, const E& b) {
  auto e = make_expr(kind);
  e->args = {a.ast, b.ast};
  return E(e);
}

P compare(CmpOp op, const E& a, const E& b) {
  auto p = std::make_shared<PredAst>();
  p->kind = PredAst::Cmp;
  p->op = op;
  p->args = {a.ast, b.ast};
  return P(p);
}

P combine(PredAst::Kind kind, const P& a, const P& b) {
  auto p = std::make_shared<PredAst>();
  p->kind = kind;
  // Flatten nested ANDs/ORs so a chain of && is one conjunction.
  for (const auto& side : {a.ast, b.ast}) {
    if (side->kind == kind) {
      p->children.insert(p->children.end(), side->children.begin(), side->children.end());
    } else {
      p->children.push_back(side);
    }
  }
  return P(p);
}

}  // namespace

E::E(int value) : E(static_cast<std::int64_t>(value)) {}
E::E(std::int64_t value) : ast(make_expr(ExprAst::IntLit)) { ast->ivalue = value; }

E col(const std::string& name) {
  auto e = make_expr(ExprAst::Column);
  e->text = name;
  return E(e);
}

E dec(const std::string& text) {
  const DecimalLiteral d = parse_decimal(text);
  auto e = make_expr(ExprAst::DecLit);
  e->ivalue = d.value;
  e->scale = d.scale;
  return E(e);
}

E date(const std::string& text) {
  auto e = make_expr(ExprAst::DateLit);
  e->ivalue = parse_date(text);
  return E(e);
}

E str(const std::string& text) {
  auto e = make_expr(ExprAst::StrLit);
  e->text = text;
  return E(e);
}

E dbl(double value) {
  auto e = make_expr(ExprAst::DoubleLit);
  e->dvalue = value;
  return E(e);
}

E case_when(const P& cond, const E& then_value, const E& else_value) {
  auto e = make_expr(ExprAst::Case);
  e->cond = cond.ast;
  e->args = {then_value.ast, else_value.ast};
  return E(e);
}

E operator+(const E& a, const E& b) { return binary(ExprAst::Add, a, b); }
E operator-(const E& a, const E& b) { return binary(ExprAst::Sub, a, b); }
E operator*(const E& a, const E& b) { return binary(ExprAst::Mul, a, b); }
E operator/(const E& a, const E& b) { return binary(ExprAst::Div, a, b); }

P operator==(const E& a, const E& b) { return compare(CmpOp::Eq, a, b); }
P operator!=(const E& a, const E& b) { return compare(CmpOp::Ne, a, b); }
P operator<(const E& a, const E& b) { return compare(CmpOp::Lt, a, b); }
P operator<=(const E& a, const E& b) { return compare(CmpOp::Le, a, b); }
P operator>(const E& a, const E& b) { return compare(CmpOp::Gt, a, b); }
P operator>=(const E& a, const E& b) { return compare(CmpOp::Ge, a, b); }
P operator&&(const P& a, const P& b) { return combine(PredAst::And, a, b); }
P operator||(const P& a, const P& b) { return combine(PredAst::Or, a, b); }

P between(const E& x, const E& lo, const E& hi) {
  auto p = std::make_shared<PredAst>();
  p->kind = PredAst::Between;
  p->args = {x.ast, lo.ast, hi.ast};
  return P(p);
}

P in(const E& x, const std::vector<std::string>& values) {
  auto p = std::make_shared<PredAst>();
  p->kind = PredAst::In;
  p->args = {x.ast};
  p->strings = values;
  return P(p);
}

P not_in(const E& x, const std::vector<std::string>& values) {
  P p = in(x, values);
  p.ast->negated = true;
  return p;
}

P like(const E& x, const std::string& pattern) {
  auto p = std::make_shared<PredAst>();
  p->kind = PredAst::Like;
  p->args = {x.ast};
  p->strings = {pattern};
  return P(p);
}

bool like_match(std::string_view s, std::string_view p) {
  // Iterative matcher with backtracking to the last '%': O(|s| * |p|) worst case.
  std::size_t si = 0, pi = 0, star = std::string_view::npos, mark = 0;
  while (si < s.size()) {
    if (pi < p.size() && (p[pi] == '_' || p[pi] == s[si])) {
      ++si;
      ++pi;
    } else if (pi < p.size() && p[pi] == '%') {
      star = pi++;
      mark = si;
    } else if (star != std::string_view::npos) {
      pi = star + 1;
      si = ++mark;
    } else {
      return false;
    }
  }
  while (pi < p.size() && p[pi] == '%') ++pi;
  return pi == p.size();
}

// ---------------------------------------------------------------------------
// Bound expressions

std::unique_ptr<ExprState> BoundExpr::make_state() const {
  auto state = std::make_unique<ExprState>();
  state->out.resize(static_cast<std::size_t>(kVectorSize) * 16);
  for (const auto& child : children) state->children.push_back(child->make_state());
  return state;
}

std::unique_ptr<PredState> BoundPred::make_state() const {
  auto state = std::make_unique<PredState>();
  for (const auto& e : exprs) state->exprs.push_back(e->make_state());
  for (const auto& c : children) state->children.push_back(c->make_state());
  return state;
}

namespace {

class ColumnRef final : public BoundExpr {
 public:
  ColumnRef(int index, Type t) : BoundExpr(t), index_(index) {}
  const void* eval(const Chunk& chunk, const Sel&, ExprState&) const override {
    return chunk.cols[static_cast<std::size_t>(index_)];
  }
  std::unique_ptr<ExprState> make_state() const override { return std::make_unique<ExprState>(); }
  int column_index() const override { return index_; }

 private:
  int index_;
};

// A literal used as a full vector (e.g. the 1 in CASE ... THEN 1 ELSE 0).
class ConstVector final : public BoundExpr {
 public:
  ConstVector(Type t, std::int64_t i, double d) : BoundExpr(t), i_(i), d_(d) {}
  const void* eval(const Chunk&, const Sel&, ExprState& state) const override {
    return state.out.data();
  }
  std::unique_ptr<ExprState> make_state() const override {
    auto state = std::make_unique<ExprState>();
    state->out.resize(static_cast<std::size_t>(kVectorSize) * 8);
    for (std::uint32_t k = 0; k < kVectorSize; ++k) {
      switch (type.id) {
        case TypeId::Double:
          state->out.as<double>()[k] = d_;
          break;
        case TypeId::Int32:
        case TypeId::Date:
          state->out.as<std::int32_t>()[k] = static_cast<std::int32_t>(i_);
          break;
        default:
          state->out.as<std::int64_t>()[k] = i_;
      }
    }
    return state;
  }

 private:
  std::int64_t i_;
  double d_;
};

template <class From>
void cast_to_i64(const From* in, std::int64_t mul, std::int64_t* out, const Chunk& chunk,
                 const Sel& sel) {
  for_each_computed(chunk, sel, [&](std::uint32_t i) {
    out[i] = static_cast<std::int64_t>(in[i]) * mul;
  });
}

template <class From>
void cast_to_double(const From* in, double div, double* out, const Chunk& chunk, const Sel& sel) {
  for_each_computed(chunk, sel, [&](std::uint32_t i) { out[i] = static_cast<double>(in[i]) / div; });
}

class CastExpr final : public BoundExpr {
 public:
  CastExpr(std::unique_ptr<BoundExpr> child, Type target) : BoundExpr(target) {
    children.push_back(std::move(child));
  }
  const void* eval(const Chunk& chunk, const Sel& sel, ExprState& state) const override {
    const BoundExpr& child = *children[0];
    const void* in = child.eval(chunk, sel, *state.children[0]);
    const Type from = child.type;
    if (type.id == TypeId::Double) {
      auto* out = state.out.as<double>();
      const double div = static_cast<double>(pow10_i64(from.scale));
      switch (from.id) {
        case TypeId::Int32:
          cast_to_double(static_cast<const std::int32_t*>(in), div, out, chunk, sel);
          break;
        case TypeId::Int64:
          cast_to_double(static_cast<const std::int64_t*>(in), div, out, chunk, sel);
          break;
        case TypeId::Int128:
          cast_to_double(static_cast<const int128*>(in), div, out, chunk, sel);
          break;
        default:
          return in;  // already double
      }
      return out;
    }
    // Integer to int64 at an equal or larger scale.
    auto* out = state.out.as<std::int64_t>();
    const std::int64_t mul = pow10_i64(type.scale - from.scale);
    if (from.id == TypeId::Int32) {
      cast_to_i64(static_cast<const std::int32_t*>(in), mul, out, chunk, sel);
    } else if (compute_dense(chunk, sel)) {
      active_kernels().mul_const_i64(static_cast<const std::int64_t*>(in), mul, out, chunk.size);
    } else {
      cast_to_i64(static_cast<const std::int64_t*>(in), mul, out, chunk, sel);
    }
    return out;
  }
};

enum class ArithOp { Add, Sub, Mul, Div };

class ArithI64 final : public BoundExpr {
 public:
  ArithI64(ArithOp op, std::unique_ptr<BoundExpr> a, std::unique_ptr<BoundExpr> b, Type t)
      : BoundExpr(t), op_(op) {
    children.push_back(std::move(a));
    children.push_back(std::move(b));
  }
  const void* eval(const Chunk& chunk, const Sel& sel, ExprState& state) const override {
    const auto* a = static_cast<const std::int64_t*>(children[0]->eval(chunk, sel, *state.children[0]));
    const auto* b = static_cast<const std::int64_t*>(children[1]->eval(chunk, sel, *state.children[1]));
    auto* out = state.out.as<std::int64_t>();
    if (compute_dense(chunk, sel)) {
      const Kernels& k = active_kernels();
      switch (op_) {
        case ArithOp::Add:
          k.add_i64(a, b, out, chunk.size);
          break;
        case ArithOp::Sub:
          k.sub_i64(a, b, out, chunk.size);
          break;
        default:
          k.mul_i64(a, b, out, chunk.size);
      }
    } else {
      for_each_selected(sel, [&](std::uint32_t i) {
        const auto x = static_cast<std::uint64_t>(a[i]), y = static_cast<std::uint64_t>(b[i]);
        out[i] = static_cast<std::int64_t>(op_ == ArithOp::Add ? x + y : op_ == ArithOp::Sub ? x - y : x * y);
      });
    }
    return out;
  }

 private:
  ArithOp op_;
};

// x op c, or c op x when const_left.
class ArithConstI64 final : public BoundExpr {
 public:
  ArithConstI64(ArithOp op, std::unique_ptr<BoundExpr> x, std::int64_t c, bool const_left, Type t)
      : BoundExpr(t), op_(op), c_(c), const_left_(const_left) {
    children.push_back(std::move(x));
  }
  const void* eval(const Chunk& chunk, const Sel& sel, ExprState& state) const override {
    const auto* x = static_cast<const std::int64_t*>(children[0]->eval(chunk, sel, *state.children[0]));
    auto* out = state.out.as<std::int64_t>();
    const bool rsub = op_ == ArithOp::Sub && const_left_;
    // x - c is x + (-c).
    const std::int64_t c = (op_ == ArithOp::Sub && !const_left_)
                               ? static_cast<std::int64_t>(0ull - static_cast<std::uint64_t>(c_))
                               : c_;
    if (compute_dense(chunk, sel)) {
      const Kernels& k = active_kernels();
      if (op_ == ArithOp::Mul) {
        k.mul_const_i64(x, c, out, chunk.size);
      } else if (rsub) {
        k.rsub_const_i64(x, c, out, chunk.size);
      } else {
        k.add_const_i64(x, c, out, chunk.size);
      }
    } else {
      const auto uc = static_cast<std::uint64_t>(c);
      for_each_selected(sel, [&](std::uint32_t i) {
        const auto v = static_cast<std::uint64_t>(x[i]);
        out[i] = static_cast<std::int64_t>(op_ == ArithOp::Mul ? v * uc : rsub ? uc - v : v + uc);
      });
    }
    return out;
  }

 private:
  ArithOp op_;
  std::int64_t c_;
  bool const_left_;
};

class ArithF64 final : public BoundExpr {
 public:
  ArithF64(ArithOp op, std::unique_ptr<BoundExpr> a, std::unique_ptr<BoundExpr> b)
      : BoundExpr(Type::float64()), op_(op) {
    children.push_back(std::move(a));
    children.push_back(std::move(b));
  }
  const void* eval(const Chunk& chunk, const Sel& sel, ExprState& state) const override {
    const auto* a = static_cast<const double*>(children[0]->eval(chunk, sel, *state.children[0]));
    const auto* b = static_cast<const double*>(children[1]->eval(chunk, sel, *state.children[1]));
    auto* out = state.out.as<double>();
    for_each_computed(chunk, sel, [&](std::uint32_t i) {
      switch (op_) {
        case ArithOp::Add:
          out[i] = a[i] + b[i];
          break;
        case ArithOp::Sub:
          out[i] = a[i] - b[i];
          break;
        case ArithOp::Mul:
          out[i] = a[i] * b[i];
          break;
        default:
          out[i] = a[i] / b[i];
      }
    });
    return out;
  }

 private:
  ArithOp op_;
};

class CaseExpr final : public BoundExpr {
 public:
  CaseExpr(std::unique_ptr<BoundPred> cond, std::unique_ptr<BoundExpr> then_value,
           std::unique_ptr<BoundExpr> else_value)
      : BoundExpr(then_value->type), cond_(std::move(cond)) {
    children.push_back(std::move(then_value));
    children.push_back(std::move(else_value));
  }
  std::unique_ptr<ExprState> make_state() const override {
    auto state = BoundExpr::make_state();
    auto pred = cond_->make_state();
    pred->bufs.emplace_back(kSelCapacity * sizeof(std::uint32_t));
    state->preds.push_back(std::move(pred));
    return state;
  }
  const void* eval(const Chunk& chunk, const Sel& sel, ExprState& state) const override {
    const void* t = children[0]->eval(chunk, sel, *state.children[0]);
    const void* e = children[1]->eval(chunk, sel, *state.children[1]);
    PredState& ps = *state.preds[0];
    auto* hits = ps.bufs.back().as<std::uint32_t>();
    const std::uint32_t n = cond_->select(chunk, sel, hits, ps);
    const std::size_t w = type_width(type.id);
    auto* out = state.out.data();
    for_each_computed(chunk, sel, [&](std::uint32_t i) {
      std::memcpy(out + i * w, static_cast<const std::uint8_t*>(e) + i * w, w);
    });
    for (std::uint32_t j = 0; j < n; ++j) {
      std::memcpy(out + hits[j] * w, static_cast<const std::uint8_t*>(t) + hits[j] * w, w);
    }
    return out;
  }

 private:
  std::unique_ptr<BoundPred> cond_;
};

// ---------------------------------------------------------------------------
// Binding helpers

bool is_number_literal(const ExprAst& a) {
  return a.kind == ExprAst::IntLit || a.kind == ExprAst::DecLit;
}
bool is_literal(const ExprAst& a) {
  return is_number_literal(a) || a.kind == ExprAst::DateLit || a.kind == ExprAst::StrLit ||
         a.kind == ExprAst::DoubleLit;
}

std::int64_t checked_scale(std::int64_t v, int from, int to) {
  const std::int64_t m = pow10_i64(to - from);
  if (v != 0 && (v > std::numeric_limits<std::int64_t>::max() / m ||
                 v < std::numeric_limits<std::int64_t>::min() / m)) {
    throw std::invalid_argument("constant overflows after rescaling");
  }
  return v * m;
}

std::unique_ptr<BoundExpr> to_double(std::unique_ptr<BoundExpr> e) {
  if (e->type.id == TypeId::Double) return e;
  if (!e->type.is_integer()) throw std::invalid_argument("cannot convert " + type_name(e->type) + " to double");
  return std::make_unique<CastExpr>(std::move(e), Type::float64());
}

std::unique_ptr<BoundExpr> to_int64(std::unique_ptr<BoundExpr> e, int scale) {
  const Type t = e->type;
  if (t.id != TypeId::Int32 && t.id != TypeId::Int64) {
    throw std::invalid_argument("expected an int32/int64 operand, got " + type_name(t));
  }
  if (t.scale > scale) throw std::invalid_argument("cannot lower the scale of an operand");
  if (t.id == TypeId::Int64 && t.scale == scale) return e;
  return std::make_unique<CastExpr>(std::move(e), Type::int64(scale));
}

std::unique_ptr<BoundExpr> bind_literal(const ExprAst& a) {
  switch (a.kind) {
    case ExprAst::IntLit:
      return std::make_unique<ConstVector>(Type::int64(0), a.ivalue, 0.0);
    case ExprAst::DecLit:
      return std::make_unique<ConstVector>(Type::int64(a.scale), a.ivalue, 0.0);
    case ExprAst::DateLit:
      return std::make_unique<ConstVector>(Type::date(), a.ivalue, 0.0);
    case ExprAst::DoubleLit:
      return std::make_unique<ConstVector>(Type::float64(), 0, a.dvalue);
    default:
      throw std::invalid_argument("string literals are only allowed in predicates");
  }
}

std::unique_ptr<BoundExpr> bind_arith(const ExprAst& ast, const Schema& schema) {
  const ArithOp op = ast.kind == ExprAst::Add   ? ArithOp::Add
                     : ast.kind == ExprAst::Sub ? ArithOp::Sub
                     : ast.kind == ExprAst::Mul ? ArithOp::Mul
                                                : ArithOp::Div;
  const ExprAst& la = *ast.args[0];
  const ExprAst& ra = *ast.args[1];
  const Type lt = expr_type(la, schema);
  const Type rt = expr_type(ra, schema);
  const bool floating = op == ArithOp::Div || lt.id == TypeId::Double || rt.id == TypeId::Double ||
                        lt.id == TypeId::Int128 || rt.id == TypeId::Int128;
  if (floating) {
    return std::make_unique<ArithF64>(op, to_double(bind_expr(la, schema)),
                                      to_double(bind_expr(ra, schema)));
  }
  if (!lt.is_integer() || !rt.is_integer()) {
    throw std::invalid_argument("arithmetic needs numeric operands");
  }
  const int scale = op == ArithOp::Mul ? lt.scale + rt.scale : std::max(lt.scale, rt.scale);
  if (scale > 18) throw std::invalid_argument("decimal scale above 18");
  const Type out = Type::int64(scale);

  // Fold a literal operand into the kernel: 1 - x becomes rsub_const(x, 100).
  const bool lconst = is_number_literal(la), rconst = is_number_literal(ra);
  if (lconst != rconst) {
    const ExprAst& lit = lconst ? la : ra;
    const ExprAst& other = lconst ? ra : la;
    const Type ot = lconst ? rt : lt;
    std::int64_t c = lit.ivalue;
    std::unique_ptr<BoundExpr> x;
    if (op == ArithOp::Mul) {
      x = to_int64(bind_expr(other, schema), ot.scale);
    } else {
      x = to_int64(bind_expr(other, schema), scale);
      c = checked_scale(c, lit.kind == ExprAst::DecLit ? lit.scale : 0, scale);
    }
    return std::make_unique<ArithConstI64>(op, std::move(x), c, lconst, out);
  }
  auto l = bind_expr(la, schema);
  auto r = bind_expr(ra, schema);
  if (op == ArithOp::Mul) {
    l = to_int64(std::move(l), lt.scale);
    r = to_int64(std::move(r), rt.scale);
  } else {
    l = to_int64(std::move(l), scale);
    r = to_int64(std::move(r), scale);
  }
  return std::make_unique<ArithI64>(op, std::move(l), std::move(r), out);
}

}  // namespace

Type expr_type(const ExprAst& ast, const Schema& schema) {
  switch (ast.kind) {
    case ExprAst::Column:
      return schema.types[static_cast<std::size_t>(schema.require(ast.text))];
    case ExprAst::IntLit:
      return Type::int64(0);
    case ExprAst::DecLit:
      return Type::int64(ast.scale);
    case ExprAst::DateLit:
      return Type::date();
    case ExprAst::DoubleLit:
      return Type::float64();
    case ExprAst::StrLit:
      return Type::string();
    default:
      return bind_expr(ast, schema)->type;
  }
}

std::unique_ptr<BoundExpr> bind_expr(const ExprAst& ast, const Schema& schema) {
  switch (ast.kind) {
    case ExprAst::Column: {
      const int i = schema.require(ast.text);
      return std::make_unique<ColumnRef>(i, schema.types[static_cast<std::size_t>(i)]);
    }
    case ExprAst::IntLit:
    case ExprAst::DecLit:
    case ExprAst::DateLit:
    case ExprAst::DoubleLit:
    case ExprAst::StrLit:
      return bind_literal(ast);
    case ExprAst::Add:
    case ExprAst::Sub:
    case ExprAst::Mul:
    case ExprAst::Div:
      return bind_arith(ast, schema);
    case ExprAst::Case: {
      auto cond = bind_pred(*ast.cond, schema);
      auto t = bind_expr(*ast.args[0], schema);
      auto e = bind_expr(*ast.args[1], schema);
      if (t->type.id == TypeId::Double || e->type.id == TypeId::Double) {
        t = to_double(std::move(t));
        e = to_double(std::move(e));
      } else {
        const int scale = std::max(t->type.scale, e->type.scale);
        t = to_int64(std::move(t), scale);
        e = to_int64(std::move(e), scale);
      }
      return std::make_unique<CaseExpr>(std::move(cond), std::move(t), std::move(e));
    }
  }
  throw std::logic_error("unhandled expression kind");
}

// ---------------------------------------------------------------------------
// Bound predicates

namespace {

template <class F>
std::uint32_t select_where(const Sel& sel, std::uint32_t* out, F&& pass) {
  std::uint32_t k = 0;
  for_each_selected(sel, [&](std::uint32_t i) {
    out[k] = i;
    k += pass(i) ? 1u : 0u;
  });
  return k;
}

template <class T>
bool compare_values(CmpOp op, T a, T b) {
  switch (op) {
    case CmpOp::Eq:
      return a == b;
    case CmpOp::Ne:
      return a != b;
    case CmpOp::Lt:
      return a < b;
    case CmpOp::Le:
      return a <= b;
    case CmpOp::Gt:
      return a > b;
    default:
      return a >= b;
  }
}

// expr op constant, for int32/date and int64 through the SIMD kernels.
class CmpConstInt final : public BoundPred {
 public:
  CmpConstInt(std::unique_ptr<BoundExpr> x, CmpOp op, std::int64_t c) : op_(op), c_(c) {
    wide_ = x->type.id == TypeId::Int64;
    exprs.push_back(std::move(x));
  }
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    const void* v = exprs[0]->eval(chunk, sel, *state.exprs[0]);
    const Kernels& k = active_kernels();
    if (wide_) {
      const auto* x = static_cast<const std::int64_t*>(v);
      return sel.idx ? k.refine_i64(x, sel.idx, sel.count, op_, c_, out)
                     : k.select_i64(x, sel.count, op_, c_, out);
    }
    const auto* x = static_cast<const std::int32_t*>(v);
    const auto c = static_cast<std::int32_t>(c_);
    return sel.idx ? k.refine_i32(x, sel.idx, sel.count, op_, c, out)
                   : k.select_i32(x, sel.count, op_, c, out);
  }

 private:
  CmpOp op_;
  std::int64_t c_;
  bool wide_;
};

class BetweenInt final : public BoundPred {
 public:
  BetweenInt(std::unique_ptr<BoundExpr> x, std::int64_t lo, std::int64_t hi) : lo_(lo), hi_(hi) {
    wide_ = x->type.id == TypeId::Int64;
    exprs.push_back(std::move(x));
  }
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    const void* v = exprs[0]->eval(chunk, sel, *state.exprs[0]);
    const Kernels& k = active_kernels();
    if (wide_) {
      const auto* x = static_cast<const std::int64_t*>(v);
      return sel.idx ? k.refine_between_i64(x, sel.idx, sel.count, lo_, hi_, out)
                     : k.between_i64(x, sel.count, lo_, hi_, out);
    }
    const auto* x = static_cast<const std::int32_t*>(v);
    const auto lo = static_cast<std::int32_t>(lo_), hi = static_cast<std::int32_t>(hi_);
    return sel.idx ? k.refine_between_i32(x, sel.idx, sel.count, lo, hi, out)
                   : k.between_i32(x, sel.count, lo, hi, out);
  }

 private:
  std::int64_t lo_, hi_;
  bool wide_;
};

// expr op constant for int128 and double (aggregation output, HAVING).
template <class T>
class CmpConstScalar final : public BoundPred {
 public:
  CmpConstScalar(std::unique_ptr<BoundExpr> x, CmpOp op, T c) : op_(op), c_(c) {
    exprs.push_back(std::move(x));
  }
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    const auto* x = static_cast<const T*>(exprs[0]->eval(chunk, sel, *state.exprs[0]));
    return select_where(sel, out, [&](std::uint32_t i) { return compare_values(op_, x[i], c_); });
  }

 private:
  CmpOp op_;
  T c_;
};

class CmpColsI32 final : public BoundPred {
 public:
  CmpColsI32(std::unique_ptr<BoundExpr> a, std::unique_ptr<BoundExpr> b, CmpOp op) : op_(op) {
    exprs.push_back(std::move(a));
    exprs.push_back(std::move(b));
  }
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    const auto* a = static_cast<const std::int32_t*>(exprs[0]->eval(chunk, sel, *state.exprs[0]));
    const auto* b = static_cast<const std::int32_t*>(exprs[1]->eval(chunk, sel, *state.exprs[1]));
    const Kernels& k = active_kernels();
    return sel.idx ? k.refine_cols_i32(a, b, sel.idx, sel.count, op_, out)
                   : k.select_cols_i32(a, b, sel.count, op_, out);
  }

 private:
  CmpOp op_;
};

template <class T>
class CmpColsScalar final : public BoundPred {
 public:
  CmpColsScalar(std::unique_ptr<BoundExpr> a, std::unique_ptr<BoundExpr> b, CmpOp op) : op_(op) {
    exprs.push_back(std::move(a));
    exprs.push_back(std::move(b));
  }
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    const auto* a = static_cast<const T*>(exprs[0]->eval(chunk, sel, *state.exprs[0]));
    const auto* b = static_cast<const T*>(exprs[1]->eval(chunk, sel, *state.exprs[1]));
    return select_where(sel, out, [&](std::uint32_t i) { return compare_values(op_, a[i], b[i]); });
  }

 private:
  CmpOp op_;
};

// Any string predicate on a dictionary column, precomputed per code.
class DictLookup final : public BoundPred {
 public:
  DictLookup(std::unique_ptr<BoundExpr> x, const std::function<bool(const std::string&)>& f) {
    std::memset(pass_, 0, sizeof pass_);
    const Dictionary& d = *x->type.dict;
    for (std::size_t c = 0; c < d.values.size(); ++c) pass_[c] = f(d.values[c]) ? 1 : 0;
    exprs.push_back(std::move(x));
  }
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    const auto* codes = static_cast<const std::uint8_t*>(exprs[0]->eval(chunk, sel, *state.exprs[0]));
    std::uint32_t k = 0;
    for_each_selected(sel, [&](std::uint32_t i) {
      out[k] = i;
      k += pass_[codes[i]];
    });
    return k;
  }

 private:
  std::uint8_t pass_[256];
};

class StringMatch final : public BoundPred {
 public:
  StringMatch(std::unique_ptr<BoundExpr> x, std::function<bool(std::string_view)> f)
      : f_(std::move(f)) {
    exprs.push_back(std::move(x));
  }
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    const auto* s = static_cast<const std::string_view*>(exprs[0]->eval(chunk, sel, *state.exprs[0]));
    return select_where(sel, out, [&](std::uint32_t i) { return f_(s[i]); });
  }

 private:
  std::function<bool(std::string_view)> f_;
};

class AndPred final : public BoundPred {
 public:
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    Sel cur = sel;
    for (std::size_t c = 0; c < children.size(); ++c) {
      cur.count = children[c]->select(chunk, cur, out, *state.children[c]);
      cur.idx = out;
      if (cur.count == 0) break;
    }
    return cur.count;
  }
};

class OrPred final : public BoundPred {
 public:
  std::unique_ptr<PredState> make_state() const override {
    auto state = BoundPred::make_state();
    for (std::size_t i = 0; i < children.size() + 1; ++i) {
      state->bufs.emplace_back(kSelCapacity * sizeof(std::uint32_t));
    }
    return state;
  }
  std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                       PredState& state) const override {
    // Each branch selects from the input; the sorted results are merged.
    // The input is read in full before `out` (which may alias it) is written.
    auto* acc = state.bufs[0].as<std::uint32_t>();
    std::uint32_t n = children[0]->select(chunk, sel, acc, *state.children[0]);
    for (std::size_t c = 1; c < children.size(); ++c) {
      auto* branch = state.bufs[c].as<std::uint32_t>();
      auto* merged = state.bufs[children.size()].as<std::uint32_t>();
      const std::uint32_t m = children[c]->select(chunk, sel, branch, *state.children[c]);
      std::uint32_t i = 0, j = 0, k = 0;
      while (i < n && j < m) {
        const std::uint32_t a = acc[i], b = branch[j];
        merged[k++] = a < b ? a : b;
        i += a <= b;
        j += b <= a;
      }
      while (i < n) merged[k++] = acc[i++];
      while (j < m) merged[k++] = branch[j++];
      std::memcpy(acc, merged, k * sizeof(std::uint32_t));
      n = k;
    }
    std::memcpy(out, acc, n * sizeof(std::uint32_t));
    return n;
  }
};

std::int64_t literal_as(const ExprAst& lit, const Type& t, bool* inexact) {
  *inexact = false;
  if (t.id == TypeId::Date) {
    if (lit.kind != ExprAst::DateLit) throw std::invalid_argument("compare a date with a date literal");
    return lit.ivalue;
  }
  if (!is_number_literal(lit)) throw std::invalid_argument("expected a numeric literal");
  const int ls = lit.kind == ExprAst::DecLit ? lit.scale : 0;
  if (ls > t.scale) {
    *inexact = true;
    return 0;
  }
  return checked_scale(lit.ivalue, ls, t.scale);
}

bool fits_i32(std::int64_t v) {
  return v >= std::numeric_limits<std::int32_t>::min() && v <= std::numeric_limits<std::int32_t>::max();
}

std::function<bool(std::string_view)> string_cmp(CmpOp op, std::string c) {
  return [op, c = std::move(c)](std::string_view s) { return compare_values(op, s, std::string_view(c)); };
}

std::unique_ptr<BoundPred> bind_cmp_const(std::unique_ptr<BoundExpr> x, CmpOp op, const ExprAst& lit) {
  const Type t = x->type;
  if (t.id == TypeId::Dict || t.id == TypeId::String) {
    if (lit.kind != ExprAst::StrLit) throw std::invalid_argument("compare strings with str(...)");
    auto f = string_cmp(op, lit.text);
    if (t.id == TypeId::Dict) {
      return std::make_unique<DictLookup>(std::move(x), [f](const std::string& s) { return f(s); });
    }
    return std::make_unique<StringMatch>(std::move(x), f);
  }
  if (t.id == TypeId::Double) {
    const double c = lit.kind == ExprAst::DoubleLit ? lit.dvalue
                     : lit.kind == ExprAst::DecLit
                         ? static_cast<double>(lit.ivalue) / static_cast<double>(pow10_i64(lit.scale))
                         : static_cast<double>(lit.ivalue);
    return std::make_unique<CmpConstScalar<double>>(std::move(x), op, c);
  }
  bool inexact = false;
  std::int64_t c = literal_as(lit, t, &inexact);
  if (inexact) {
    // The literal has more decimals than the column: compare at its scale.
    x = to_int64(std::move(x), lit.scale);
    return bind_cmp_const(std::move(x), op, lit);
  }
  if (t.id == TypeId::Int128) {
    return std::make_unique<CmpConstScalar<int128>>(std::move(x), op, static_cast<int128>(c));
  }
  if (t.id == TypeId::Int64) return std::make_unique<CmpConstInt>(std::move(x), op, c);
  if (!fits_i32(c)) {
    x = to_int64(std::move(x), t.scale);
    return std::make_unique<CmpConstInt>(std::move(x), op, c);
  }
  return std::make_unique<CmpConstInt>(std::move(x), op, c);
}

}  // namespace

std::unique_ptr<BoundPred> bind_pred(const PredAst& ast, const Schema& schema) {
  switch (ast.kind) {
    case PredAst::And:
    case PredAst::Or: {
      std::unique_ptr<BoundPred> p;
      if (ast.kind == PredAst::And) {
        p = std::make_unique<AndPred>();
      } else {
        p = std::make_unique<OrPred>();
      }
      for (const auto& c : ast.children) p->children.push_back(bind_pred(*c, schema));
      return p;
    }
    case PredAst::Cmp: {
      const ExprAst* a = ast.args[0].get();
      const ExprAst* b = ast.args[1].get();
      CmpOp op = ast.op;
      if (is_literal(*a) && !is_literal(*b)) {
        std::swap(a, b);
        op = flip(op);
      }
      if (is_literal(*b)) return bind_cmp_const(bind_expr(*a, schema), op, *b);
      auto l = bind_expr(*a, schema);
      auto r = bind_expr(*b, schema);
      const Type lt = l->type, rt = r->type;
      if ((lt.id == TypeId::Date && rt.id == TypeId::Date) ||
          (lt.id == TypeId::Int32 && rt.id == TypeId::Int32 && lt.scale == rt.scale)) {
        return std::make_unique<CmpColsI32>(std::move(l), std::move(r), op);
      }
      if (lt.id == TypeId::Double || rt.id == TypeId::Double || lt.id == TypeId::Int128 ||
          rt.id == TypeId::Int128) {
        return std::make_unique<CmpColsScalar<double>>(to_double(std::move(l)), to_double(std::move(r)), op);
      }
      const int scale = std::max(lt.scale, rt.scale);
      return std::make_unique<CmpColsScalar<std::int64_t>>(to_int64(std::move(l), scale),
                                                           to_int64(std::move(r), scale), op);
    }
    case PredAst::Between: {
      auto x = bind_expr(*ast.args[0], schema);
      const Type t = x->type;
      const ExprAst& lo = *ast.args[1];
      const ExprAst& hi = *ast.args[2];
      if (t.id == TypeId::Int32 || t.id == TypeId::Int64 || t.id == TypeId::Date) {
        bool inexact_lo = false, inexact_hi = false;
        const std::int64_t l = literal_as(lo, t, &inexact_lo);
        const std::int64_t h = literal_as(hi, t, &inexact_hi);
        if (!inexact_lo && !inexact_hi && (t.id == TypeId::Int64 || (fits_i32(l) && fits_i32(h)))) {
          return std::make_unique<BetweenInt>(std::move(x), l, h);
        }
      }
      P expanded = E(ast.args[0]) >= E(ast.args[1]) && E(ast.args[0]) <= E(ast.args[2]);
      return bind_pred(*expanded.ast, schema);
    }
    case PredAst::In:
    case PredAst::Like: {
      auto x = bind_expr(*ast.args[0], schema);
      const bool negated = ast.negated;
      std::function<bool(std::string_view)> f;
      if (ast.kind == PredAst::In) {
        f = [values = ast.strings, negated](std::string_view s) {
          for (const auto& v : values) {
            if (s == v) return !negated;
          }
          return negated;
        };
      } else {
        f = [pattern = ast.strings[0], negated](std::string_view s) {
          return like_match(s, pattern) != negated;
        };
      }
      if (x->type.id == TypeId::Dict) {
        return std::make_unique<DictLookup>(std::move(x), [f](const std::string& s) { return f(s); });
      }
      if (x->type.id == TypeId::String) return std::make_unique<StringMatch>(std::move(x), f);
      throw std::invalid_argument("IN/LIKE need a string column");
    }
  }
  throw std::logic_error("unhandled predicate kind");
}

}  // namespace morsel
