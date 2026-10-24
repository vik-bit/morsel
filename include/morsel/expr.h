#pragma once

// Expressions and predicates.
//
// Plans are written with a small C++ DSL (E for values, P for predicates):
//
//   col("l_extendedprice") * (1 - col("l_discount"))
//   col("l_shipdate") < date("1995-01-01") && in(col("l_shipmode"), {"MAIL", "SHIP"})
//
// Binding against a schema resolves names, checks types, aligns decimal
// scales and folds constants into the kernels (x < 24 becomes a compare of
// the scaled int64 against 2400). String predicates on dictionary columns are
// evaluated once against the dictionary at bind time and become a 256-entry
// lookup table on the codes.

#include <memory>
#include <string>
#include <vector>

#include "morsel/chunk.h"
#include "morsel/kernels.h"
#include "morsel/storage.h"
#include "morsel/types.h"

namespace morsel {

struct Schema {
  std::vector<std::string> names;
  std::vector<Type> types;

  int index_of(const std::string& name) const;  // -1 if absent
  int require(const std::string& name) const;   // throws if absent
  std::size_t size() const { return names.size(); }
};

// ---------------------------------------------------------------------------
// Unbound syntax trees

struct PredAst;

struct ExprAst {
  enum Kind { Column, IntLit, DecLit, DateLit, StrLit, DoubleLit, Add, Sub, Mul, Div, Case };
  Kind kind = Column;
  std::string text;          // column name or string literal
  std::int64_t ivalue = 0;   // IntLit, DecLit (scaled), DateLit (days)
  int scale = 0;             // DecLit
  double dvalue = 0;         // DoubleLit
  std::vector<std::shared_ptr<ExprAst>> args;
  std::shared_ptr<PredAst> cond;  // Case
};

struct PredAst {
  enum Kind { Cmp, Between, In, Like, And, Or };
  Kind kind = Cmp;
  CmpOp op = CmpOp::Eq;
  bool negated = false;  // NOT IN / NOT LIKE
  std::vector<std::shared_ptr<ExprAst>> args;
  std::vector<std::string> strings;  // In list, Like pattern
  std::vector<std::shared_ptr<PredAst>> children;
};

class E {
 public:
  E(std::shared_ptr<ExprAst> a) : ast(std::move(a)) {}  // NOLINT: implicit by design
  E(int value);                                              // NOLINT: allows 1 - col("x")
  E(std::int64_t value);                                     // NOLINT
  std::shared_ptr<ExprAst> ast;
};

class P {
 public:
  explicit P(std::shared_ptr<PredAst> a) : ast(std::move(a)) {}
  std::shared_ptr<PredAst> ast;
};

E col(const std::string& name);
E dec(const std::string& text);   // decimal literal, e.g. dec("0.05")
E date(const std::string& text);  // date literal, e.g. date("1994-01-01")
E str(const std::string& text);   // string literal, for predicates
E dbl(double value);
E case_when(const P& cond, const E& then_value, const E& else_value);

E operator+(const E& a, const E& b);
E operator-(const E& a, const E& b);
E operator*(const E& a, const E& b);
E operator/(const E& a, const E& b);  // always produces a double

P operator==(const E& a, const E& b);
P operator!=(const E& a, const E& b);
P operator<(const E& a, const E& b);
P operator<=(const E& a, const E& b);
P operator>(const E& a, const E& b);
P operator>=(const E& a, const E& b);
P operator&&(const P& a, const P& b);
P operator||(const P& a, const P& b);
P between(const E& x, const E& lo, const E& hi);  // inclusive, like SQL
P in(const E& x, const std::vector<std::string>& values);
P not_in(const E& x, const std::vector<std::string>& values);
P like(const E& x, const std::string& pattern);  // SQL LIKE with % and _

bool like_match(std::string_view s, std::string_view pattern);

// ---------------------------------------------------------------------------
// Bound expressions: evaluated one vector at a time. Every operator instance
// keeps one state tree per thread, holding that thread's output buffers.

struct ExprState {
  Buffer out;
  std::vector<std::unique_ptr<ExprState>> children;
  std::vector<std::unique_ptr<struct PredState>> preds;
};

class BoundExpr {
 public:
  explicit BoundExpr(Type t) : type(t) {}
  virtual ~BoundExpr() = default;

  // Returns a vector valid at every position the active set selects (and, in
  // dense mode, at every position of the chunk). Column references return
  // the input pointer without copying.
  virtual const void* eval(const Chunk& chunk, const Sel& sel, ExprState& state) const = 0;
  virtual std::unique_ptr<ExprState> make_state() const;
  virtual int column_index() const { return -1; }

  Type type;
  std::vector<std::unique_ptr<BoundExpr>> children;
};

struct PredState {
  std::vector<Buffer> bufs;  // selection scratch space
  std::vector<std::unique_ptr<ExprState>> exprs;
  std::vector<std::unique_ptr<PredState>> children;
};

class BoundPred {
 public:
  virtual ~BoundPred() = default;
  // Writes the positions of `sel` that pass into out (which may alias
  // sel.idx) and returns how many passed. out needs kSelCapacity entries.
  virtual std::uint32_t select(const Chunk& chunk, const Sel& sel, std::uint32_t* out,
                               PredState& state) const = 0;
  virtual std::unique_ptr<PredState> make_state() const;

  std::vector<std::unique_ptr<BoundExpr>> exprs;
  std::vector<std::unique_ptr<BoundPred>> children;
};

std::unique_ptr<BoundExpr> bind_expr(const ExprAst& ast, const Schema& schema);
std::unique_ptr<BoundPred> bind_pred(const PredAst& ast, const Schema& schema);

// Result type of an expression without building it (used by the planner).
Type expr_type(const ExprAst& ast, const Schema& schema);

}  // namespace morsel
