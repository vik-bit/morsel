#pragma once

// A small physical-plan builder. Queries are written as a tree of operators
// in C++, with join order, build sides and projections chosen by hand:
//
//   Plan::scan(db, "lineitem", {"l_shipdate", "l_extendedprice", "l_discount"})
//       .filter(col("l_shipdate") >= date("1994-01-01"))
//       .aggregate({}, {sum(col("l_extendedprice") * col("l_discount"), "revenue")});
//
// Every step binds its expressions against the input schema right away, so a
// typo in a column name fails when the plan is built, not when it runs.

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "morsel/expr.h"
#include "morsel/storage.h"

namespace morsel {

enum class JoinType { Inner, Semi, Anti };
enum class AggKind { Sum, Count, Avg, Min, Max, Any };

struct Agg {
  AggKind kind;
  std::shared_ptr<ExprAst> input;  // null for count(*)
  std::string name;
};

Agg sum(const E& x, const std::string& name);
Agg avg(const E& x, const std::string& name);
Agg min(const E& x, const std::string& name);
Agg max(const E& x, const std::string& name);
// Any value of the group. For columns that are functionally dependent on the
// grouping key (o_orderdate given o_orderkey), this is cheaper than grouping
// by them too.
Agg any_value(const E& x, const std::string& name);
Agg count_star(const std::string& name);

struct SortKey {
  std::string column;
  bool descending = false;
};
inline SortKey asc(const std::string& c) { return {c, false}; }
inline SortKey desc(const std::string& c) { return {c, true}; }

struct BoundAgg {
  AggKind kind;
  int input = -1;  // column of the aggregation's input, -1 for count(*)
  Type input_type;
  Type output_type;
  std::string name;
};

struct PlanNode {
  enum Kind { Scan, Filter, Project, Join, Aggregate, OrderBy };
  Kind kind = Scan;
  Schema schema;  // output columns
  std::vector<std::shared_ptr<PlanNode>> children;  // Join: {probe, build}
  std::string label;

  Relation relation;                              // Scan
  std::shared_ptr<BoundPred> pred;                // Filter
  std::vector<std::shared_ptr<BoundExpr>> exprs;  // Project
  JoinType join_type = JoinType::Inner;           // Join
  std::vector<int> probe_keys, build_keys, build_payload;
  std::vector<int> group_cols;                    // Aggregate
  std::vector<BoundAgg> aggs;
  std::vector<int> sort_cols;                     // OrderBy
  std::vector<bool> sort_desc;
  std::size_t limit = 0;                          // 0 = no limit
};

class Plan {
 public:
  // Scans the named columns of a table (loading them if needed).
  static Plan scan(Database& db, const std::string& table, const std::vector<std::string>& columns);
  // Scans an existing relation, e.g. a MaterializedTable's view(). The
  // relation's storage must outlive the plan's execution.
  static Plan scan(const Relation& relation, const std::string& label = "relation");

  Plan filter(const P& pred) const;
  Plan project(const std::vector<std::pair<std::string, E>>& exprs) const;
  Plan select(const std::vector<std::string>& columns) const;

  // Hash join; `build` is materialized into the hash table. Inner joins
  // output the probe columns followed by the build columns that are not keys;
  // semi and anti joins output only the probe columns.
  Plan join(const Plan& build, const std::vector<std::string>& probe_keys,
            const std::vector<std::string>& build_keys, JoinType type = JoinType::Inner) const;

  Plan aggregate(const std::vector<std::string>& group_by, const std::vector<Agg>& aggs) const;
  Plan order_by(const std::vector<SortKey>& keys, std::size_t limit = 0) const;

  const Schema& schema() const { return node->schema; }
  std::string explain() const;

  std::shared_ptr<PlanNode> node;
};

}  // namespace morsel
