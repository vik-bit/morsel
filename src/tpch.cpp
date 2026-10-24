// TPC-H plans. Join order and build sides are chosen by hand (smaller,
// filtered input on the build side), and each probe side is projected down
// to the columns later operators need before the join gathers them. The SQL
// each plan implements is DuckDB's `tpch_queries()` text.

#include "morsel/tpch.h"

#include <stdexcept>
#include <string>

namespace morsel {

namespace {

E revenue() { return col("l_extendedprice") * (1 - col("l_discount")); }

// Q1: pricing summary report. One scan, 98% of lineitem passes the filter,
// four groups: dominated by expression evaluation and aggregation.
Plan q1(Database& db) {
  return Plan::scan(db, "lineitem",
                    {"l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice", "l_discount", "l_tax",
                     "l_shipdate"})
      .filter(col("l_shipdate") <= date("1998-09-02"))
      .project({{"l_returnflag", col("l_returnflag")},
                {"l_linestatus", col("l_linestatus")},
                {"l_quantity", col("l_quantity")},
                {"l_extendedprice", col("l_extendedprice")},
                {"l_discount", col("l_discount")},
                {"l_tax", col("l_tax")},
                {"disc_price", revenue()}})
      .aggregate({"l_returnflag", "l_linestatus"},
                 {sum(col("l_quantity"), "sum_qty"), sum(col("l_extendedprice"), "sum_base_price"),
                  sum(col("disc_price"), "sum_disc_price"), sum(col("disc_price") * (1 + col("l_tax")), "sum_charge"),
                  avg(col("l_quantity"), "avg_qty"), avg(col("l_extendedprice"), "avg_price"),
                  avg(col("l_discount"), "avg_disc"), count_star("count_order")})
      .order_by({asc("l_returnflag"), asc("l_linestatus")});
}

// Q3: shipping priority. customer -> orders -> lineitem.
Plan q3(Database& db) {
  const Plan customers = Plan::scan(db, "customer", {"c_custkey", "c_mktsegment"})
                             .filter(col("c_mktsegment") == str("BUILDING"))
                             .select({"c_custkey"});
  const Plan orders = Plan::scan(db, "orders", {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"})
                          .filter(col("o_orderdate") < date("1995-03-15"))
                          .join(customers, {"o_custkey"}, {"c_custkey"}, JoinType::Semi)
                          .select({"o_orderkey", "o_orderdate", "o_shippriority"});
  return Plan::scan(db, "lineitem", {"l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"})
      .filter(col("l_shipdate") > date("1995-03-15"))
      .select({"l_orderkey", "l_extendedprice", "l_discount"})
      .join(orders, {"l_orderkey"}, {"o_orderkey"})
      .aggregate({"l_orderkey", "o_orderdate", "o_shippriority"}, {sum(revenue(), "revenue")})
      .order_by({desc("revenue"), asc("o_orderdate")}, 10)
      .select({"l_orderkey", "revenue", "o_orderdate", "o_shippriority"});
}

// Q5: local supplier volume. Six tables; the last join has a two-column key
// (supplier and nation), packed into one 64-bit hash key.
Plan q5(Database& db) {
  const Plan asia = Plan::scan(db, "region", {"r_regionkey", "r_name"})
                        .filter(col("r_name") == str("ASIA"))
                        .select({"r_regionkey"});
  const Plan nations = Plan::scan(db, "nation", {"n_nationkey", "n_name", "n_regionkey"})
                           .join(asia, {"n_regionkey"}, {"r_regionkey"}, JoinType::Semi)
                           .select({"n_nationkey", "n_name"});
  const Plan customers = Plan::scan(db, "customer", {"c_custkey", "c_nationkey"})
                             .join(nations.select({"n_nationkey"}), {"c_nationkey"}, {"n_nationkey"}, JoinType::Semi);
  const Plan orders = Plan::scan(db, "orders", {"o_orderkey", "o_custkey", "o_orderdate"})
                          .filter(col("o_orderdate") >= date("1994-01-01") && col("o_orderdate") < date("1995-01-01"))
                          .select({"o_orderkey", "o_custkey"})
                          .join(customers, {"o_custkey"}, {"c_custkey"})
                          .select({"o_orderkey", "c_nationkey"});
  const Plan suppliers = Plan::scan(db, "supplier", {"s_suppkey", "s_nationkey"})
                             .join(nations, {"s_nationkey"}, {"n_nationkey"})
                             .select({"s_suppkey", "s_nationkey", "n_name"});
  return Plan::scan(db, "lineitem", {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"})
      .join(orders, {"l_orderkey"}, {"o_orderkey"})
      .select({"l_suppkey", "l_extendedprice", "l_discount", "c_nationkey"})
      .join(suppliers, {"l_suppkey", "c_nationkey"}, {"s_suppkey", "s_nationkey"})
      .aggregate({"n_name"}, {sum(revenue(), "revenue")})
      .order_by({desc("revenue")});
}

// Q6: forecasting revenue change. A scan with a selective conjunctive filter
// (about 2% of rows pass): the SIMD selection kernels do most of the work.
Plan q6(Database& db) {
  return Plan::scan(db, "lineitem", {"l_shipdate", "l_discount", "l_quantity", "l_extendedprice"})
      .filter(col("l_shipdate") >= date("1994-01-01") && col("l_shipdate") < date("1995-01-01") &&
              between(col("l_discount"), dec("0.05"), dec("0.07")) && col("l_quantity") < 24)
      .aggregate({}, {sum(col("l_extendedprice") * col("l_discount"), "revenue")});
}

// Q12: shipping modes and order priority. The filtered lineitem side (about
// 30k rows at SF1) is the build side; all of orders probes it.
Plan q12(Database& db) {
  const Plan lines =
      Plan::scan(db, "lineitem", {"l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"})
          .filter(col("l_receiptdate") >= date("1994-01-01") && col("l_receiptdate") < date("1995-01-01") &&
                  in(col("l_shipmode"), {"MAIL", "SHIP"}) && col("l_commitdate") < col("l_receiptdate") &&
                  col("l_shipdate") < col("l_commitdate"))
          .select({"l_orderkey", "l_shipmode"});
  const std::vector<std::string> high = {"1-URGENT", "2-HIGH"};
  return Plan::scan(db, "orders", {"o_orderkey", "o_orderpriority"})
      .join(lines, {"o_orderkey"}, {"l_orderkey"})
      .aggregate({"l_shipmode"}, {sum(case_when(in(col("o_orderpriority"), high), 1, 0), "high_line_count"),
                                  sum(case_when(not_in(col("o_orderpriority"), high), 1, 0), "low_line_count")})
      .order_by({asc("l_shipmode")});
}

// Q14: promotion effect. LIKE 'PROMO%' on the p_type dictionary is evaluated
// once per distinct value at bind time.
Plan q14(Database& db) {
  const Plan parts = Plan::scan(db, "part", {"p_partkey", "p_type"});
  return Plan::scan(db, "lineitem", {"l_partkey", "l_extendedprice", "l_discount", "l_shipdate"})
      .filter(col("l_shipdate") >= date("1995-09-01") && col("l_shipdate") < date("1995-10-01"))
      .select({"l_partkey", "l_extendedprice", "l_discount"})
      .join(parts, {"l_partkey"}, {"p_partkey"})
      .aggregate({}, {sum(case_when(like(col("p_type"), "PROMO%"), revenue(), 0), "promo"),
                      sum(revenue(), "total")})
      .project({{"promo_revenue", dbl(100.0) * col("promo") / col("total")}});
}

// Q18: large volume customer. The IN subquery is an aggregation over all of
// lineitem (1.5M groups at SF1, merged in parallel by partition) feeding a
// semi join. Columns that depend on o_orderkey are carried with any_value
// rather than grouped on.
Plan q18(Database& db) {
  const Plan big_orders = Plan::scan(db, "lineitem", {"l_orderkey", "l_quantity"})
                              .aggregate({"l_orderkey"}, {sum(col("l_quantity"), "qty")})
                              .filter(col("qty") > 300)
                              .select({"l_orderkey"});
  const Plan customers = Plan::scan(db, "customer", {"c_custkey", "c_name"});
  const Plan orders = Plan::scan(db, "orders", {"o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"})
                          .join(big_orders, {"o_orderkey"}, {"l_orderkey"}, JoinType::Semi)
                          .join(customers, {"o_custkey"}, {"c_custkey"});
  return Plan::scan(db, "lineitem", {"l_orderkey", "l_quantity"})
      .join(orders, {"l_orderkey"}, {"o_orderkey"})
      .aggregate({"l_orderkey"}, {any_value(col("c_name"), "c_name"), any_value(col("o_custkey"), "c_custkey"),
                                  any_value(col("o_orderdate"), "o_orderdate"),
                                  any_value(col("o_totalprice"), "o_totalprice"),
                                  sum(col("l_quantity"), "sum(l_quantity)")})
      .order_by({desc("o_totalprice"), asc("o_orderdate")}, 100)
      .project({{"c_name", col("c_name")},
                {"c_custkey", col("c_custkey")},
                {"o_orderkey", col("l_orderkey")},
                {"o_orderdate", col("o_orderdate")},
                {"o_totalprice", col("o_totalprice")},
                {"sum(l_quantity)", col("sum(l_quantity)")}});
}

// Q19: discounted revenue. A disjunction of three conjunctions over both
// tables. The single-table parts of each branch are pushed below the join
// (as DuckDB's optimizer also does); the full OR is applied after it.
Plan q19(Database& db) {
  const std::vector<std::string> sm = {"SM CASE", "SM BOX", "SM PACK", "SM PKG"};
  const std::vector<std::string> med = {"MED BAG", "MED BOX", "MED PKG", "MED PACK"};
  const std::vector<std::string> lg = {"LG CASE", "LG BOX", "LG PACK", "LG PKG"};
  auto part_branch = [](const char* brand, const std::vector<std::string>& containers, int max_size) {
    return col("p_brand") == str(brand) && in(col("p_container"), containers) && between(col("p_size"), 1, max_size);
  };
  const Plan parts = Plan::scan(db, "part", {"p_partkey", "p_brand", "p_container", "p_size"})
                         .filter(part_branch("Brand#12", sm, 5) || part_branch("Brand#23", med, 10) ||
                                 part_branch("Brand#34", lg, 15));
  return Plan::scan(db, "lineitem",
                    {"l_partkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipmode", "l_shipinstruct"})
      .filter(between(col("l_quantity"), 1, 30) && in(col("l_shipmode"), {"AIR", "AIR REG"}) &&
              col("l_shipinstruct") == str("DELIVER IN PERSON"))
      .select({"l_partkey", "l_quantity", "l_extendedprice", "l_discount"})
      .join(parts, {"l_partkey"}, {"p_partkey"})
      .filter((part_branch("Brand#12", sm, 5) && between(col("l_quantity"), 1, 11)) ||
              (part_branch("Brand#23", med, 10) && between(col("l_quantity"), 10, 20)) ||
              (part_branch("Brand#34", lg, 15) && between(col("l_quantity"), 20, 30)))
      .aggregate({}, {sum(revenue(), "revenue")});
}

}  // namespace

const std::vector<int>& tpch_supported_queries() {
  static const std::vector<int> queries = {1, 3, 5, 6, 12, 14, 18, 19};
  return queries;
}

Plan tpch_plan(Database& db, int query) {
  switch (query) {
    case 1:
      return q1(db);
    case 3:
      return q3(db);
    case 5:
      return q5(db);
    case 6:
      return q6(db);
    case 12:
      return q12(db);
    case 14:
      return q14(db);
    case 18:
      return q18(db);
    case 19:
      return q19(db);
    default:
      throw std::invalid_argument("TPC-H query " + std::to_string(query) + " is not implemented");
  }
}

}  // namespace morsel
