#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "check.h"
#include "morsel/executor.h"
#include "morsel/plan.h"

using namespace morsel;

namespace {

MaterializedTable run(const Plan& plan, int threads) {
  Executor exec(threads);
  return exec.run(plan);
}

template <class T>
T at(const MaterializedTable& r, std::size_t col, std::size_t row) {
  return r.columns[col].as<T>()[row];
}

// fact(k, g, v, s): k joins dim.k (some keys missing, some duplicated in dim),
// g is a 4-value dictionary group, v a scale-2 decimal, s a string.
struct Data {
  Database db;
  std::vector<std::int32_t> k, dk, dw;
  std::vector<std::int64_t> v;
  std::vector<std::string> g;
};

Data make_data(int n, int keys, std::uint32_t seed) {
  Data d;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> key(0, keys - 1), val(-5000, 50000);
  const char* groups[] = {"AIR", "MAIL", "RAIL", "SHIP"};
  std::vector<std::string> s(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    d.k.push_back(key(rng));
    d.v.push_back(val(rng));
    d.g.push_back(groups[rng() % 4]);
    s[static_cast<std::size_t>(i)] = "s" + std::to_string(i % 97);
  }
  Table& fact = d.db.add_table("fact");
  fact.add_column("k", Type::int32(), d.k);
  fact.add_column("v", Type::int64(2), d.v);
  fact.add_dict_column("g", d.g);
  fact.add_string_column("s", s);
  // dim: every even key once, every multiple of 6 a second time, odd keys absent.
  for (int x = 0; x < keys; x += 2) {
    d.dk.push_back(x);
    d.dw.push_back(x * 10);
    if (x % 6 == 0) {
      d.dk.push_back(x);
      d.dw.push_back(x * 10 + 1);
    }
  }
  Table& dim = d.db.add_table("dim");
  dim.add_column("dk", Type::int32(), d.dk);
  dim.add_column("w", Type::int32(), d.dw);
  return d;
}

}  // namespace

TEST(grouped_aggregates_match_reference) {
  Data d = make_data(100000, 1000, 11);
  std::map<std::string, std::tuple<int128, std::int64_t, std::int64_t, std::int64_t>> ref;  // sum, count, min, max
  for (std::size_t i = 0; i < d.v.size(); ++i) {
    auto it = ref.find(d.g[i]);
    if (it == ref.end()) it = ref.emplace(d.g[i], std::make_tuple(int128{0}, 0LL, INT64_MAX, INT64_MIN)).first;
    auto& [s, c, mn, mx] = it->second;
    s += d.v[i];
    ++c;
    mn = std::min(mn, d.v[i]);
    mx = std::max(mx, d.v[i]);
  }
  for (int threads : {1, 4}) {
    const auto r = run(Plan::scan(d.db, "fact", {"g", "v"})
                           .aggregate({"g"}, {sum(col("v"), "s"), count_star("c"), min(col("v"), "mn"),
                                              max(col("v"), "mx"), avg(col("v"), "a")})
                           .order_by({asc("g")}),
                       threads);
    CHECK(r.rows == ref.size());
    std::size_t row = 0;
    for (const auto& [g, t] : ref) {
      const auto& [s, c, mn, mx] = t;
      CHECK(r.value(0, row) == g);
      CHECK(at<int128>(r, 1, row) == s);
      CHECK(at<std::int64_t>(r, 2, row) == c);
      CHECK(at<std::int64_t>(r, 3, row) == mn);
      CHECK(at<std::int64_t>(r, 4, row) == mx);
      CHECK(std::abs(at<double>(r, 5, row) - static_cast<double>(s) / c / 100.0) < 1e-9);
      ++row;
    }
  }
}

// More groups than kParallelMergeGroups, so the partitioned parallel merge
// runs; every group must appear exactly once with the right sum.
TEST(high_cardinality_aggregation_merges_partitions_correctly) {
  Data d = make_data(300000, 60000, 12);
  std::map<std::int32_t, int128> ref;
  for (std::size_t i = 0; i < d.k.size(); ++i) ref[d.k[i]] += d.v[i];
  for (int threads : {1, 2, 8}) {
    const auto r = run(Plan::scan(d.db, "fact", {"k", "v"}).aggregate({"k"}, {sum(col("v"), "s")}), threads);
    CHECK(r.rows == ref.size());
    std::map<std::int32_t, int128> got;
    for (std::size_t i = 0; i < r.rows; ++i) got[at<std::int32_t>(r, 0, i)] += at<int128>(r, 1, i);
    CHECK(got == ref);
  }
}

TEST(multi_column_group_keys) {
  Data d = make_data(50000, 300, 13);
  std::map<std::pair<std::string, std::int32_t>, std::int64_t> ref;
  for (std::size_t i = 0; i < d.k.size(); ++i) ++ref[{d.g[i], d.k[i]}];
  const auto r = run(Plan::scan(d.db, "fact", {"g", "k"}).aggregate({"g", "k"}, {count_star("c")}), 4);
  std::map<std::pair<std::string, std::int32_t>, std::int64_t> got;
  for (std::size_t i = 0; i < r.rows; ++i) got[{r.value(0, i), at<std::int32_t>(r, 1, i)}] = at<std::int64_t>(r, 2, i);
  CHECK(got == ref);
}

TEST(aggregate_without_groups_returns_one_row_even_when_empty) {
  Data d = make_data(1000, 10, 14);
  const auto r = run(Plan::scan(d.db, "fact", {"k", "v"}).filter(col("k") > 100).aggregate({}, {count_star("c"), sum(col("v"), "s")}), 2);
  CHECK(r.rows == 1);
  CHECK(at<std::int64_t>(r, 0, 0) == 0);
  CHECK(at<int128>(r, 1, 0) == 0);
}

// Inner join with duplicate build keys (1:N) and missing keys, more matches
// than one output vector holds, across thread counts.
TEST(inner_join_matches_nested_loop_reference) {
  Data d = make_data(20000, 500, 15);
  std::multiset<std::tuple<std::int32_t, std::int64_t, std::int32_t>> ref;
  std::multimap<std::int32_t, std::int32_t> dim;
  for (std::size_t j = 0; j < d.dk.size(); ++j) dim.emplace(d.dk[j], d.dw[j]);
  for (std::size_t i = 0; i < d.k.size(); ++i) {
    auto range = dim.equal_range(d.k[i]);
    for (auto it = range.first; it != range.second; ++it) ref.emplace(d.k[i], d.v[i], it->second);
  }
  for (int threads : {1, 4}) {
    const auto r = run(Plan::scan(d.db, "fact", {"k", "v"}).join(Plan::scan(d.db, "dim", {"dk", "w"}), {"k"}, {"dk"}),
                       threads);
    CHECK(r.names == (std::vector<std::string>{"k", "v", "w"}));
    std::multiset<std::tuple<std::int32_t, std::int64_t, std::int32_t>> got;
    for (std::size_t i = 0; i < r.rows; ++i) got.emplace(at<std::int32_t>(r, 0, i), at<std::int64_t>(r, 1, i), at<std::int32_t>(r, 2, i));
    CHECK(got == ref);
  }
}

TEST(semi_and_anti_joins_keep_each_probe_row_at_most_once) {
  Data d = make_data(20000, 500, 16);
  std::set<std::int32_t> dim(d.dk.begin(), d.dk.end());
  std::size_t semi = 0, anti = 0;
  for (std::int32_t k : d.k) (dim.count(k) ? semi : anti)++;
  const Plan build = Plan::scan(d.db, "dim", {"dk"});
  const Plan probe = Plan::scan(d.db, "fact", {"k", "s"});
  const auto rs = run(probe.join(build, {"k"}, {"dk"}, JoinType::Semi), 3);
  const auto ra = run(probe.join(build, {"k"}, {"dk"}, JoinType::Anti), 3);
  CHECK(rs.rows == semi);
  CHECK(ra.rows == anti);
  CHECK(rs.names == (std::vector<std::string>{"k", "s"}));
  bool ok = true;
  for (std::size_t i = 0; i < rs.rows; ++i) ok &= at<std::int32_t>(rs, 0, i) % 2 == 0;
  for (std::size_t i = 0; i < ra.rows; ++i) ok &= at<std::int32_t>(ra, 0, i) % 2 == 1;
  CHECK(ok);
}

// A two-column key (int32 + dictionary) packed into one hash key.
TEST(join_on_composite_key) {
  Database db;
  Table& a = db.add_table("a");
  a.add_column("x", Type::int32(), std::vector<std::int32_t>{1, 1, 2, 2, 3});
  a.add_column("y", Type::int64(), std::vector<std::int64_t>{10, 20, 10, 20, 10});
  Table& b = db.add_table("b");
  b.add_column("bx", Type::int32(), std::vector<std::int32_t>{1, 2, 3});
  b.add_column("by", Type::int64(), std::vector<std::int64_t>{20, 10, 99});
  b.add_column("tag", Type::int32(), std::vector<std::int32_t>{7, 8, 9});
  const auto r = run(Plan::scan(db, "a", {"x", "y"})
                         .join(Plan::scan(db, "b", {"bx", "by", "tag"}), {"x", "y"}, {"bx", "by"})
                         .order_by({asc("x")}),
                     1);
  CHECK(r.rows == 2);
  CHECK(at<std::int32_t>(r, 0, 0) == 1 && at<std::int64_t>(r, 1, 0) == 20 && at<std::int32_t>(r, 2, 0) == 7);
  CHECK(at<std::int32_t>(r, 0, 1) == 2 && at<std::int64_t>(r, 1, 1) == 10 && at<std::int32_t>(r, 2, 1) == 8);
}

TEST(order_by_with_limit_matches_full_sort) {
  Data d = make_data(30000, 1000, 17);
  std::vector<std::pair<std::int64_t, std::int32_t>> ref;
  for (std::size_t i = 0; i < d.v.size(); ++i) ref.emplace_back(d.v[i], d.k[i]);
  // v descending, then k ascending
  std::sort(ref.begin(), ref.end(), [](const auto& x, const auto& y) {
    return x.first != y.first ? x.first > y.first : x.second < y.second;
  });
  for (std::size_t limit : {std::size_t{0}, std::size_t{25}}) {
    const auto r = run(Plan::scan(d.db, "fact", {"v", "k"}).order_by({desc("v"), asc("k")}, limit), 4);
    const std::size_t n = limit ? limit : ref.size();
    CHECK(r.rows == n);
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok &= at<std::int64_t>(r, 0, i) == ref[i].first && at<std::int32_t>(r, 1, i) == ref[i].second;
    CHECK(ok);
  }
}

TEST(order_by_strings_and_dictionaries) {
  Data d = make_data(500, 10, 18);
  const auto r = run(Plan::scan(d.db, "fact", {"g", "s"}).order_by({desc("g"), asc("s")}, 0), 2);
  bool ok = true;
  for (std::size_t i = 1; i < r.rows; ++i) {
    const std::string g0 = r.value(0, i - 1), g1 = r.value(0, i);
    ok &= g0 > g1 || (g0 == g1 && r.value(1, i - 1) <= r.value(1, i));
  }
  CHECK(ok);
  CHECK(r.rows == 500);
}

// The same query must give identical results with 1 thread and many, and with
// and without SIMD.
TEST(results_do_not_depend_on_threads_or_simd) {
  Data d = make_data(200000, 5000, 19);
  auto plan = [&] {
    return Plan::scan(d.db, "fact", {"k", "v", "g"})
        .filter(col("v") > 1000 && in(col("g"), {"AIR", "SHIP"}))
        .join(Plan::scan(d.db, "dim", {"dk", "w"}), {"k"}, {"dk"})
        .aggregate({"g"}, {sum(col("v") * col("w"), "x"), count_star("n")})
        .order_by({asc("g")});
  };
  const std::string expect = run(plan(), 1).to_tsv();
  for (int threads : {2, 7}) CHECK(run(plan(), threads).to_tsv() == expect);
  set_simd_enabled(false);
  CHECK(run(plan(), 3).to_tsv() == expect);
  set_simd_enabled(true);
}

TEST(any_value_carries_dependent_columns) {
  Database db;
  Table& t = db.add_table("t");
  t.add_column("k", Type::int32(), std::vector<std::int32_t>{1, 2, 1, 2, 1});
  t.add_string_column("name", {"one", "two", "one", "two", "one"});
  const auto r = run(Plan::scan(db, "t", {"k", "name"})
                         .aggregate({"k"}, {any_value(col("name"), "name"), count_star("c")})
                         .order_by({asc("k")}),
                     1);
  CHECK(r.rows == 2);
  CHECK(r.value(1, 0) == "one" && at<std::int64_t>(r, 2, 0) == 3);
  CHECK(r.value(1, 1) == "two" && at<std::int64_t>(r, 2, 1) == 2);
}

TEST(explain_shows_the_operator_tree) {
  Data d = make_data(10, 4, 20);
  const std::string text = Plan::scan(d.db, "fact", {"k", "v"})
                               .join(Plan::scan(d.db, "dim", {"dk"}), {"k"}, {"dk"}, JoinType::Semi)
                               .aggregate({}, {sum(col("v"), "s")})
                               .explain();
  CHECK(text.find("HashAggregate") != std::string::npos);
  CHECK(text.find("HashJoin Semi probe(k) = build(dk)") != std::string::npos);
  CHECK(text.find("Scan dim") != std::string::npos);
}
