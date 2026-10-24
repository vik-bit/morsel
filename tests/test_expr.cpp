#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "morsel/executor.h"
#include "morsel/plan.h"

using namespace morsel;

namespace {

// A table with one row per value of i in [0, n): ints, decimals, dates, a
// dictionary column and a plain string column.
Database make_db(int n) {
  Database db;
  Table& t = db.add_table("t");
  std::vector<std::int32_t> id(n), day(n);
  std::vector<std::int64_t> price(n), disc(n);
  std::vector<std::string> color(n), note(n);
  const char* colors[] = {"red", "green", "blue", "PROMO red"};
  for (int i = 0; i < n; ++i) {
    id[i] = i;
    day[i] = parse_date("1995-01-01") + i % 400;
    price[i] = 100000 + 37 * i;  // scale 2
    disc[i] = i % 11;            // 0.00 .. 0.10
    color[i] = colors[i % 4];
    note[i] = "note " + std::to_string(i % 7);
  }
  t.add_column("id", Type::int32(), id);
  t.add_column("day", Type::date(), day);
  t.add_column("price", Type::int64(2), price);
  t.add_column("disc", Type::int64(2), disc);
  t.add_dict_column("color", color);
  t.add_string_column("note", note);
  return db;
}

MaterializedTable run(const Plan& plan, int threads = 1) {
  Executor exec(threads);
  return exec.run(plan);
}

Plan scan_all(Database& db) { return Plan::scan(db, "t", {"id", "day", "price", "disc", "color", "note"}); }

}  // namespace

TEST(dates_round_trip) {
  CHECK(parse_date("1970-01-01") == 0);
  CHECK(parse_date("1998-09-02") == 10471);
  CHECK(format_date(10471) == "1998-09-02");
  CHECK(format_date(-1) == "1969-12-31");
  for (int d = parse_date("1899-12-25"); d < parse_date("2101-01-05"); d += 13) {
    CHECK(parse_date(format_date(d)) == d);
  }
  CHECK_THROWS(parse_date("1998-13-01"), std::invalid_argument);
  CHECK_THROWS(parse_date("98-01-01"), std::invalid_argument);
}

TEST(decimals_format_and_parse_exactly) {
  CHECK(format_decimal(12345, 2) == "123.45");
  CHECK(format_decimal(-5, 2) == "-0.05");
  CHECK(format_decimal(7, 0) == "7");
  const int128 big = static_cast<int128>(1) << 100;
  CHECK(format_decimal(big, 6) == "1267650600228229401496703.205376");
  CHECK(parse_decimal("0.05").value == 5 && parse_decimal("0.05").scale == 2);
  CHECK(parse_decimal("-12.5").value == -125 && parse_decimal("-12.5").scale == 1);
  CHECK(parse_decimal("24").value == 24 && parse_decimal("24").scale == 0);
  CHECK_THROWS(parse_decimal("1.2.3"), std::invalid_argument);
}

TEST(like_matches_sql_semantics) {
  CHECK(like_match("PROMO BRUSHED", "PROMO%"));
  CHECK(!like_match("STANDARD PROMO", "PROMO%"));
  CHECK(like_match("forest green", "%green%"));
  CHECK(like_match("abc", "a_c"));
  CHECK(!like_match("abbc", "a_c"));
  CHECK(like_match("", "%"));
  CHECK(like_match("special requests", "%special%requests%"));
  CHECK(!like_match("special", "%special%requests%"));
}

// price * (1 - disc): scale 2 * scale 2 = scale 4, computed exactly.
TEST(decimal_arithmetic_aligns_scales_exactly) {
  Database db = make_db(5000);
  const auto r = run(scan_all(db).project({{"x", col("price") * (1 - col("disc"))},
                                           {"y", col("price") + dec("0.005")},
                                           {"z", col("disc") * 3 - 1}}));
  CHECK(r.types[0].id == TypeId::Int64 && r.types[0].scale == 4);
  CHECK(r.types[1].scale == 3);
  CHECK(r.types[2].scale == 2);
  bool ok = true;
  for (int i = 0; i < 5000; ++i) {
    const std::int64_t price = 100000 + 37 * i, disc = i % 11;
    ok &= r.columns[0].as<std::int64_t>()[i] == price * (100 - disc);
    ok &= r.columns[1].as<std::int64_t>()[i] == price * 10 + 5;
    ok &= r.columns[2].as<std::int64_t>()[i] == disc * 3 - 100;
  }
  CHECK(ok);
}

// Every filter gives the same rows as evaluating the condition directly,
// including literals with more decimals than the column, strings on a
// dictionary, plain strings, OR and CASE.
TEST(filters_match_direct_evaluation) {
  const int n = 9000;
  Database db = make_db(n);
  struct Case {
    P pred;
    bool (*expect)(int);
  };
  const std::vector<Case> cases = {
      {col("disc") < dec("0.055"), [](int i) { return i % 11 < 5.5; }},
      {col("disc") <= 3, [](int i) { return i % 11 <= 300; }},
      {between(col("disc"), dec("0.05"), dec("0.07")), [](int i) { return i % 11 >= 5 && i % 11 <= 7; }},
      {col("day") >= date("1995-06-01") && col("day") < date("1995-07-01"),
       [](int i) { return i % 400 >= 151 && i % 400 < 181; }},
      {col("color") == str("blue"), [](int i) { return i % 4 == 2; }},
      {col("color") != str("blue"), [](int i) { return i % 4 != 2; }},
      {like(col("color"), "PROMO%"), [](int i) { return i % 4 == 3; }},
      {in(col("color"), {"red", "green"}), [](int i) { return i % 4 < 2; }},
      {not_in(col("color"), {"red", "green"}), [](int i) { return i % 4 >= 2; }},
      {col("color") < str("green"), [](int i) { return i % 4 == 2 || i % 4 == 3; }},  // "PROMO red" < "blue"
      {col("note") == str("note 3"), [](int i) { return i % 7 == 3; }},
      {like(col("note"), "%5"), [](int i) { return i % 7 == 5; }},
      {col("id") < 100 || col("id") >= 8950 || col("color") == str("red"),
       [](int i) { return i < 100 || i >= 8950 || i % 4 == 0; }},
      {(col("id") < 10 && col("disc") == 0) || (col("id") > 8990 && col("disc") > dec("0.05")),
       [](int i) { return (i < 10 && i % 11 == 0) || (i > 8990 && i % 11 > 5); }},
      {col("price") * (1 - col("disc")) > 2500,
       [](int i) { return (100000LL + 37 * i) * (100 - i % 11) > 25000000LL; }},
      {col("price") > col("disc") * 50000, [](int i) { return 100000LL + 37 * i > (i % 11) * 50000LL; }},
      {col("day") < col("day"), [](int) { return false; }},
  };
  for (std::size_t c = 0; c < cases.size(); ++c) {
    for (int threads : {1, 3}) {
      const auto r = run(scan_all(db).filter(cases[c].pred).select({"id"}), threads);
      std::vector<bool> seen(n, false);
      for (std::size_t k = 0; k < r.rows; ++k) seen[static_cast<std::size_t>(r.columns[0].as<std::int32_t>()[k])] = true;
      bool ok = true;
      for (int i = 0; i < n; ++i) ok &= seen[static_cast<std::size_t>(i)] == cases[c].expect(i);
      if (!ok) std::fprintf(stderr, "  filter case %zu (threads %d) wrong\n", c, threads);
      CHECK(ok);
    }
  }
}

TEST(case_when_picks_per_row) {
  Database db = make_db(3000);
  const auto r = run(scan_all(db).project(
      {{"v", case_when(col("color") == str("red"), col("price"), 0)},
       {"w", case_when(col("id") < 1500, 1, 0)}}));
  bool ok = true;
  for (int i = 0; i < 3000; ++i) {
    ok &= r.columns[0].as<std::int64_t>()[i] == (i % 4 == 0 ? 100000 + 37 * i : 0);
    ok &= r.columns[1].as<std::int64_t>()[i] == (i < 1500 ? 1 : 0);
  }
  CHECK(ok);
}

TEST(binding_errors_are_reported_when_the_plan_is_built) {
  Database db = make_db(10);
  CHECK_THROWS(scan_all(db).filter(col("nope") < 3), std::invalid_argument);
  CHECK_THROWS(scan_all(db).project({{"x", col("price") * col("price") * col("price") * col("price") *
                                                col("price") * col("price") * col("price") * col("price") *
                                                col("price") * col("price")}}),
               std::invalid_argument);
  CHECK_THROWS(scan_all(db).filter(col("note") < 3), std::invalid_argument);
  CHECK_THROWS(scan_all(db).project({{"id", col("id")}, {"id", col("day")}}), std::invalid_argument);
  CHECK_THROWS(scan_all(db).aggregate({"note"}, {count_star("n")}), std::invalid_argument);
}
