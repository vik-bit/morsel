// morsel command-line tool.
//
//   morsel run <data_dir> <query|all> [--threads N] [--repeat R] [--scalar] [--explain] [--profile]
//       Runs TPC-H queries, prints each result as TSV on stdout and timings on stderr.
//   morsel serve <data_dir>
//       Reads "<query> <threads> [scalar]" lines on stdin and answers each with
//       "<milliseconds> <rows>". The benchmark drives this, so data is loaded
//       once and runs of both engines can be interleaved.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "morsel/executor.h"
#include "morsel/kernels.h"
#include "morsel/tpch.h"

namespace {

using Clock = std::chrono::steady_clock;

int usage() {
  std::fprintf(stderr,
               "usage: morsel run <data_dir> <query|all> [--threads N] [--repeat R] [--scalar] [--explain] "
               "[--profile]\n"
               "       morsel serve <data_dir>\n");
  return 2;
}

std::vector<int> parse_queries(const std::string& arg) {
  if (arg == "all") return morsel::tpch_supported_queries();
  std::vector<int> qs;
  std::stringstream ss(arg);
  for (std::string item; std::getline(ss, item, ',');) {
    if (!item.empty() && (item[0] == 'q' || item[0] == 'Q')) item = item.substr(1);
    qs.push_back(std::stoi(item));
  }
  return qs;
}

double run_once(morsel::Database& db, morsel::Executor& exec, int q, morsel::MaterializedTable* out) {
  const auto t0 = Clock::now();
  const morsel::Plan plan = morsel::tpch_plan(db, q);
  morsel::MaterializedTable result = exec.run(plan);
  const auto t1 = Clock::now();
  if (out) *out = std::move(result);
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int cmd_run(int argc, char** argv) {
  if (argc < 4) return usage();
  const std::string dir = argv[2];
  const std::vector<int> queries = parse_queries(argv[3]);
  int threads = 0, repeat = 1;
  bool explain = false, profile = false;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
      threads = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) {
      repeat = std::max(1, std::atoi(argv[++i]));
    } else if (!std::strcmp(argv[i], "--scalar")) {
      morsel::set_simd_enabled(false);
    } else if (!std::strcmp(argv[i], "--explain")) {
      explain = true;
    } else if (!std::strcmp(argv[i], "--profile")) {
      profile = true;
    } else {
      return usage();
    }
  }
  morsel::Database db(dir);
  morsel::Executor exec(threads);
  std::fprintf(stderr, "morsel: %d threads, kernels: %s\n", exec.threads(),
               morsel::simd_enabled() ? morsel::simd_name() : "scalar (forced)");
  for (int q : queries) {
    const morsel::Plan plan = morsel::tpch_plan(db, q);  // loads the columns, untimed
    if (explain) std::fprintf(stderr, "Q%d plan:\n%s", q, plan.explain().c_str());
    morsel::MaterializedTable result;
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) times.push_back(run_once(db, exec, q, &result));
    std::printf("-- Q%d\n%s", q, result.to_tsv().c_str());
    std::string ts;
    for (double t : times) ts += (ts.empty() ? "" : " ") + std::to_string(t);
    std::fprintf(stderr, "Q%d: %zu rows, ms: %s\n", q, result.rows, ts.c_str());
    if (profile) {
      for (const auto& p : exec.last_profile()) std::fprintf(stderr, "  %8.2f ms  %s\n", p.ms, p.name.c_str());
    }
  }
  return 0;
}

int cmd_serve(int argc, char** argv) {
  if (argc < 3) return usage();
  morsel::Database db(argv[2]);
  std::map<int, std::unique_ptr<morsel::Executor>> executors;
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string query, mode;
    int threads = 1;
    if (!(in >> query >> threads)) continue;
    in >> mode;
    try {
      morsel::set_simd_enabled(mode != "scalar");
      auto& exec = executors[threads];
      if (!exec) exec = std::make_unique<morsel::Executor>(threads);
      const int q = parse_queries(query).at(0);
      morsel::tpch_plan(db, q);  // first use loads columns; keep that out of the timing
      morsel::MaterializedTable result;
      const double ms = run_once(db, *exec, q, &result);
      std::printf("%.4f %zu\n", ms, result.rows);
    } catch (const std::exception& e) {
      std::printf("error %s\n", e.what());
    }
    std::fflush(stdout);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  try {
    if (!std::strcmp(argv[1], "run")) return cmd_run(argc, argv);
    if (!std::strcmp(argv[1], "serve")) return cmd_serve(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return usage();
}
