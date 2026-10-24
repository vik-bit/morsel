"""Benchmark morsel against DuckDB on TPC-H, with interleaved repeated runs.

Usage: python bench/run_bench.py --morsel build/morsel --data data/sf1 \
           [--threads 1 12] [--reps 10] [--scalar] [--out bench/results/sf1.json]

Both engines hold the data in memory before any timing starts: DuckDB copies
<data>/tpch.duckdb into an in-memory database, and `morsel serve` loads the
column files it needs on each query's warm-up run. Per (threads, query), each
engine gets one untimed warm-up, then `reps` timed runs in alternating order
(morsel first on even reps, DuckDB first on odd ones), so slow drift in clock
speed or background load hits both engines alike.

What each time covers:
  DuckDB: con.execute(sql).fetchall() from Python: parse, optimize, execute,
          and convert the (small) result to Python objects.
  morsel: building the C++ plan (binding) and executing it, measured inside
          the morsel process; the pipe round trip is not included.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import duckdb

QUERIES = [1, 3, 5, 6, 12, 14, 18, 19]


class Morsel:
    def __init__(self, binary, data):
        self.proc = subprocess.Popen([binary, "serve", data], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

    def run(self, query, threads, scalar=False):
        self.proc.stdin.write(f"{query} {threads}{' scalar' if scalar else ''}\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline().split()
        if not reply or reply[0] == "error":
            raise RuntimeError(f"morsel failed on Q{query}: {' '.join(reply)}")
        return float(reply[0])

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


class DuckDB:
    def __init__(self, data):
        self.con = duckdb.connect()
        path = os.path.join(data, "tpch.duckdb").replace("'", "''")
        self.con.execute(f"ATTACH '{path}' AS src (READ_ONLY)")
        self.con.execute("COPY FROM DATABASE src TO memory")
        self.con.execute("DETACH src")
        self.con.execute("LOAD tpch")
        self.sql = dict(self.con.execute("SELECT query_nr, query FROM tpch_queries()").fetchall())
        self.threads = None

    def run(self, query, threads):
        if threads != self.threads:
            self.con.execute(f"SET threads = {threads}")
            self.threads = threads
        t0 = time.perf_counter()
        self.con.execute(self.sql[query]).fetchall()
        return (time.perf_counter() - t0) * 1000.0


def summarize(times):
    return {"median": statistics.median(times), "min": min(times), "max": max(times), "runs": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--morsel", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, os.cpu_count() or 4])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--queries", type=int, nargs="+", default=QUERIES)
    parser.add_argument("--scalar", action="store_true",
                        help="also time morsel with SIMD disabled (single-threaded configs only)")
    parser.add_argument("--out", help="write raw results as JSON here")
    args = parser.parse_args()

    print(f"loading {args.data} into DuckDB {duckdb.__version__} ...", flush=True)
    duck = DuckDB(args.data)
    morsel = Morsel(args.morsel, args.data)
    results = []
    try:
        for threads in args.threads:
            for q in args.queries:
                engines = {"morsel": lambda: morsel.run(q, threads), "duckdb": lambda: duck.run(q, threads)}
                if args.scalar and threads == 1:
                    engines["morsel-scalar"] = lambda: morsel.run(q, threads, scalar=True)
                for run in engines.values():
                    run()  # warm-up: loads columns, warms caches
                times = {name: [] for name in engines}
                names = list(engines)
                for rep in range(args.reps):
                    order = names if rep % 2 == 0 else names[::-1]
                    for name in order:
                        times[name].append(engines[name]())
                row = {"query": q, "threads": threads, **{n: summarize(t) for n, t in times.items()}}
                results.append(row)
                ratio = row["duckdb"]["median"] / row["morsel"]["median"]
                cells = "  ".join(f"{n} {row[n]['median']:8.1f} ms [{row[n]['min']:.1f}-{row[n]['max']:.1f}]"
                                  for n in names)
                print(f"threads={threads:<2} Q{q:<2}  {cells}  duckdb/morsel {ratio:.2f}x", flush=True)
    finally:
        morsel.close()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        meta = {
            "duckdb_version": duckdb.__version__,
            "data": os.path.abspath(args.data),
            "reps": args.reps,
            "machine": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        }
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
