"""Summarize benchmark runs into Markdown tables.

Usage: python bench/summarize.py bench/results/sf1/run*.json

Each run file holds, per (threads, query), the median of that run's timed
repetitions for each engine. Across runs, the tables show the range of those
medians ("lo-hi"), so a cell with a wide range is one this machine cannot
measure precisely. The speedup column is DuckDB's median divided by morsel's,
computed per run and shown as a range.
"""

import json
import sys
from collections import defaultdict


def fmt_range(values, digits=1):
    lo, hi = min(values), max(values)
    if f"{lo:.{digits}f}" == f"{hi:.{digits}f}":
        return f"{lo:.{digits}f}"
    return f"{lo:.{digits}f}-{hi:.{digits}f}"


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    cells = defaultdict(lambda: defaultdict(list))  # (threads, query) -> engine -> [median per run]
    for run in runs:
        for row in run["results"]:
            key = (row["threads"], row["query"])
            for engine in ("morsel", "duckdb", "morsel-scalar"):
                if engine in row:
                    cells[key][engine].append(row[engine]["median"])
            cells[key]["ratio"].append(row["duckdb"]["median"] / row["morsel"]["median"])

    meta = runs[0]["meta"]
    print(f"{len(runs)} runs x {meta['reps']} reps, DuckDB {meta['duckdb_version']}, {meta['cpu_count']} hardware threads\n")
    for threads in sorted({t for t, _ in cells}):
        print(f"**{threads} thread{'s' if threads > 1 else ''}** (ms, median per run; range across runs)\n")
        print("| Query | morsel | DuckDB | DuckDB / morsel |")
        print("|---|---|---|---|")
        for (t, q), c in sorted(cells.items()):
            if t != threads:
                continue
            print(f"| Q{q} | {fmt_range(c['morsel'])} | {fmt_range(c['duckdb'])} | {fmt_range(c['ratio'], 2)}x |")
        print()

    single = {q: c for (t, q), c in cells.items() if t == 1 and c.get("morsel-scalar")}
    if single:
        print("**SIMD vs scalar kernels, 1 thread** (ms)\n")
        print("| Query | AVX2 | scalar | scalar / AVX2 |")
        print("|---|---|---|---|")
        for q, c in sorted(single.items()):
            ratios = [s / v for s, v in zip(c["morsel-scalar"], c["morsel"])]
            print(f"| Q{q} | {fmt_range(c['morsel'])} | {fmt_range(c['morsel-scalar'])} | {fmt_range(ratios, 2)}x |")
        print()

    scaling = defaultdict(dict)
    for (t, q), c in cells.items():
        scaling[q][t] = c
    ts = sorted({t for t, _ in cells})
    if len(ts) == 2:
        lo, hi = ts
        print(f"**Speedup from {lo} to {hi} threads** (ratio of medians, range across runs)\n")
        print("| Query | morsel | DuckDB |")
        print("|---|---|---|")
        for q in sorted(scaling):
            m = [a / b for a, b in zip(scaling[q][lo]["morsel"], scaling[q][hi]["morsel"])]
            d = [a / b for a, b in zip(scaling[q][lo]["duckdb"], scaling[q][hi]["duckdb"])]
            print(f"| Q{q} | {fmt_range(m, 1)}x | {fmt_range(d, 1)}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
