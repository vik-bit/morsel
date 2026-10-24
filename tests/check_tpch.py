"""Check morsel's TPC-H answers against DuckDB's, value by value.

Usage: python tests/check_tpch.py --morsel build/morsel --data data/sf0.01 [--threads 1 4] [--scalar]

Runs every implemented query through `morsel run` (once per thread count, and
once more with the scalar kernels when --scalar is given) and through DuckDB
on the same generated data (<data>/tpch.duckdb). Integers, decimals, dates and
strings must match exactly. DOUBLE columns (averages, Q14's ratio) must match
to a relative 1e-12: both engines compute them from the same exact sums, but
the final division may round differently in the last bit.
"""

import argparse
import datetime
import decimal
import math
import os
import subprocess
import sys

import duckdb

QUERIES = [1, 3, 5, 6, 12, 14, 18, 19]
REL_TOL = 1e-12


def parse_morsel_output(text):
    results, current = {}, None
    for line in text.splitlines():
        if line.startswith("-- Q"):
            current = int(line[4:])
            results[current] = {"header": None, "rows": []}
        elif current is not None:
            fields = line.split("\t")
            if results[current]["header"] is None:
                results[current]["header"] = fields
            else:
                results[current]["rows"].append(fields)
    return results


def values_equal(ours, theirs):
    """Returns (equal, exact_comparison)."""
    if isinstance(theirs, bool) or theirs is None:
        return False, True
    if isinstance(theirs, decimal.Decimal):
        return decimal.Decimal(ours) == theirs, True
    if isinstance(theirs, int):
        return decimal.Decimal(ours) == theirs, True
    if isinstance(theirs, float):
        o = float(ours)
        return math.isclose(o, theirs, rel_tol=REL_TOL, abs_tol=1e-300), False
    if isinstance(theirs, datetime.date):
        return ours == theirs.isoformat(), True
    return ours == str(theirs), True


def check(ours, expected):
    errors, exact, approx = [], 0, 0
    if len(ours["rows"]) != len(expected):
        return [f"row count {len(ours['rows'])} != DuckDB {len(expected)}"], 0, 0
    for r, (orow, erow) in enumerate(zip(ours["rows"], expected)):
        if len(orow) != len(erow):
            errors.append(f"row {r}: {len(orow)} columns != DuckDB {len(erow)}")
            continue
        for c, (o, e) in enumerate(zip(orow, erow)):
            ok, is_exact = values_equal(o, e)
            if not ok:
                errors.append(f"row {r} column {ours['header'][c]}: morsel {o!r} != DuckDB {e!r}")
            elif is_exact:
                exact += 1
            else:
                approx += 1
    return errors, exact, approx


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--morsel", required=True, help="path to the morsel binary")
    parser.add_argument("--data", required=True, help="directory written by scripts/gen_tpch.py")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, os.cpu_count() or 4])
    parser.add_argument("--scalar", action="store_true", help="also check with SIMD disabled")
    args = parser.parse_args()

    con = duckdb.connect(os.path.join(args.data, "tpch.duckdb"), read_only=True)
    con.execute("LOAD tpch")
    sql = dict(con.execute("SELECT query_nr, query FROM tpch_queries()").fetchall())
    expected = {q: con.execute(sql[q]).fetchall() for q in QUERIES}

    configs = [(t, False) for t in args.threads] + ([(args.threads[-1], True)] if args.scalar else [])
    failures = 0
    for threads, scalar in configs:
        cmd = [args.morsel, "run", args.data, ",".join(map(str, QUERIES)), "--threads", str(threads)]
        if scalar:
            cmd.append("--scalar")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(proc.stderr)
            return 1
        ours = parse_morsel_output(proc.stdout)
        label = f"threads={threads}{' scalar' if scalar else ''}"
        for q in QUERIES:
            errors, exact, approx = check(ours.get(q, {"rows": [], "header": []}), expected[q])
            status = "FAIL" if errors else " OK "
            print(f"[{status}] Q{q:<2} {label:<18} {len(expected[q]):>3} rows, "
                  f"{exact} values exact, {approx} doubles within {REL_TOL:g}")
            for e in errors[:5]:
                print(f"         {e}")
            failures += bool(errors)
    print("all queries match DuckDB" if failures == 0 else f"{failures} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
