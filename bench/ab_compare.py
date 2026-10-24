"""A/B-compare two morsel binaries on one query, alternating runs.

Usage: python bench/ab_compare.py OLD_BINARY NEW_BINARY --data data/sf1 --query 18 \
           [--threads 1] [--rounds 15] [--pipeline "lineitem -> aggregate"]

Each round runs both binaries once, in alternating order, each in a fresh
process. Reports the median and range of the query time, or of one
pipeline's time (matched by substring in --profile output) when --pipeline is
given. On a laptop, single runs vary too much to compare; medians over many
alternating rounds are needed to see changes of 10-20%.
"""

import argparse
import re
import statistics
import subprocess
import sys


def measure(binary, args):
    cmd = [binary, "run", args.data, str(args.query), "--threads", str(args.threads), "--profile"]
    err = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
    if args.pipeline:
        for line in err.splitlines():
            if args.pipeline in line:
                return float(line.split()[0])
        raise SystemExit(f"pipeline {args.pipeline!r} not found in:\n{err}")
    return float(re.search(r"ms: ([0-9.]+)", err).group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--data", required=True)
    parser.add_argument("--query", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--pipeline")
    args = parser.parse_args()

    times = {"old": [], "new": []}
    for r in range(args.rounds):
        order = ["old", "new"] if r % 2 == 0 else ["new", "old"]
        for name in order:
            times[name].append(measure(getattr(args, name), args))
    for name in ("old", "new"):
        t = times[name]
        print(f"{name}: median {statistics.median(t):8.2f} ms  range {min(t):.2f}-{max(t):.2f}  ({len(t)} runs)")
    print(f"new/old median ratio: {statistics.median(times['new']) / statistics.median(times['old']):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
