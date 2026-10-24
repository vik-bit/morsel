"""Generate TPC-H data with DuckDB's dbgen and write it in morsel's column format.

Usage: python scripts/gen_tpch.py --sf 1 --out data/sf1

Writes <out>/tpch.duckdb (the same data, for DuckDB to query) and one file per
column under <out>/<table>/<column>.col:

    magic   8 bytes  b"MCOL1\\0\\0\\0"
    type    u32      0 int32, 1 int64, 2 date, 3 dict, 4 string
    scale   u32      decimal digits after the point (int64 columns only)
    rows    u64
    aux     u64      dictionary size (dict) or total string bytes (string)
    payload          int32[rows] | int64[rows] |
                     aux x (u32 length, bytes) then u8 codes[rows] |
                     u64 offsets[rows + 1] then bytes

Decimals become scaled int64 (12.34 -> 1234), dates become days since
1970-01-01, and strings with at most 256 distinct values become sorted
dictionaries so that code order equals string order.
"""

import argparse
import os
import struct
import sys

import duckdb
import numpy as np

TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
MAGIC = b"MCOL1\0\0\0"
INT32, INT64, DATE, DICT, STRING = range(5)


def write_header(f, type_id, scale, rows, aux):
    f.write(MAGIC)
    f.write(struct.pack("<IIQQ", type_id, scale, rows, aux))


def convert_column(con, table, name, sql_type, path):
    rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    t = sql_type.upper()
    with open(path, "wb") as f:
        if t.startswith("DECIMAL"):
            scale = int(t.split(",")[1].rstrip(")").strip())
            arr = con.execute(
                f"SELECT CAST({name} * {10 ** scale} AS BIGINT) AS v FROM {table}"
            ).fetchnumpy()["v"]
            write_header(f, INT64, scale, rows, 0)
            f.write(np.ascontiguousarray(arr, dtype="<i8").tobytes())
        elif t in ("INTEGER", "BIGINT"):
            arr = con.execute(f"SELECT {name} AS v FROM {table}").fetchnumpy()["v"]
            arr = np.asarray(arr, dtype=np.int64)
            if arr.size == 0 or (arr.min() >= -(2**31) and arr.max() < 2**31):
                write_header(f, INT32, 0, rows, 0)
                f.write(arr.astype("<i4").tobytes())
            else:
                write_header(f, INT64, 0, rows, 0)
                f.write(arr.astype("<i8").tobytes())
        elif t == "DATE":
            arr = con.execute(
                f"SELECT CAST({name} - DATE '1970-01-01' AS INTEGER) AS v FROM {table}"
            ).fetchnumpy()["v"]
            write_header(f, DATE, 0, rows, 0)
            f.write(np.asarray(arr, dtype="<i4").tobytes())
        elif t.startswith("VARCHAR"):
            distinct = con.execute(f"SELECT count(DISTINCT {name}) FROM {table}").fetchone()[0]
            values = con.execute(f"SELECT {name} FROM {table}").fetchnumpy()[name]
            if distinct <= 256:
                dictionary, codes = np.unique(values.astype(str), return_inverse=True)
                write_header(f, DICT, 0, rows, len(dictionary))
                for s in dictionary:
                    b = s.encode()
                    f.write(struct.pack("<I", len(b)))
                    f.write(b)
                f.write(codes.astype(np.uint8).tobytes())
            else:
                encoded = [s.encode() for s in values]
                offsets = np.zeros(rows + 1, dtype="<u8")
                np.cumsum([len(b) for b in encoded], out=offsets[1:])
                write_header(f, STRING, 0, rows, int(offsets[-1]))
                f.write(offsets.tobytes())
                f.write(b"".join(encoded))
        else:
            raise SystemExit(f"unsupported type {sql_type} for {table}.{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sf", type=float, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db_path = os.path.join(args.out, "tpch.duckdb")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    print(f"generating TPC-H sf={args.sf:g} with DuckDB {duckdb.__version__} ...", flush=True)
    con.execute(f"CALL dbgen(sf={args.sf})")

    for table in TABLES:
        table_dir = os.path.join(args.out, table)
        os.makedirs(table_dir, exist_ok=True)
        columns = con.execute(f"DESCRIBE {table}").fetchall()
        with open(os.path.join(table_dir, "_schema"), "w") as schema:
            for name, sql_type, *_ in columns:
                convert_column(con, table, name, sql_type, os.path.join(table_dir, f"{name}.col"))
                schema.write(f"{name}\n")
        rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:<9} {rows:>10,} rows, {len(columns)} columns", flush=True)
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
