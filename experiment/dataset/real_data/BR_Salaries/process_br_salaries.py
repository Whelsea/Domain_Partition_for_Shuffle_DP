#!/usr/bin/env python3
"""
Process the Brazilian payroll snapshot into UserLevelDataset CSV format.

Raw data:
    data_raw.csv

User definition
---------------
Each user is one payroll row, identified by the unique ``Id`` column.

Record definition
-----------------
Each retained payroll row contributes one record for every strictly positive,
nonzero payroll component among:
    - Month_salary
    - 13_salary
    - eventual_salary
    - indemnity
    - extra_salary

We intentionally exclude:
    - discount_salary  (deduction; typically negative)
    - total_salary     (aggregate of the component values)

Encoding
--------
Each record is encoded in a type-aware way so equal monetary amounts from
different component types do not collide:

    raw_value = component_offset[type] + round(component_amount)
    value = raw_value % (U + 1)

Collision-free U for the local file is 816048.

Data cleaning
-------------
Some rows contain an unquoted comma inside the text sector field, which shifts
the numeric columns to the right. For rows of length 11, we repair the parse by
merging the split sector token back into a single field.

Usage:
    python process_br_salaries.py --n 1084364 --M 1048576 --U 816048

Output:
    data/brsalaries_n1084364_M1048576_U816048.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

# Import UserLevelDataset from our canonical module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulated_data"))
from dataset import UserLevelDataset


RAW_HEADER = [
    "Id",
    "job",
    "sector",
    "Month_salary",
    "13_salary",
    "eventual_salary",
    "indemnity",
    "extra_salary",
    "discount_salary",
    "total_salary",
]
COMPONENT_FIELDS = [
    "Month_salary",
    "13_salary",
    "eventual_salary",
    "indemnity",
    "extra_salary",
]
COMPONENT_MAX = {
    "Month_salary": 131128,
    "13_salary": 86381,
    "eventual_salary": 155862,
    "indemnity": 327498,
    "extra_salary": 115175,
}


def _build_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    running = 0
    for field in COMPONENT_FIELDS:
        offsets[field] = running
        running += COMPONENT_MAX[field] + 1
    return offsets


COMPONENT_OFFSETS = _build_offsets()
COLLISION_FREE_U = sum(COMPONENT_MAX[f] + 1 for f in COMPONENT_FIELDS) - 1


def _repair_row(row: list[str]) -> list[str] | None:
    """Repair a row if possible, otherwise return None."""
    if len(row) == len(RAW_HEADER):
        return row
    if len(row) == len(RAW_HEADER) + 1:
        return [row[0], row[1], f"{row[2].strip()}, {row[3].strip()}", *row[4:]]
    return None


def _round_amount(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def parse_br_salaries_raw(
    raw_path: str,
    n: int,
    M: int,
    U: int,
) -> UserLevelDataset:
    """Parse the Brazilian payroll snapshot into a UserLevelDataset."""
    rows_out: list[list[int]] = []
    total_rows = 0
    repaired_rows = 0
    skipped_bad_length = 0
    skipped_invalid_numeric = 0
    skipped_no_positive_components = 0

    with open(raw_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header

        for row in reader:
            total_rows += 1
            fixed = _repair_row(row)
            if fixed is None:
                skipped_bad_length += 1
                continue
            if len(row) == len(RAW_HEADER) + 1:
                repaired_rows += 1

            try:
                numeric = {
                    field: Decimal(value.strip())
                    for field, value in zip(RAW_HEADER[3:], fixed[3:])
                }
            except (InvalidOperation, ValueError):
                skipped_invalid_numeric += 1
                continue

            user_records: list[int] = []
            for field in COMPONENT_FIELDS:
                value = numeric[field]
                if value <= 0:
                    continue
                rounded = _round_amount(value)
                encoded = (COMPONENT_OFFSETS[field] + rounded) % (U + 1)
                user_records.append(encoded)

            if not user_records:
                skipped_no_positive_components += 1
                continue

            rows_out.append(user_records[:M])
            if len(rows_out) >= n:
                break

    metadata: dict[str, Any] = {
        "source": "br-payroll-snapshot",
        "raw_file": os.path.basename(raw_path),
        "encoding": "component_type_plus_rounded_value",
        "user_key": "Id",
        "record_definition": "positive payroll components per row",
        "component_fields": "+".join(COMPONENT_FIELDS),
        "excluded_fields": "discount_salary+total_salary",
        "selection_order": "raw_file_order",
        "n_requested": n,
        "n_available": len(rows_out),
        "M": M,
        "U": U,
        "collision_free_U": COLLISION_FREE_U,
        "total_rows": total_rows,
        "repaired_rows": repaired_rows,
        "skipped_bad_length": skipped_bad_length,
        "skipped_invalid_numeric": skipped_invalid_numeric,
        "skipped_no_positive_components": skipped_no_positive_components,
    }

    return UserLevelDataset(
        records=rows_out,
        n=len(rows_out),
        M=M,
        U=U,
        metadata=metadata,
    )


DEFAULT_RAW_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data_raw.csv",
)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process BR payroll snapshot into UserLevelDataset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_data", type=str, default=DEFAULT_RAW_PATH,
                   help="Path to data_raw.csv.")
    p.add_argument("--n", type=int, required=True, help="Number of users.")
    p.add_argument("--M", type=int, required=True, help="Max records per user.")
    p.add_argument("--U", type=int, required=True,
                   help="Domain upper bound (values in {0,...,U}).")
    p.add_argument("--output", type=str, default=None,
                   help="Exact output path. Overrides auto-naming.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for auto-named output. Default: real_data/BR_Salaries/data/")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.raw_data):
        print(f"Error: raw data file not found: {args.raw_data}", file=sys.stderr)
        sys.exit(1)

    ds = parse_br_salaries_raw(
        raw_path=args.raw_data,
        n=args.n,
        M=args.M,
        U=args.U,
    )

    if args.output:
        out_path = args.output
    else:
        out_dir = args.output_dir or DEFAULT_DATA_DIR
        out_path = os.path.join(out_dir, f"brsalaries_n{ds.n}_M{args.M}_U{args.U}.csv")

    ds.save_csv(out_path)

    if not args.quiet:
        print(ds.summary())
        print(f"\n  Saved to: {out_path}")
        size_bytes = os.path.getsize(out_path)
        if size_bytes < 1024:
            print(f"  File size: {size_bytes} B")
        elif size_bytes < 1024 * 1024:
            print(f"  File size: {size_bytes / 1024:.1f} KB")
        else:
            print(f"  File size: {size_bytes / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
