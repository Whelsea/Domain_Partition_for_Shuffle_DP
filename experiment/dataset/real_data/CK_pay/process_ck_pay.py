#!/usr/bin/env python3
"""
Process the Cook County employee payroll dataset into UserLevelDataset CSV format.

Raw data: Employee_Payroll.csv
    Columns include fiscal period, employee attributes, bureau, and Base Pay.
    The local file contains 234,299 rows covering 2016Q1 to 2018Q2.

User definition
---------------
Each user is identified by the pair:
    (Bureau, Employee Identifier)

Record definition
-----------------
Each retained payroll row contributes one record equal to its rounded,
nonnegative Base Pay value. We intentionally keep row-level entries rather
than aggregating within quarter, matching the construction used in prior work.

Encoding
--------
    drop the row if Base Pay < 0
    raw_value = round(Base Pay to nearest integer USD)
    value = raw_value % (U + 1)

Rows with missing or invalid salary are skipped.

Usage:
    python process_ck_pay.py --n 5000 --M 14 --U 242576

Output:
    data/ckpay_n5000_M14_U242576.csv
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


REQUIRED_COLUMNS = {
    "Bureau",
    "Employee Identifier",
    "Base Pay",
}


def parse_ck_pay_raw(
    raw_path: str,
    n: int,
    M: int,
    U: int,
) -> UserLevelDataset:
    """Parse Employee_Payroll.csv and produce a UserLevelDataset.

    Users are selected in order of first valid appearance in the raw CSV.
    For each selected user, at most the first M retained salary records are kept.
    """
    user_records: dict[tuple[str, str], list[int]] = {}
    total_rows = 0
    missing_salary_rows = 0
    invalid_salary_rows = 0
    negative_salary_rows = 0
    valid_salary_rows = 0

    with open(raw_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        missing_cols = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"CSV header missing required columns: {sorted(missing_cols)}"
            )

        for row in reader:
            total_rows += 1
            salary_str = row["Base Pay"].strip()
            if not salary_str:
                missing_salary_rows += 1
                continue

            try:
                salary = Decimal(salary_str)
            except (InvalidOperation, ValueError):
                invalid_salary_rows += 1
                continue

            if salary < 0:
                negative_salary_rows += 1
                continue

            raw_value = int(salary.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

            bureau = row["Bureau"].strip()
            employee_id = row["Employee Identifier"].strip()
            user_key = (bureau, employee_id)
            if user_key not in user_records:
                user_records[user_key] = []

            valid_salary_rows += 1
            if len(user_records[user_key]) < M:
                user_records[user_key].append(raw_value % (U + 1))

    selected_records = list(user_records.values())[:n]

    metadata: dict[str, Any] = {
        "source": "ck-payroll",
        "raw_file": os.path.basename(raw_path),
        "encoding": "salary_dollars_rounded",
        "user_key": "Bureau+Employee Identifier",
        "record_definition": "rounded nonnegative Base Pay per payroll row",
        "selection_order": "first_valid_appearance",
        "n_requested": n,
        "n_available": len(user_records),
        "M": M,
        "U": U,
        "total_rows": total_rows,
        "valid_salary_rows_before_clipping": valid_salary_rows,
        "skipped_missing_salary": missing_salary_rows,
        "skipped_invalid_salary": invalid_salary_rows,
        "skipped_negative_salary": negative_salary_rows,
    }

    return UserLevelDataset(
        records=selected_records,
        n=len(selected_records),
        M=M,
        U=U,
        metadata=metadata,
    )


DEFAULT_RAW_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Employee_Payroll.csv",
)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process Cook County payroll data into UserLevelDataset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_data", type=str, default=DEFAULT_RAW_PATH,
                   help="Path to Employee_Payroll.csv.")
    p.add_argument("--n", type=int, required=True, help="Number of users.")
    p.add_argument("--M", type=int, required=True, help="Max records per user.")
    p.add_argument("--U", type=int, required=True,
                   help="Domain upper bound (values in {0,...,U}).")
    p.add_argument("--output", type=str, default=None,
                   help="Exact output path. Overrides auto-naming.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for auto-named output. Default: real_data/CK_pay/data/")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.raw_data):
        print(f"Error: raw data file not found: {args.raw_data}", file=sys.stderr)
        sys.exit(1)

    ds = parse_ck_pay_raw(
        raw_path=args.raw_data,
        n=args.n,
        M=args.M,
        U=args.U,
    )

    if args.output:
        out_path = args.output
    else:
        out_dir = args.output_dir or DEFAULT_DATA_DIR
        out_path = os.path.join(out_dir, f"ckpay_n{ds.n}_M{args.M}_U{args.U}.csv")

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
