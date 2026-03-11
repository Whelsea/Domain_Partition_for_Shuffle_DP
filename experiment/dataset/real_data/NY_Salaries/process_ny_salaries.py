#!/usr/bin/env python3
"""
Process the New York State Authorities salary dataset into UserLevelDataset CSV format.

Raw data:
    Salary_Information_for_State_Authorities.csv

User definition
---------------
Each user is identified by the normalized tuple:
    (Authority Name, Last Name, First Name, Middle Initial)

Record definition
-----------------
Each retained salary row contributes one record equal to the rounded,
nonnegative Actual Salary Paid value. We intentionally keep multiple salary
rows for the same user in the same fiscal year as separate records.

Encoding
--------
    rounded_salary = round(Actual Salary Paid to nearest integer USD)
    value = rounded_salary % (U + 1)

Rows with missing salary are skipped. Negative salaries are dropped.

Usage:
    python process_ny_salaries.py --n 204682 --M 1048576 --U 3370819

Output:
    data/nysalaries_n204682_M1048576_U3370819.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

# Import UserLevelDataset from our canonical module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulated_data"))
from dataset import UserLevelDataset


RAW_FILE = "Salary_Information_for_State_Authorities.csv"
REQUIRED_COLUMNS = {
    "Authority Name",
    "Fiscal Year End Date",
    "Last Name",
    "Middle Initial",
    "First Name",
    "Actual Salary Paid",
}
_WS_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize whitespace and non-breaking spaces."""
    value = value.replace("\xa0", " ")
    return _WS_RE.sub(" ", value.strip())


def parse_salary(value: str) -> Decimal:
    """Parse a salary-like string such as '12345.67'."""
    cleaned = normalize_text(value).replace("$", "").replace(",", "")
    return Decimal(cleaned)


def parse_ny_salaries_raw(
    raw_path: str,
    n: int,
    M: int,
    U: int,
) -> UserLevelDataset:
    """Parse the NY salary file into a UserLevelDataset."""
    user_records: dict[tuple[str, str, str, str], list[int]] = {}
    total_rows = 0
    missing_salary_rows = 0
    invalid_salary_rows = 0
    negative_salary_rows = 0
    zero_salary_rows = 0
    duplicate_user_year_rows = 0
    valid_salary_rows = 0
    seen_user_year: set[tuple[str, str, str, str, str]] = set()

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

            salary_str = normalize_text(row["Actual Salary Paid"])
            if not salary_str:
                missing_salary_rows += 1
                continue

            try:
                salary = parse_salary(salary_str)
            except (InvalidOperation, ValueError):
                invalid_salary_rows += 1
                continue

            if salary < 0:
                negative_salary_rows += 1
                continue

            rounded_salary = int(
                salary.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )
            if rounded_salary == 0:
                zero_salary_rows += 1

            authority = normalize_text(row["Authority Name"]).lower()
            last_name = normalize_text(row["Last Name"]).lower()
            first_name = normalize_text(row["First Name"]).lower()
            middle_initial = normalize_text(row["Middle Initial"]).lower()
            year = normalize_text(row["Fiscal Year End Date"])[-4:]
            user_key = (authority, last_name, first_name, middle_initial)
            user_year_key = user_key + (year,)

            if user_year_key in seen_user_year:
                duplicate_user_year_rows += 1
            else:
                seen_user_year.add(user_year_key)

            if user_key not in user_records:
                user_records[user_key] = []

            valid_salary_rows += 1
            if len(user_records[user_key]) < M:
                user_records[user_key].append(rounded_salary % (U + 1))

    selected_records = list(user_records.values())[:n]

    metadata: dict[str, Any] = {
        "source": "ny-state-authorities-salaries",
        "raw_file": os.path.basename(raw_path),
        "encoding": "actual_salary_paid_rounded",
        "user_key": "Authority+Last+First+Middle",
        "record_definition": "rounded nonnegative Actual Salary Paid per row",
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
        "zero_salary_rows": zero_salary_rows,
        "duplicate_user_year_rows": duplicate_user_year_rows,
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
    RAW_FILE,
)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process NY salary data into UserLevelDataset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_data", type=str, default=DEFAULT_RAW_PATH,
                   help=f"Path to {RAW_FILE}.")
    p.add_argument("--n", type=int, required=True, help="Number of users.")
    p.add_argument("--M", type=int, required=True, help="Max records per user.")
    p.add_argument("--U", type=int, required=True,
                   help="Domain upper bound (values in {0,...,U}).")
    p.add_argument("--output", type=str, default=None,
                   help="Exact output path. Overrides auto-naming.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for auto-named output. Default: real_data/NY_Salaries/data/")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.raw_data):
        print(f"Error: raw data file not found: {args.raw_data}", file=sys.stderr)
        sys.exit(1)

    ds = parse_ny_salaries_raw(
        raw_path=args.raw_data,
        n=args.n,
        M=args.M,
        U=args.U,
    )

    if args.output:
        out_path = args.output
    else:
        out_dir = args.output_dir or DEFAULT_DATA_DIR
        out_path = os.path.join(out_dir, f"nysalaries_n{ds.n}_M{args.M}_U{args.U}.csv")

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
