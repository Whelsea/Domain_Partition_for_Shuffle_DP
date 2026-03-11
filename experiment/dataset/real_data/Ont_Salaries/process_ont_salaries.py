#!/usr/bin/env python3
"""
Process the Ontario public sector salary datasets into UserLevelDataset CSV format.

Raw data:
    ontario-public-sector-salary-2016.csv
    ontario-public-sector-salary-2017.csv
    ontario-public-sector-salary-2018.csv
    ontario-public-sector-salary-2019.csv

User definition
---------------
Each user is identified by the normalized tuple:
    (First Name, Last Name, Employer)

Record definition
-----------------
Each salary row contributes one record equal to the rounded annual salary.
We intentionally keep multiple salary rows for the same user in the same year
as separate records (no within-year aggregation).

Encoding
--------
The Ontario sunshine list contains salaries at or above $100,000. We encode:

    rounded_salary = round(Salary Paid to nearest integer USD)
    raw_value = rounded_salary - 100000
    value = raw_value % (U + 1)

This shift reduces the collision-free domain upper bound from 1,746,825 to
1,646,825 while preserving all salary distinctions.

Data cleaning
-------------
- Normalize whitespace (including non-breaking spaces) in user-identifying text.
- Lowercase the user key for grouping.
- Repair malformed rows whose year field is corrupted by inferring the year from
  the source filename.
- Ignore Taxable Benefits entirely; this processor uses Salary Paid only.

Usage:
    python process_ont_salaries.py --n 273292 --M 1048576 --U 1646825

Output:
    data/ontsalaries_n273292_M1048576_U1646825.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

# Import UserLevelDataset from our canonical module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulated_data"))
from dataset import UserLevelDataset


FILE_GLOB = "ontario-public-sector-salary-*.csv"
SALARY_FLOOR = 100000
EXPECTED_COLUMNS = 8
_WS_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize whitespace and non-breaking spaces."""
    value = value.replace("\xa0", " ")
    return _WS_RE.sub(" ", value.strip())


def parse_salary(value: str) -> Decimal:
    """Parse a salary-like string such as '$105,595.39'."""
    cleaned = normalize_text(value).replace("$", "").replace(",", "").replace(" ", "")
    return Decimal(cleaned)


def parse_ont_salaries_raw(
    raw_dir: str,
    n: int,
    M: int,
    U: int,
) -> UserLevelDataset:
    """Parse the Ontario sunshine list files into a UserLevelDataset.

    Users are grouped by normalized (first_name, last_name, employer).
    For each user, up to the first M salary records are kept after cleaning.
    Selected users are ordered lexicographically by their normalized key.
    """
    base = Path(raw_dir)
    files = sorted(base.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matching {FILE_GLOB!r} found in {raw_dir}")

    user_records: dict[tuple[str, str, str], list[int]] = {}
    repaired_rows = 0
    skipped_bad_length = 0
    skipped_invalid_salary = 0
    skipped_below_floor = 0
    total_rows = 0

    for path in files:
        file_year = int(path.stem.rsplit("-", 1)[-1])
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header

            for row in reader:
                if len(row) != EXPECTED_COLUMNS:
                    skipped_bad_length += 1
                    continue

                total_rows += 1
                _, last_name, first_name, salary_str, _, employer, job_title, year_str = row
                first_name = normalize_text(first_name)
                last_name = normalize_text(last_name)
                employer = normalize_text(employer)
                job_title = normalize_text(job_title)
                year_str = normalize_text(year_str)

                if not year_str.isdigit():
                    # Observed in a small number of 2016 rows where part of the
                    # employer/title spills into the year column. We preserve the
                    # row by using the filename year and folding the extra token
                    # into the employer string so the user key remains stable.
                    employer = normalize_text(f"{employer}, {job_title}")
                    repaired_rows += 1

                try:
                    salary = parse_salary(salary_str)
                except (InvalidOperation, ValueError):
                    skipped_invalid_salary += 1
                    continue

                rounded_salary = int(
                    salary.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                )
                raw_value = rounded_salary - SALARY_FLOOR
                if raw_value < 0:
                    skipped_below_floor += 1
                    continue

                user_key = (
                    first_name.lower(),
                    last_name.lower(),
                    employer.lower(),
                )
                if user_key not in user_records:
                    user_records[user_key] = []
                if len(user_records[user_key]) < M:
                    user_records[user_key].append(raw_value % (U + 1))

    selected_keys = sorted(user_records.keys())[:n]
    selected_records = [user_records[key] for key in selected_keys]

    metadata: dict[str, Any] = {
        "source": "ontario-sunshine-list",
        "raw_dir": os.path.basename(os.path.abspath(raw_dir)),
        "files": len(files),
        "years": "2016-2019",
        "encoding": "salary_paid_rounded_shifted",
        "salary_floor": SALARY_FLOOR,
        "user_key": "normalized First Name+Last Name+Employer",
        "record_definition": "Salary Paid per row (no within-year aggregation)",
        "selection_order": "sorted_normalized_user_key",
        "n_requested": n,
        "n_available": len(user_records),
        "M": M,
        "U": U,
        "total_rows": total_rows,
        "repaired_rows": repaired_rows,
        "skipped_bad_length": skipped_bad_length,
        "skipped_invalid_salary": skipped_invalid_salary,
        "skipped_below_floor": skipped_below_floor,
    }

    return UserLevelDataset(
        records=selected_records,
        n=len(selected_records),
        M=M,
        U=U,
        metadata=metadata,
    )


DEFAULT_RAW_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process Ontario sunshine list data into UserLevelDataset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_data", type=str, default=DEFAULT_RAW_DIR,
                   help="Directory containing the Ontario salary CSV files.")
    p.add_argument("--n", type=int, required=True, help="Number of users.")
    p.add_argument("--M", type=int, required=True, help="Max records per user.")
    p.add_argument("--U", type=int, required=True,
                   help="Domain upper bound (values in {0,...,U}).")
    p.add_argument("--output", type=str, default=None,
                   help="Exact output path. Overrides auto-naming.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for auto-named output. Default: real_data/Ont_Salaries/data/")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.isdir(args.raw_data):
        print(f"Error: raw data directory not found: {args.raw_data}", file=sys.stderr)
        sys.exit(1)

    ds = parse_ont_salaries_raw(
        raw_dir=args.raw_data,
        n=args.n,
        M=args.M,
        U=args.U,
    )

    if args.output:
        out_path = args.output
    else:
        out_dir = args.output_dir or DEFAULT_DATA_DIR
        out_path = os.path.join(out_dir, f"ontsalaries_n{ds.n}_M{args.M}_U{args.U}.csv")

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
