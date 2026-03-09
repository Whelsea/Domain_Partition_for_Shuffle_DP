#!/usr/bin/env python3
"""
Process the AOL search log dataset into UserLevelDataset CSV format.

Raw data: user-ct-test-collection-01.txt
    Tab-separated:  AnonID  Query  QueryTime  ItemRank  ClickURL
    ~3.5M rows, ~650K unique users.

Each row is one search event.  We extract a "URL" from:
    1. ClickURL column (if non-empty) — actual clicked URL
    2. Query column (if it looks like a domain: contains dot, no spaces)
    3. Otherwise skip the row

URL-to-integer conversion (follows prior work):
    1. Extract domain name (strip http(s)://, www.)
    2. Take first 3 chars of domain (pad with 'a' if shorter)
    3. Convert each char to 8-bit binary → 24-bit integer
    4. Map to [0, U] via modulo (U + 1)

Usage:
    # From experiment/dataset/real_data/aol/
    python process_aol.py --n 1000 --M 64 --U 100

    # Custom raw data path and output
    python process_aol.py --raw_data /path/to/user-ct-test-collection-01.txt --n 2000 --M 128 --U 1000

    # Output: data/aol_n1000_M64_U100.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Any

# Import UserLevelDataset from our canonical module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulated_data"))
from dataset import UserLevelDataset


# ======================================================================
# URL extraction & conversion
# ======================================================================

# Regex to extract domain from full URL: http(s)://www.domain.com → domain
_URL_RE = re.compile(r'https?://(?:www\.)?([a-zA-Z0-9][\w-]*)', re.IGNORECASE)

# Regex to check if a query string looks like a domain (e.g., "rentdirect.com")
_DOMAIN_LIKE_RE = re.compile(
    r'^(?:www\.)?([a-zA-Z][\w-]*)\.(?:com|org|net|edu|gov|info|biz|co|io)\b',
    re.IGNORECASE,
)


def _extract_domain(click_url: str, query: str) -> str | None:
    """Extract a domain name from a row's ClickURL or Query column.

    Returns the domain string (e.g., "westchestergov") or None if
    no valid domain can be extracted from either column.
    """
    # Priority 1: ClickURL (actual navigation target)
    if click_url:
        m = _URL_RE.search(click_url)
        if m:
            return m.group(1).lower()

    # Priority 2: Query that looks like a domain
    if query and "." in query and " " not in query:
        m = _DOMAIN_LIKE_RE.match(query.strip())
        if m:
            return m.group(1).lower()

    return None


def domain_to_int(domain: str, U: int) -> int:
    """Convert a domain name to an integer in [0, U].

    Method (matches prior work transfer_url.py):
        1. Take first 3 chars of domain (pad with 'a' if shorter)
        2. Each char → 8-bit binary → concatenate → 24-bit integer
        3. Mod (U + 1) to fit in [0, U]
    """
    prefix = domain[:3].lower()
    if len(prefix) < 3:
        prefix = prefix.ljust(3, "a")

    # 3 chars × 8 bits = 24-bit integer
    binary_str = "".join(format(ord(c), "08b") for c in prefix)
    raw_value = int(binary_str, 2)

    return raw_value % (U + 1)


# ======================================================================
# Raw data parsing
# ======================================================================

def parse_aol_raw(
    raw_path: str,
    n: int,
    M: int,
    U: int,
) -> UserLevelDataset:
    """Parse the AOL search log and produce a UserLevelDataset.

    Parameters
    ----------
    raw_path : str
        Path to ``user-ct-test-collection-01.txt``.
    n : int
        Number of users to include.  Reads the file until *n* users
        with at least one valid record are found.
    M : int
        Max records per user.  If a user has more than M valid records,
        only the first M are kept.
    U : int
        Domain upper bound.  Record values will be in {0, …, U}.

    Returns
    -------
    UserLevelDataset
    """
    # Read file line by line.
    # The AOL data is grouped by user (rows for the same AnonID are
    # contiguous), so when the AnonID changes we can finalize the
    # previous user and check if they contributed any records.
    records: list[list[int]] = []
    current_uid: str | None = None
    current_recs: list[int] = []

    with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()  # skip header line

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            anon_id = parts[0].strip()
            query = parts[1].strip() if len(parts) > 1 else ""
            click_url = parts[4].strip() if len(parts) > 4 else ""

            # User boundary: finalize previous user
            if anon_id != current_uid:
                if current_uid is not None and len(current_recs) > 0:
                    records.append(current_recs)
                    if len(records) >= n:
                        break
                current_uid = anon_id
                current_recs = []

            # Skip if we already have M records for this user
            if len(current_recs) >= M:
                continue

            # Extract domain and convert to integer
            domain = _extract_domain(click_url, query)
            if domain is None:
                continue

            value = domain_to_int(domain, U)
            current_recs.append(value)

    # Finalize the last user
    if current_uid is not None and len(current_recs) > 0 and len(records) < n:
        records.append(current_recs)

    metadata: dict[str, Any] = {
        "source": "aol",
        "raw_file": os.path.basename(raw_path),
        "n_requested": n,
        "M": M,
        "U": U,
    }

    return UserLevelDataset(
        records=records,
        n=len(records),
        M=M,
        U=U,
        metadata=metadata,
    )


# ======================================================================
# CLI
# ======================================================================

DEFAULT_RAW_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "user-ct-test-collection-01.txt",
)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process AOL search log into UserLevelDataset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_data", type=str, default=DEFAULT_RAW_PATH,
                   help="Path to raw AOL data file (user-ct-test-collection-01.txt).")
    p.add_argument("--n", type=int, required=True, help="Number of users.")
    p.add_argument("--M", type=int, required=True, help="Max records per user.")
    p.add_argument("--U", type=int, required=True, help="Domain upper bound (values in {0,...,U}).")
    p.add_argument("--output", type=str, default=None,
                   help="Exact output path. Overrides auto-naming.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for auto-named output. Default: real_data/aol/data/")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.raw_data):
        print(f"Error: raw data file not found: {args.raw_data}", file=sys.stderr)
        sys.exit(1)

    # Process
    ds = parse_aol_raw(
        raw_path=args.raw_data,
        n=args.n,
        M=args.M,
        U=args.U,
    )

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        out_dir = args.output_dir or DEFAULT_DATA_DIR
        out_path = os.path.join(out_dir, f"aol_n{ds.n}_M{args.M}_U{args.U}.csv")

    # Save
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
