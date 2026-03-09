#!/usr/bin/env python3
"""
Process the Netflix Prize dataset into UserLevelDataset CSV format.

Raw data: training_set/ directory containing 17,770 files (mv_NNNNNNN.txt),
one per movie.  Each file has the movie ID on the first line (e.g. ``1:``),
followed by lines of ``CustomerID,Rating,Date``.

    - MovieIDs range from 1 to 17770.
    - CustomerIDs range from 1 to 2649429 (with gaps).  480,189 unique users.
    - Ratings are integers 1–5.
    - Dates are YYYY-MM-DD (Oct 1998 – Dec 2005).

Encoding modes
--------------
movie_rating_date (default):
    Composite value  = (movieId - 1) * 100000 + day_offset * 5 + (rating - 1)
    where day_offset = days since 1998-10-01 (dataset collection start).

    Max day_offset ≈ 2648 (Dec 2005).
    Max raw value ≈ (17770 - 1) * 100000 + 2648 * 5 + 4 = 1,776,913,244
    Each (movie, date, rating) triple is a distinct domain element.
    Ideal for large-domain experiments (U ≥ 10^7).

movie_rating:
    Composite value  = (movieId - 1) * 5 + (rating - 1)
    where rating ∈ {1, 2, 3, 4, 5}, index ∈ {0, 1, 2, 3, 4}.
    Max raw value = (17770 - 1) * 5 + 4 = 88,849.
    Each (movie, rating) pair is a distinct domain element.

In both modes the raw value is mapped to [0, U] via modulo (U + 1).

Usage:
    # Default mode (movie_rating_date), large domain
    python process_netflix.py --n 5000 --M 64 --U 1000000000

    # Movie-rating only mode (smaller domain)
    python process_netflix.py --n 5000 --M 64 --U 100000 --mode movie_rating

    # Custom paths
    python process_netflix.py --raw_data /path/to/training_set --n 2000 --M 128 --U 10000000

    # Output: data/netflix_n5000_M64_U1000000000.csv
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys
from collections import defaultdict
from typing import Any

# Import UserLevelDataset from our canonical module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulated_data"))
from dataset import UserLevelDataset


# ======================================================================
# Date handling
# ======================================================================

# Dataset collection started October 1998
_REFERENCE_DATE = datetime.date(1998, 10, 1)


def _date_to_day_offset(date_str: str) -> int | None:
    """Convert YYYY-MM-DD string to days since 1998-10-01.

    Returns None if the date string is malformed.
    """
    try:
        y, m, d = date_str.split("-")
        dt = datetime.date(int(y), int(m), int(d))
        offset = (dt - _REFERENCE_DATE).days
        return max(offset, 0)  # clamp negative offsets (shouldn't happen)
    except (ValueError, TypeError):
        return None


# ======================================================================
# Rating encoding
# ======================================================================

_VALID_RATINGS = {1, 2, 3, 4, 5}


def encode_movie_rating_date(
    movie_id: int, rating: int, day_offset: int, U: int,
) -> int | None:
    """Encode a (movieId, rating, date) triple into an integer in [0, U].

    Returns None if the rating is invalid.

    Encoding:
        raw = (movieId - 1) * 100000 + day_offset * 5 + (rating - 1)
        value = raw % (U + 1)

    The multiplier 100000 is chosen so that max(day_offset * 5 + 4) ≈ 13244
    fits comfortably within a single movie's block, ensuring unique encoding
    for every (movie, date, rating) triple.
    """
    if rating not in _VALID_RATINGS:
        return None
    raw = (movie_id - 1) * 100000 + day_offset * 5 + (rating - 1)
    return raw % (U + 1)


def encode_movie_rating(movie_id: int, rating: int, U: int) -> int | None:
    """Encode a (movieId, rating) pair into an integer in [0, U].

    Returns None if the rating is invalid.

    Encoding:
        raw = (movieId - 1) * 5 + (rating - 1)
        value = raw % (U + 1)
    """
    if rating not in _VALID_RATINGS:
        return None
    raw = (movie_id - 1) * 5 + (rating - 1)
    return raw % (U + 1)


# ======================================================================
# Raw data parsing
# ======================================================================

def parse_netflix_raw(
    training_dir: str,
    n: int,
    M: int,
    U: int,
    mode: str = "movie_rating_date",
) -> UserLevelDataset:
    """Parse Netflix training_set/ directory and produce a UserLevelDataset.

    Parameters
    ----------
    training_dir : str
        Path to the ``training_set/`` directory containing ``mv_*.txt`` files.
    n : int
        Number of users to include.  After reading all movie files, the
        first *n* users (by earliest customer ID) with at least one valid
        record are selected.
    M : int
        Max records per user.  If a user has more than M valid records,
        only the first M (in file-iteration order) are kept.
    U : int
        Domain upper bound.  Record values will be in {0, …, U}.
    mode : str
        ``"movie_rating_date"`` (default) or ``"movie_rating"``.

    Returns
    -------
    UserLevelDataset
    """
    if mode not in ("movie_rating_date", "movie_rating"):
        raise ValueError(f"Unknown encoding mode: {mode!r}")

    # Discover movie files
    pattern = os.path.join(training_dir, "mv_*.txt")
    movie_files = sorted(glob.glob(pattern))
    if not movie_files:
        raise FileNotFoundError(
            f"No mv_*.txt files found in {training_dir}"
        )

    # Accumulate per-user records across all movie files
    user_records: dict[str, list[int]] = defaultdict(list)
    skipped_ratings = 0
    skipped_dates = 0
    total_lines = 0
    num_movie_files = len(movie_files)

    for file_idx, fpath in enumerate(movie_files, 1):
        if file_idx % 2000 == 0 or file_idx == num_movie_files:
            print(
                f"  Reading movie files: {file_idx}/{num_movie_files}"
                f"  ({len(user_records)} users so far)",
                end="\r",
                flush=True,
            )

        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            # First line: MovieID:
            header = fh.readline().strip()
            movie_id = int(header.rstrip(":"))

            for line in fh:
                total_lines += 1
                parts = line.rstrip("\n").split(",")
                if len(parts) < 3:
                    continue

                customer_id = parts[0].strip()

                # Skip if we already have M records for this user
                if len(user_records[customer_id]) >= M:
                    continue

                try:
                    rating = int(parts[1].strip())
                except ValueError:
                    skipped_ratings += 1
                    continue

                if mode == "movie_rating_date":
                    date_str = parts[2].strip()
                    day_offset = _date_to_day_offset(date_str)
                    if day_offset is None:
                        skipped_dates += 1
                        continue
                    value = encode_movie_rating_date(movie_id, rating, day_offset, U)
                else:
                    value = encode_movie_rating(movie_id, rating, U)

                if value is None:
                    skipped_ratings += 1
                    continue

                user_records[customer_id].append(value)

    print()  # newline after progress

    # Select first n users (sorted by customer ID numerically)
    # Only include users with at least 1 record
    sorted_uids = sorted(
        (uid for uid, recs in user_records.items() if len(recs) > 0),
        key=lambda uid: int(uid),
    )

    selected_uids = sorted_uids[:n]
    records: list[list[int]] = [user_records[uid] for uid in selected_uids]

    metadata: dict[str, Any] = {
        "source": "netflix",
        "raw_dir": os.path.basename(training_dir),
        "encoding": mode,
        "n_requested": n,
        "n_available": len(sorted_uids),
        "M": M,
        "U": U,
        "total_rating_lines": total_lines,
        "num_movie_files": num_movie_files,
    }
    if skipped_ratings > 0:
        metadata["skipped_invalid_ratings"] = skipped_ratings
    if skipped_dates > 0:
        metadata["skipped_invalid_dates"] = skipped_dates

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

DEFAULT_TRAINING_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "training_set",
)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process Netflix Prize training data into UserLevelDataset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_data", type=str, default=DEFAULT_TRAINING_DIR,
                   help="Path to training_set/ directory containing mv_*.txt files.")
    p.add_argument("--n", type=int, required=True, help="Number of users.")
    p.add_argument("--M", type=int, required=True, help="Max records per user.")
    p.add_argument("--U", type=int, required=True,
                   help="Domain upper bound (values in {0,...,U}).")
    p.add_argument("--mode", type=str, default="movie_rating_date",
                   choices=["movie_rating_date", "movie_rating"],
                   help="Encoding mode: 'movie_rating_date' (large domain, default) "
                        "or 'movie_rating' (smaller domain, ignores date).")
    p.add_argument("--output", type=str, default=None,
                   help="Exact output path. Overrides auto-naming.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for auto-named output. Default: real_data/Netf/data/")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.isdir(args.raw_data):
        print(f"Error: training_set directory not found: {args.raw_data}", file=sys.stderr)
        sys.exit(1)

    # Process
    ds = parse_netflix_raw(
        training_dir=args.raw_data,
        n=args.n,
        M=args.M,
        U=args.U,
        mode=args.mode,
    )

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        out_dir = args.output_dir or DEFAULT_DATA_DIR
        out_path = os.path.join(out_dir, f"netflix_n{ds.n}_M{args.M}_U{args.U}.csv")

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
