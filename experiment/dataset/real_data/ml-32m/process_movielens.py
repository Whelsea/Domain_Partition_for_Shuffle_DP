#!/usr/bin/env python3
"""
Process the MovieLens 32M dataset into UserLevelDataset CSV format.

Raw data: ratings.csv
    Header:  userId,movieId,rating,timestamp
    ~32M rows, ~200K unique users, ~87K unique movies.
    Sorted by userId, then movieId within each user.

Encoding modes
--------------
movie_rating (default):
    Composite value  = (movieId - 1) * 10 + half_star_index
    where half_star_index = int(rating * 2) - 1  ∈ {0, 1, …, 9}
    Max raw value ≈ (292757 - 1) * 10 + 9 = 2,927,569
    Each (movie, rating) pair is a distinct domain element.
    Ideal for frequency estimation on large domains (U ≥ 10^7).

movie_id:
    value = movieId - 1   (0-indexed)
    Ignores the rating — just counts how many users rated each movie.
    Max raw value = 292,756.

In both modes the raw value is mapped to [0, U] via modulo (U + 1).

Usage:
    # Default mode (movie_rating), U = 10^7
    python process_movielens.py --n 5000 --M 64 --U 10000000

    # Movie-ID only mode
    python process_movielens.py --n 5000 --M 64 --U 300000 --mode movie_id

    # Custom paths
    python process_movielens.py --raw_data /path/to/ratings.csv --n 2000 --M 128 --U 10000000

    # Output: data/ml32m_n5000_M64_U10000000.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Import UserLevelDataset from our canonical module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulated_data"))
from dataset import UserLevelDataset


# ======================================================================
# Rating encoding
# ======================================================================

# Valid half-star ratings: 0.5, 1.0, 1.5, ..., 5.0  →  10 levels
_VALID_RATINGS = {r / 2 for r in range(1, 11)}


def encode_movie_rating(movie_id: int, rating: float, U: int) -> int | None:
    """Encode a (movieId, rating) pair into a single integer in [0, U].

    Returns None if the rating is invalid.

    Encoding:
        half_star_index = int(rating * 2) - 1   →  {0, 1, ..., 9}
        raw_value = (movieId - 1) * 10 + half_star_index
        value = raw_value % (U + 1)
    """
    if rating not in _VALID_RATINGS:
        return None
    half_star_index = int(rating * 2) - 1  # 0.5→0, 1.0→1, ..., 5.0→9
    raw_value = (movie_id - 1) * 10 + half_star_index
    return raw_value % (U + 1)


def encode_movie_id(movie_id: int, U: int) -> int:
    """Encode a movieId into [0, U] (ignores rating)."""
    return (movie_id - 1) % (U + 1)


# ======================================================================
# Raw data parsing
# ======================================================================

def parse_movielens_raw(
    raw_path: str,
    n: int,
    M: int,
    U: int,
    mode: str = "movie_rating",
) -> UserLevelDataset:
    """Parse MovieLens ratings.csv and produce a UserLevelDataset.

    Parameters
    ----------
    raw_path : str
        Path to ``ratings.csv``.
    n : int
        Number of users to include.  Reads the file until *n* users
        with at least one valid record are found.
    M : int
        Max records per user.  If a user has more than M valid records,
        only the first M are kept.
    U : int
        Domain upper bound.  Record values will be in {0, …, U}.
    mode : str
        ``"movie_rating"`` (default) or ``"movie_id"``.

    Returns
    -------
    UserLevelDataset
    """
    if mode not in ("movie_rating", "movie_id"):
        raise ValueError(f"Unknown encoding mode: {mode!r}")

    records: list[list[int]] = []
    current_uid: str | None = None
    current_recs: list[int] = []
    skipped_ratings = 0

    with open(raw_path, "r", encoding="utf-8") as fh:
        header = fh.readline()  # skip header: userId,movieId,rating,timestamp

        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue

            user_id = parts[0]
            try:
                movie_id = int(parts[1])
                rating = float(parts[2])
            except (ValueError, IndexError):
                continue

            # User boundary: finalize previous user
            if user_id != current_uid:
                if current_uid is not None and len(current_recs) > 0:
                    records.append(current_recs)
                    if len(records) >= n:
                        break
                current_uid = user_id
                current_recs = []

            # Skip if we already have M records for this user
            if len(current_recs) >= M:
                continue

            # Encode
            if mode == "movie_rating":
                value = encode_movie_rating(movie_id, rating, U)
                if value is None:
                    skipped_ratings += 1
                    continue
            else:
                value = encode_movie_id(movie_id, U)

            current_recs.append(value)

    # Finalize the last user
    if current_uid is not None and len(current_recs) > 0 and len(records) < n:
        records.append(current_recs)

    metadata: dict[str, Any] = {
        "source": "movielens-32m",
        "raw_file": os.path.basename(raw_path),
        "encoding": mode,
        "n_requested": n,
        "M": M,
        "U": U,
    }
    if skipped_ratings > 0:
        metadata["skipped_invalid_ratings"] = skipped_ratings

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
    "ratings.csv",
)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process MovieLens 32M ratings into UserLevelDataset CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_data", type=str, default=DEFAULT_RAW_PATH,
                   help="Path to raw ratings.csv file.")
    p.add_argument("--n", type=int, required=True, help="Number of users.")
    p.add_argument("--M", type=int, required=True, help="Max records per user.")
    p.add_argument("--U", type=int, required=True,
                   help="Domain upper bound (values in {0,...,U}).")
    p.add_argument("--mode", type=str, default="movie_rating",
                   choices=["movie_rating", "movie_id"],
                   help="Encoding mode: 'movie_rating' (composite) or 'movie_id' (ignores rating).")
    p.add_argument("--output", type=str, default=None,
                   help="Exact output path. Overrides auto-naming.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for auto-named output. Default: real_data/ml-32m/data/")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.raw_data):
        print(f"Error: raw data file not found: {args.raw_data}", file=sys.stderr)
        sys.exit(1)

    # Process
    ds = parse_movielens_raw(
        raw_path=args.raw_data,
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
        out_path = os.path.join(out_dir, f"ml32m_n{ds.n}_M{args.M}_U{args.U}.csv")

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
