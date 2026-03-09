#!/usr/bin/env python3
"""
CLI tool — generate a simulated dataset and save it as a .csv file.

Usage examples:

    # Default: n=1000, M=2^20, m_max=1024, U=100, zipf contributions, uniform values
    python generate_data.py

    # Custom parameters
    python generate_data.py --n 2000 --M 1048576 --m_max 512 --U 200 --contrib_dist gaussian --value_dist gaussian --seed 0

    # Output goes to:
    #   experiment/dataset/simulated_data/data/n2000_U200_gaussian_mmax512_seed0.csv

    # Custom output directory
    python generate_data.py --n 500 --M 1024 --m_max 32 --U 50 --output_dir /tmp/datasets

    # Specify exact output path
    python generate_data.py --n 1000 --M 1048576 --m_max 1024 --U 100 --output my_dataset.csv
"""

from __future__ import annotations

import argparse
import os
import sys

from dataset import UserLevelDataset, DatasetGenerator


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_CONTRIB_ZIPF_ALPHA = 6.0
DEFAULT_CONTRIB_ZIPF_SHIFT = 1.36
DEFAULT_VALUE_ZIPF_ALPHA = 1.5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a simulated user-level dataset and save to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Essential parameters
    p.add_argument("--n", type=int, default=1000, help="Number of users.")
    p.add_argument(
        "--M",
        type=int,
        default=2**20,
        help="Global public upper bound on records per user (used by protocols).",
    )
    p.add_argument(
        "--m_max",
        type=int,
        default=1024,
        help="Target actual max contribution in generated data (must satisfy m_max <= M).",
    )
    p.add_argument("--U", type=int, default=100, help="Domain upper bound (values in {0,...,U}).")

    # Distribution selection
    p.add_argument(
        "--contrib_dist", type=str, default="zipf",
        choices=list(DatasetGenerator.CONTRIB_DISTRIBUTIONS),
        help="Contribution count distribution (Stage 1).",
    )
    p.add_argument(
        "--value_dist", type=str, default="uniform",
        choices=list(DatasetGenerator.VALUE_DISTRIBUTIONS),
        help="Record value distribution (Stage 2).",
    )

    # Distribution parameters
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Deprecated shared Zipf exponent. If set, applies to contribution and value Zipf unless overridden.",
    )
    p.add_argument(
        "--contrib_alpha",
        type=float,
        default=None,
        help=f"Contribution Zipf exponent b. Default: {DEFAULT_CONTRIB_ZIPF_ALPHA}.",
    )
    p.add_argument(
        "--zipf_shift",
        type=float,
        default=DEFAULT_CONTRIB_ZIPF_SHIFT,
        help=(
            "Shift a in Pr[X=x] proportional to (x+a)^(-b) for contribution zipf; "
            f"must satisfy a > -1. Default: {DEFAULT_CONTRIB_ZIPF_SHIFT}."
        ),
    )
    p.add_argument(
        "--value_alpha",
        type=float,
        default=None,
        help=f"Value Zipf exponent. Default: {DEFAULT_VALUE_ZIPF_ALPHA}.",
    )
    p.add_argument("--mu", type=float, default=None, help="Mean (for gaussian dist).")
    p.add_argument("--sigma", type=float, default=None, help="Std dev (for gaussian dist).")
    p.add_argument("--p", type=float, default=0.3, help="Success probability (for geometric dist).")
    p.add_argument("--m_each", type=int, default=5, help="Records per user (for uniform_fixed dist).")
    p.add_argument("--m_heavy", type=int, default=None, help="Heavy-user records (one_heavy/mixed).")
    p.add_argument("--m_rest", type=int, default=1, help="Light-user records (one_heavy/mixed).")
    p.add_argument("--n_heavy", type=int, default=None, help="Number of heavy users (mixed).")
    p.add_argument("--value_mu", type=float, default=None, help="Value mean (for gaussian values).")
    p.add_argument("--value_sigma", type=float, default=None, help="Value std dev (for gaussian values).")
    p.add_argument("--constant_value", type=int, default=1, help="Constant record value (for constant values).")

    # Output control
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--output", type=str, default=None,
                   help="Exact output file path. Overrides auto-naming.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for auto-named output. Default: simulated_data/data/")
    p.add_argument("--quiet", action="store_true", help="Suppress summary output.")

    return p


def _build_contrib_params(args: argparse.Namespace) -> dict:
    """Build Stage 1 distribution parameters from CLI args."""
    dist = args.contrib_dist
    params: dict = {}
    contrib_alpha = (
        args.contrib_alpha
        if args.contrib_alpha is not None
        else args.alpha
        if args.alpha is not None
        else DEFAULT_CONTRIB_ZIPF_ALPHA
    )

    if dist == "zipf":
        params["alpha"] = contrib_alpha
        params["shift"] = args.zipf_shift
    elif dist == "gaussian":
        params["mu"] = args.mu if args.mu is not None else args.m_max / 2
        params["sigma"] = args.sigma if args.sigma is not None else args.m_max / 4
    elif dist == "geometric":
        params["p"] = args.p
    elif dist == "uniform_fixed":
        params["m_each"] = args.m_each
    elif dist == "uniform_random":
        params["low"] = 1
        params["high"] = args.m_max
    elif dist == "one_heavy":
        params["m_heavy"] = args.m_heavy if args.m_heavy is not None else args.m_max
        params["m_rest"] = args.m_rest
    elif dist == "mixed":
        params["n_heavy"] = args.n_heavy if args.n_heavy is not None else max(1, args.n // 10)
        params["m_heavy"] = args.m_heavy if args.m_heavy is not None else 10
        params["m_rest"] = args.m_rest

    return params


def _build_value_params(args: argparse.Namespace) -> dict:
    """Build Stage 2 distribution parameters from CLI args."""
    dist = args.value_dist
    params: dict = {}
    value_alpha = (
        args.value_alpha
        if args.value_alpha is not None
        else args.alpha
        if args.alpha is not None
        else DEFAULT_VALUE_ZIPF_ALPHA
    )

    if dist == "zipf":
        params["alpha"] = value_alpha
    elif dist == "gaussian":
        params["mu"] = args.value_mu if args.value_mu is not None else args.U / 2
        params["sigma"] = args.value_sigma if args.value_sigma is not None else args.U / 6
    elif dist == "constant":
        params["value"] = args.constant_value

    return params


def main() -> None:
    args = build_parser().parse_args()
    if args.m_max <= 0:
        raise ValueError(f"m_max must be positive, got {args.m_max}")
    if args.m_max > args.M:
        raise ValueError(f"m_max must satisfy m_max <= M, got m_max={args.m_max}, M={args.M}")

    # Build distribution parameters
    contrib_params = _build_contrib_params(args)
    value_params = _build_value_params(args)

    # Generate dataset
    ds = DatasetGenerator.generate(
        n=args.n,
        M=args.M,
        m_max=args.m_max,
        U=args.U,
        contrib_dist=args.contrib_dist,
        value_dist=args.value_dist,
        contrib_params=contrib_params,
        value_params=value_params,
        seed=args.seed,
    )

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        out_dir = args.output_dir or DEFAULT_DATA_DIR
        filename = UserLevelDataset.canonical_filename(
            n=args.n, M=args.M, U=args.U,
            m_max=args.m_max,
            contrib_dist=args.contrib_dist, seed=args.seed,
        )
        out_path = os.path.join(out_dir, filename)

    # Save
    ds.save_csv(out_path)

    if not args.quiet:
        print(ds.summary())
        print(f"\n  Saved to: {out_path}")
        # Show file size
        size_bytes = os.path.getsize(out_path)
        if size_bytes < 1024:
            print(f"  File size: {size_bytes} B")
        elif size_bytes < 1024 * 1024:
            print(f"  File size: {size_bytes / 1024:.1f} KB")
        else:
            print(f"  File size: {size_bytes / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
