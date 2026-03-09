#!/usr/bin/env python3
"""
Batch experiment sweep — generate datasets & run experiments across all
parameter combinations.

Edit SWEEP_CONFIG below, then:

    # Preview what will run (no side-effects)
    python run_sweep.py --dry-run

    # Run everything (skip already-finished settings)
    python run_sweep.py

    # On a server
    nohup python run_sweep.py > sweep.log 2>&1 &

Results are saved as one CSV per setting in OUTPUT_DIR.
A combined summary CSV is written at the end.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ======================================================================
# Paths (relative to this file: experiment/)
# ======================================================================

_THIS_DIR = Path(__file__).resolve().parent
_DATASET_DIR = _THIS_DIR / "dataset"
_SIMULATED_DIR = _DATASET_DIR / "simulated_data"
_REAL_DATA_DIR = _DATASET_DIR / "real_data"
_STATIC_DIR = _THIS_DIR / "static"


# ######################################################################
#
#  EDIT THIS SECTION TO DEFINE YOUR SWEEP
#
# ######################################################################

SWEEP_CONFIG: dict[str, Any] = {
    # ------------------------------------------------------------------
    # Datasets to generate & sweep over
    # ------------------------------------------------------------------
    "datasets": [
        # ---- Simulated data ----
        {
            "type": "simulated",
            "n": [1000, 5000],
            # Public global bound used by protocols (M in the paper)
            "M": [2**20],
            # Actual max contribution used during data generation (private m_max(D))
            "m_max": [1024],
            "U": [100_000, 1_000_000, 10_000_000],
            "contrib_dist": ["zipf", "uniform_fixed"],
            "seed": 42,
        },
        # ---- AOL search log ----
        {
            "type": "aol",
            "n": [5000],
            "M": [32, 64],
            "U": [100_000, 1_000_000],
        },
        # ---- MovieLens 32M ----
        {
            "type": "movielens",
            "n": [5000],
            "M": [64],
            "U": [1_000_000, 10_000_000],
            "mode": "movie_rating",
        },
        # ---- Netflix Prize ----
        {
            "type": "netflix",
            "n": [5000],
            "M": [64],
            "U": [10_000_000, 100_000_000, 1_000_000_000],
            "mode": "movie_rating_date",
        },
    ],

    # ------------------------------------------------------------------
    # Experiment parameters (swept over ALL datasets above)
    # ------------------------------------------------------------------
    "protocols": ["two_round", "one_round", "baseline_clip_M", "baseline_random_tau"],
    "base_protocol": ["FE1"],
    "epsilon": [0.5, 1.0, 2.0, 4.0],

    # Fixed parameters
    "delta": None,          # None → run_experiment.py default (1/n^2)
    "beta": 0.1,
    "times": 50,
    "trim": 0.2,
    "seed": 42,
}

# Output directory (relative to experiment/)
OUTPUT_DIR = _THIS_DIR / "results"


# ######################################################################
#  END OF CONFIG — implementation below
# ######################################################################


# ======================================================================
# Dataset generation helpers
# ======================================================================

@dataclass
class DatasetSpec:
    """A specific dataset file to generate / use."""
    dtype: str          # simulated | aol | movielens | netflix
    n: int
    M: int
    U: int
    # Optional fields
    contrib_dist: str | None = None     # simulated only
    m_max: int | None = None            # simulated only (actual max contribution)
    seed: int | None = None             # simulated only
    mode: str | None = None             # movielens / netflix only

    @property
    def tag(self) -> str:
        """Short human-readable tag for filenames."""
        if self.dtype == "simulated":
            return f"sim_{self.contrib_dist}"
        return self.dtype

    @property
    def csv_path(self) -> Path:
        """Expected output CSV path (auto-named by the generator)."""
        if self.dtype == "simulated":
            effective_m_max = self.M if self.m_max is None else self.m_max
            name = f"n{self.n}_U{self.U}_{self.contrib_dist}_mmax{effective_m_max}"
            if self.seed is not None:
                name += f"_seed{self.seed}"
            name += ".csv"
            return _SIMULATED_DIR / "data" / name
        elif self.dtype == "aol":
            return _REAL_DATA_DIR / "aol" / "data" / f"aol_n{self.n}_M{self.M}_U{self.U}.csv"
        elif self.dtype == "movielens":
            return _REAL_DATA_DIR / "ml-32m" / "data" / f"ml32m_n{self.n}_M{self.M}_U{self.U}.csv"
        elif self.dtype == "netflix":
            return _REAL_DATA_DIR / "Netf" / "data" / f"netflix_n{self.n}_M{self.M}_U{self.U}.csv"
        raise ValueError(f"Unknown dataset type: {self.dtype}")

    def gen_command(self) -> list[str]:
        """Shell command to generate this dataset."""
        if self.dtype == "simulated":
            cmd = [
                sys.executable,
                str(_SIMULATED_DIR / "generate_data.py"),
                "--n", str(self.n),
                "--M", str(self.M),
                "--U", str(self.U),
                "--contrib_dist", self.contrib_dist or "uniform",
            ]
            if self.m_max is not None:
                cmd += ["--m_max", str(self.m_max)]
            if self.seed is not None:
                cmd += ["--seed", str(self.seed)]
            return cmd

        elif self.dtype == "aol":
            return [
                sys.executable,
                str(_REAL_DATA_DIR / "aol" / "process_aol.py"),
                "--n", str(self.n),
                "--M", str(self.M),
                "--U", str(self.U),
            ]
        elif self.dtype == "movielens":
            cmd = [
                sys.executable,
                str(_REAL_DATA_DIR / "ml-32m" / "process_movielens.py"),
                "--n", str(self.n),
                "--M", str(self.M),
                "--U", str(self.U),
            ]
            if self.mode:
                cmd += ["--mode", self.mode]
            return cmd
        elif self.dtype == "netflix":
            cmd = [
                sys.executable,
                str(_REAL_DATA_DIR / "Netf" / "process_netflix.py"),
                "--n", str(self.n),
                "--M", str(self.M),
                "--U", str(self.U),
            ]
            if self.mode:
                cmd += ["--mode", self.mode]
            return cmd
        raise ValueError(f"Unknown dataset type: {self.dtype}")


def expand_dataset_specs(config: dict[str, Any]) -> list[DatasetSpec]:
    """Expand the config into a flat list of DatasetSpec objects."""
    specs: list[DatasetSpec] = []
    for ds_cfg in config["datasets"]:
        dtype = ds_cfg["type"]
        n_list = ds_cfg["n"]
        m_list = ds_cfg["M"]
        u_list = ds_cfg["U"]

        if dtype == "simulated":
            dist_list = ds_cfg.get("contrib_dist", ["uniform"])
            mmax_list = ds_cfg.get("m_max", [None])
            seed = ds_cfg.get("seed")
            for n, M, U, dist, m_max in itertools.product(
                n_list, m_list, u_list, dist_list, mmax_list
            ):
                specs.append(DatasetSpec(
                    dtype="simulated", n=n, M=M, U=U,
                    contrib_dist=dist, m_max=m_max, seed=seed,
                ))
        else:
            mode = ds_cfg.get("mode")
            for n, M, U in itertools.product(n_list, m_list, u_list):
                specs.append(DatasetSpec(
                    dtype=dtype, n=n, M=M, U=U, mode=mode,
                ))
    return specs


# ======================================================================
# Experiment run helpers
# ======================================================================

@dataclass
class ExperimentSpec:
    """A single experiment run = dataset × protocol × base_protocol × epsilon."""
    dataset: DatasetSpec
    protocol: str
    base_protocol: str
    epsilon: float

    # Fixed params (filled from config)
    delta: float | None = None
    beta: float = 0.1
    times: int = 50
    trim: float = 0.2
    seed: int = 42

    @property
    def output_filename(self) -> str:
        """Unique output filename encoding all setting parameters."""
        tag = self.dataset.tag
        eps_str = f"{self.epsilon:.2f}".replace(".", "p")
        mmax_suffix = ""
        if self.dataset.dtype == "simulated" and self.dataset.m_max is not None and self.dataset.m_max != self.dataset.M:
            mmax_suffix = f"_mmax{self.dataset.m_max}"
        return (
            f"{tag}_n{self.dataset.n}_M{self.dataset.M}{mmax_suffix}_U{self.dataset.U}"
            f"_{self.protocol}_{self.base_protocol}_eps{eps_str}.csv"
        )

    def run_command(self, output_dir: Path) -> list[str]:
        """Shell command to run this experiment."""
        out_path = output_dir / self.output_filename
        cmd = [
            sys.executable,
            str(_STATIC_DIR / "run_experiment.py"),
            "--dataset", str(self.dataset.csv_path),
            "--protocols", self.protocol,
            "--base_protocol", self.base_protocol,
            "--epsilon", str(self.epsilon),
            "--beta", str(self.beta),
            "--times", str(self.times),
            "--trim", str(self.trim),
            "--seed", str(self.seed),
            "--output", str(out_path),
        ]
        if self.delta is not None:
            cmd += ["--delta", str(self.delta)]
        return cmd


def expand_experiment_specs(
    config: dict[str, Any],
    dataset_specs: list[DatasetSpec],
) -> list[ExperimentSpec]:
    """Expand all experiment combinations."""
    experiments: list[ExperimentSpec] = []
    protocols = config["protocols"]
    base_protocols = config["base_protocol"]
    epsilons = config["epsilon"]

    # Fixed params
    delta = config.get("delta")
    beta = config.get("beta", 0.1)
    times = config.get("times", 50)
    trim = config.get("trim", 0.2)
    seed = config.get("seed", 42)

    for ds, proto, bp, eps in itertools.product(
        dataset_specs, protocols, base_protocols, epsilons,
    ):
        experiments.append(ExperimentSpec(
            dataset=ds,
            protocol=proto,
            base_protocol=bp,
            epsilon=eps,
            delta=delta,
            beta=beta,
            times=times,
            trim=trim,
            seed=seed,
        ))
    return experiments


# ======================================================================
# Runner
# ======================================================================

def run_cmd(cmd: list[str], label: str, dry_run: bool = False) -> bool:
    """Run a shell command. Returns True on success."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY-RUN] {cmd_str}")
        return True

    print(f"  [RUN] {label}")
    print(f"        {cmd_str}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {label} (exit {result.returncode}, {elapsed:.1f}s)")
        if result.stderr:
            # Print last 20 lines of stderr
            lines = result.stderr.strip().split("\n")
            for line in lines[-20:]:
                print(f"         {line}")
        return False
    else:
        print(f"  [OK]   {label} ({elapsed:.1f}s)")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch sweep runner for shuffle-DP experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if output file already exists.",
    )
    parser.add_argument(
        "--skip-datagen", action="store_true",
        help="Skip dataset generation (assume all datasets exist).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory for result CSVs. Default: {OUTPUT_DIR}",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    config = SWEEP_CONFIG

    # Expand all specs
    dataset_specs = expand_dataset_specs(config)
    experiment_specs = expand_experiment_specs(config, dataset_specs)

    # Deduplicate datasets (same dataset may appear for multiple experiments)
    unique_datasets: dict[str, DatasetSpec] = {}
    for ds in dataset_specs:
        key = str(ds.csv_path)
        unique_datasets[key] = ds
    dataset_list = list(unique_datasets.values())

    print("=" * 70)
    print("SWEEP CONFIGURATION")
    print("=" * 70)
    print(f"  Datasets to generate : {len(dataset_list)}")
    print(f"  Experiment runs      : {len(experiment_specs)}")
    print(f"  Protocols            : {config['protocols']}")
    print(f"  Base protocols       : {config['base_protocol']}")
    print(f"  Epsilon values       : {config['epsilon']}")
    print(f"  Trials per setting   : {config.get('times', 50)}")
    print(f"  Trim fraction        : {config.get('trim', 0.2)}")
    print(f"  Output directory     : {output_dir}")
    print(f"  Dry run              : {args.dry_run}")
    print(f"  Force re-run         : {args.force}")
    print("=" * 70)

    # Phase 1: Dataset generation
    if not args.skip_datagen:
        print(f"\n{'=' * 70}")
        print("PHASE 1: DATASET GENERATION")
        print(f"{'=' * 70}")

        gen_ok = 0
        gen_skip = 0
        gen_fail = 0

        for i, ds in enumerate(dataset_list, 1):
            mmax_str = ""
            if ds.dtype == "simulated" and ds.m_max is not None and ds.m_max != ds.M:
                mmax_str = f" m_max={ds.m_max}"
            label = f"[{i}/{len(dataset_list)}] {ds.tag} n={ds.n} M={ds.M}{mmax_str} U={ds.U}"

            if ds.csv_path.exists() and not args.force:
                print(f"  [SKIP] {label} (exists: {ds.csv_path.name})")
                gen_skip += 1
                continue

            # Ensure output directory exists
            ds.csv_path.parent.mkdir(parents=True, exist_ok=True)

            ok = run_cmd(ds.gen_command(), label, dry_run=args.dry_run)
            if ok:
                gen_ok += 1
            else:
                gen_fail += 1

        print(f"\n  Dataset generation: {gen_ok} generated, {gen_skip} skipped, {gen_fail} failed")

    # Phase 2: Experiment runs
    print(f"\n{'=' * 70}")
    print("PHASE 2: EXPERIMENT RUNS")
    print(f"{'=' * 70}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_ok = 0
    exp_skip = 0
    exp_fail = 0
    exp_missing_ds = 0
    total = len(experiment_specs)
    t_start = time.time()

    for i, exp in enumerate(experiment_specs, 1):
        out_path = output_dir / exp.output_filename
        mmax_str = ""
        if exp.dataset.dtype == "simulated" and exp.dataset.m_max is not None and exp.dataset.m_max != exp.dataset.M:
            mmax_str = f" m_max={exp.dataset.m_max}"
        label = (
            f"[{i}/{total}] {exp.dataset.tag} n={exp.dataset.n} M={exp.dataset.M}{mmax_str} "
            f"U={exp.dataset.U} | {exp.protocol} {exp.base_protocol} eps={exp.epsilon}"
        )

        # Skip if output exists
        if out_path.exists() and not args.force:
            print(f"  [SKIP] {label} (exists)")
            exp_skip += 1
            continue

        # Check dataset exists
        if not exp.dataset.csv_path.exists() and not args.dry_run:
            print(f"  [MISS] {label} (dataset not found: {exp.dataset.csv_path.name})")
            exp_missing_ds += 1
            continue

        ok = run_cmd(exp.run_command(output_dir), label, dry_run=args.dry_run)
        if ok:
            exp_ok += 1
        else:
            exp_fail += 1

        # ETA
        if not args.dry_run and (exp_ok + exp_fail) > 0:
            elapsed = time.time() - t_start
            done = exp_ok + exp_fail + exp_skip + exp_missing_ds
            remaining = total - done
            if done > exp_skip + exp_missing_ds:
                avg_per_run = elapsed / (exp_ok + exp_fail)
                eta_sec = avg_per_run * remaining
                eta_min = eta_sec / 60
                print(f"        ETA: ~{eta_min:.0f} min ({remaining} remaining)")

    total_elapsed = time.time() - t_start

    # Summary
    print(f"\n{'=' * 70}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Experiments: {exp_ok} succeeded, {exp_skip} skipped, "
          f"{exp_fail} failed, {exp_missing_ds} missing dataset")
    if not args.dry_run:
        print(f"  Total time : {total_elapsed / 60:.1f} min")
    print(f"  Results in : {output_dir}/")

    if exp_fail > 0:
        print(f"\n  ⚠ {exp_fail} experiment(s) failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
