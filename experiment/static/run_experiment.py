#!/usr/bin/env python3
"""
Experiment runner for user-level shuffle-DP protocols.

Supports:
  - Two-round protocol       (Section 4.1, Algorithms 1-4)
  - One-round protocol       (Section 4.2, Algorithms 5-6)
  - Baseline: Clip-to-M      (m_tau = M, global padding)
  - Baseline: Random m_tau   (random threshold selection)

Workflow:
  1. Generate dataset ONCE (from experiment/dataset/simulated_data/):
       python generate_data.py --n 1000 --M 64 --U 100 --contrib_dist zipf --seed 42

  2. Run experiments on that dataset (from experiment/static/):
       python run_experiment.py --dataset ../dataset/simulated_data/data/n1000_U100_zipf_mmax64_seed42.csv

Usage examples:
  # Run all protocols with FE1, 50 trials, trimmed mean (trim 20%)
  python run_experiment.py --dataset data.csv --base_protocol FE1 --times 50

  # Enable Simulator fast paths explicitly
  python run_experiment.py --dataset data.csv --base_protocol FE1 --simulate

  # Sweep epsilon, output to CSV
  python run_experiment.py --dataset data.csv --base_protocol FE1 --epsilon 0.5 1.0 2.0 --output results.csv

  # Only two-round protocol
  python run_experiment.py --dataset data.csv --protocols two_round --base_protocol FE1

  # Counting-only mode (no base_protocol)
  python run_experiment.py --dataset data.csv --protocols two_round --times 20
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from two_round_protocol import TwoRoundProtocol
from one_round_protocol import OneRoundProtocol
from baseline_clip_M import BaselineClipM
from baseline_random_tau import BaselineRandomTau

# Import query protocol factories (FE1, GKMPS)
from query_protocols import FE1Protocol, GKMPSSumProtocol, QUERY_PROTOCOL_REGISTRY

# Import UserLevelDataset for CSV loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataset", "simulated_data"))
from dataset import UserLevelDataset


# ======================================================================
# Protocol Registry
# ======================================================================

PROTOCOL_REGISTRY: dict[str, type] = {
    "two_round": TwoRoundProtocol,
    "one_round": OneRoundProtocol,
    "baseline_clip_M": BaselineClipM,
    "baseline_random_tau": BaselineRandomTau,
}


# ======================================================================
# Trimmed mean utility
# ======================================================================

def trimmed_mean(values: list[float], trim_frac: float = 0.2) -> float:
    """Compute trimmed mean: sort, remove top/bottom ``trim_frac``, average rest.

    With trim_frac=0.2 and 50 values: remove lowest 10 and highest 10,
    average the middle 30.
    """
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    lo = int(n * trim_frac)
    hi = n - lo
    if hi <= lo:
        # Not enough data to trim; fall back to full mean
        return float(np.mean(arr))
    return float(np.mean(arr[lo:hi]))


# ======================================================================
# Core experiment runner
# ======================================================================

def run_protocol(
    protocol_name: str,
    datasets: list[list],
    n: int,
    M: int,
    U: int,
    epsilon: float,
    delta: float,
    beta: float,
    gamma: float,
    times: int,
    trim_frac: float,
    use_simulate: bool = True,
    base_protocol: Any | None = None,
    **protocol_kwargs: Any,
) -> dict[str, Any]:
    """Run a single protocol for ``times`` independent trials.

    Collects per-trial metrics, then computes trimmed mean (removing
    top/bottom ``trim_frac`` fraction of trials).

    Returns dict with m_tau stats, error/relative_error/msg_per_user
    (trimmed means), and raw per-trial data.
    """
    if protocol_name not in PROTOCOL_REGISTRY:
        raise ValueError(
            f"Unknown protocol: {protocol_name}. "
            f"Available: {list(PROTOCOL_REGISTRY.keys())}"
        )

    cls = PROTOCOL_REGISTRY[protocol_name]

    # Build constructor kwargs — only pass extras that the class accepts
    init_params = set(inspect.signature(cls.__init__).parameters.keys())
    init_kwargs: dict[str, Any] = dict(
        n=n, M=M, epsilon=epsilon, delta=delta, beta=beta, gamma=gamma,
    )
    for k, v in protocol_kwargs.items():
        if k in init_params:
            init_kwargs[k] = v

    # Per-trial collectors
    m_tau_values: list[int] = []
    query_results: list[Any] = []
    nmessages_list: list[int] = []
    per_trial_fe1_params: list[dict[str, Any] | None] = []

    run_sig = inspect.signature(cls.run)
    has_status_logger = "status_logger" in run_sig.parameters

    t0 = time.time()
    for trial_idx in range(times):
        run_no = trial_idx + 1
        print(f"[{protocol_name}] run {run_no}/{times}: start", flush=True)

        proto = cls(**init_kwargs)

        run_kwargs: dict[str, Any] = {
            "base_protocol": base_protocol,
            "use_simulate": use_simulate,
        }
        if has_status_logger:
            run_kwargs["status_logger"] = (
                lambda msg, p=protocol_name, r=run_no, t=times:
                print(f"[{p}] run {r}/{t}: {msg}", flush=True)
            )

        # For baseline_clip_M, use streaming path to avoid OOM
        # when n × M is large.
        if protocol_name == "baseline_clip_M" and hasattr(proto, "run_streaming"):
            m_tau, qr, nmsg = proto.run_streaming(datasets, **run_kwargs)
        # For one_round, use run_fast to only generate messages for j*
        # (faithful messages, but skip other levels).
        elif protocol_name == "one_round" and hasattr(proto, "run_fast"):
            m_tau, qr, nmsg = proto.run_fast(datasets, **run_kwargs)
        else:
            m_tau, qr, nmsg = proto.run(datasets, **run_kwargs)
        m_tau_values.append(m_tau)
        query_results.append(qr)
        nmessages_list.append(nmsg)

        # Collect FE1 runtime parameters from the factory if available.
        fe_params: dict[str, Any] | None = None
        if base_protocol is not None and hasattr(base_protocol, "consume_created_params"):
            created = base_protocol.consume_created_params()
            if created:
                target_n_eff = n * m_tau
                match = next(
                    (p for p in reversed(created) if int(p.get("n_eff", -1)) == target_n_eff),
                    None,
                )
                fe_params = match if match is not None else created[-1]
        per_trial_fe1_params.append(fe_params)

        print(
            f"[{protocol_name}] run {run_no}/{times}: done (m_tau={m_tau}, nmsg={nmsg})",
            flush=True,
        )
    elapsed = time.time() - t0

    true_m_max = max(len(d) for d in datasets)
    ge_ratio = sum(1 for v in m_tau_values if v >= true_m_max) / len(m_tau_values)

    # --- Compute per-trial error metrics ---
    per_trial_error: list[float] = []        # absolute error per trial
    per_trial_rel_error: list[float] = []    # relative error per trial (Linf / U)
    per_trial_msg_per_user: list[float] = [] # messages / n per trial
    per_trial_linf_error: list[float] = []   # max |est-true| per trial
    per_trial_p50_error: list[float] = []    # median |est-true| per trial
    per_trial_p90_error: list[float] = []    # 90% quantile |est-true|
    per_trial_p95_error: list[float] = []    # 95% quantile |est-true|
    per_trial_p99_error: list[float] = []    # 99% quantile |est-true|

    has_freq = query_results and isinstance(query_results[0], np.ndarray)
    has_scalar = query_results and query_results[0] is not None and not has_freq

    if has_freq:
        # FE1 path: query_result is a freq_vec (np.ndarray)
        true_freq = np.zeros(U + 1, dtype=np.float64)
        for d in datasets:
            for v in d:
                true_freq[v] += 1

        for i, freq_vec in enumerate(query_results):
            abs_errs = np.abs(freq_vec[: len(true_freq)] - true_freq)
            mean_abs = float(np.mean(abs_errs))
            linf_err = float(np.max(abs_errs))
            mean_rel = linf_err / max(float(U), 1.0)

            per_trial_error.append(mean_abs)
            per_trial_rel_error.append(mean_rel)
            per_trial_msg_per_user.append(nmessages_list[i] / n if n > 0 else 0)
            per_trial_linf_error.append(linf_err)
            per_trial_p50_error.append(float(np.quantile(abs_errs, 0.50)))
            per_trial_p90_error.append(float(np.quantile(abs_errs, 0.90)))
            per_trial_p95_error.append(float(np.quantile(abs_errs, 0.95)))
            per_trial_p99_error.append(float(np.quantile(abs_errs, 0.99)))

    elif has_scalar:
        # GKMPS sum estimation path: query_result is a scalar
        true_sum = sum(v for d in datasets for v in d)
        for i, qr in enumerate(query_results):
            linf_err = abs(float(qr) - true_sum)
            rel_err = linf_err / max(float(U), 1.0)
            abs_err = linf_err
            per_trial_error.append(abs_err)
            per_trial_rel_error.append(rel_err)
            per_trial_msg_per_user.append(nmessages_list[i] / n if n > 0 else 0)
            per_trial_linf_error.append(linf_err)
            per_trial_p50_error.append(abs_err)
            per_trial_p90_error.append(abs_err)
            per_trial_p95_error.append(abs_err)
            per_trial_p99_error.append(abs_err)

    else:
        # Counting-only mode: no query error, only m_tau accuracy
        for i in range(times):
            per_trial_msg_per_user.append(nmessages_list[i] / n if n > 0 else 0)

    # --- Trimmed means ---
    result: dict[str, Any] = {
        "protocol": protocol_name,
        "use_simulate": use_simulate,
        "times": times,
        "trim_frac": trim_frac,
        # m_tau statistics (full, no trimming — these are discrete)
        "m_tau_mean": float(np.mean(m_tau_values)),
        "m_tau_median": float(np.median(m_tau_values)),
        "m_tau_min": int(min(m_tau_values)),
        "m_tau_max": int(max(m_tau_values)),
        "true_m_max": true_m_max,
        "m_tau_ge_mmax_ratio": ge_ratio,
        "elapsed_sec": elapsed,
    }

    if per_trial_error:
        result["error_trimmed_mean"] = trimmed_mean(per_trial_error, trim_frac)
        result["relative_error_trimmed_mean"] = trimmed_mean(per_trial_rel_error, trim_frac)
        result["linf_error_trimmed_mean"] = trimmed_mean(per_trial_linf_error, trim_frac)
        result["error_p50_trimmed_mean"] = trimmed_mean(per_trial_p50_error, trim_frac)
        result["error_p90_trimmed_mean"] = trimmed_mean(per_trial_p90_error, trim_frac)
        result["error_p95_trimmed_mean"] = trimmed_mean(per_trial_p95_error, trim_frac)
        result["error_p99_trimmed_mean"] = trimmed_mean(per_trial_p99_error, trim_frac)

    if per_trial_msg_per_user:
        result["msg_per_user_trimmed_mean"] = trimmed_mean(per_trial_msg_per_user, trim_frac)
        result["msg_per_user_full_mean"] = float(np.mean(per_trial_msg_per_user))

    # Representative FE1 parameters (trial closest to median m_tau).
    if any(p is not None for p in per_trial_fe1_params):
        median_tau = float(np.median(m_tau_values))
        idx = min(
            range(len(m_tau_values)),
            key=lambda i: (
                abs(m_tau_values[i] - median_tau),
                0 if per_trial_fe1_params[i] is not None else 1,
            ),
        )
        rep = per_trial_fe1_params[idx]
        if rep is not None:
            result["fe1_utility_parameter"] = float(rep.get("utility_parameter", np.nan))
            result["fe1_modular_size"] = int(rep.get("modular_size", -1))
            result["fe1_big_prime"] = int(rep.get("big_prime", -1))
            result["fe1_mu"] = float(rep.get("mu", np.nan))
            result["fe1_sample_prob"] = float(rep.get("sample_prob", np.nan))
            result["fe1_collision_probability"] = float(
                rep.get("collision_probability", np.nan)
            )
            result["fe1_n_eff"] = int(rep.get("n_eff", -1))
            result["fe1_epsilon"] = float(rep.get("epsilon", np.nan))
            result["fe1_delta"] = float(rep.get("delta", np.nan))

    # Keep raw data for downstream analysis (not serialised to CSV/JSON)
    result["_raw"] = {
        "m_tau_values": m_tau_values,
        "per_trial_error": per_trial_error,
        "per_trial_rel_error": per_trial_rel_error,
        "per_trial_msg_per_user": per_trial_msg_per_user,
        "per_trial_linf_error": per_trial_linf_error,
        "per_trial_p50_error": per_trial_p50_error,
        "per_trial_p90_error": per_trial_p90_error,
        "per_trial_p95_error": per_trial_p95_error,
        "per_trial_p99_error": per_trial_p99_error,
    }

    return result


# ======================================================================
# Pretty Printing
# ======================================================================

def print_header(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def print_dataset_stats(stats: dict, source: str = "") -> None:
    print_header(f"Dataset Statistics{f'  ({source})' if source else ''}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:>15s}: {v:.2f}")
        else:
            print(f"  {k:>15s}: {v}")


def print_protocol_result(result: dict) -> None:
    times = result["times"]
    trim_pct = int(result["trim_frac"] * 100)

    print_header(f"Protocol: {result['protocol']}")
    print(f"  {'execution mode':>25s}: {result['execution_mode']}")
    print(f"  {'true m_max':>25s}: {result['true_m_max']}")
    print(f"  {'m_tau mean':>25s}: {result['m_tau_mean']:.2f}")
    print(f"  {'m_tau median':>25s}: {result['m_tau_median']:.2f}")
    print(f"  {'m_tau range':>25s}: [{result['m_tau_min']}, {result['m_tau_max']}]")
    print(f"  {'m_tau >= m_max':>25s}: {result['m_tau_ge_mmax_ratio']:.1%}")
    print(f"  {f'time ({times} trials)':>25s}: {result['elapsed_sec']:.2f}s")

    # Error metrics
    if "error_trimmed_mean" in result:
        print(f"  {'--- Error Metrics ---':>25s}  (trimmed mean, trim {trim_pct}%)")
        print(f"  {'error (trimmed)':>25s}: {result['error_trimmed_mean']:.4f}")
        print(f"  {'rel. error (Linf/U)':>25s}: {result['relative_error_trimmed_mean']:.4f}")
        print(f"  {'Linf error':>25s}: {result['linf_error_trimmed_mean']:.4f}")
        print(f"  {'50% error':>25s}: {result['error_p50_trimmed_mean']:.4f}")
        print(f"  {'90% error':>25s}: {result['error_p90_trimmed_mean']:.4f}")
        print(f"  {'95% error':>25s}: {result['error_p95_trimmed_mean']:.4f}")
        print(f"  {'99% error':>25s}: {result['error_p99_trimmed_mean']:.4f}")

    if "msg_per_user_trimmed_mean" in result:
        print(f"  {'msg/user (trimmed)':>25s}: {result['msg_per_user_trimmed_mean']:.2f}")
        print(f"  {'msg/user (full mean)':>25s}: {result['msg_per_user_full_mean']:.2f}")

    if "fe1_modular_size" in result:
        print(f"  {'--- FE1 Parameters ---':>25s}")
        print(f"  {'utility parameter':>25s}: {result['fe1_utility_parameter']:.4f}")
        print(f"  {'modular size':>25s}: {result['fe1_modular_size']}")
        print(f"  {'big prime':>25s}: {result['fe1_big_prime']}")
        print(f"  {'mu':>25s}: {result['fe1_mu']:.4f}")
        print(f"  {'sample probability':>25s}: {result['fe1_sample_prob']:.9f}")
        print(
            f"  {'collision probability':>25s}: "
            f"{result['fe1_collision_probability']:.9f}"
        )


# ======================================================================
# CSV Output
# ======================================================================

CSV_COLUMNS = [
    "protocol", "base_protocol", "execution_mode", "use_simulate",
    "n", "M", "U", "epsilon", "delta", "beta",
    "times", "trim_frac",
    "m_tau_mean", "m_tau_median", "m_tau_min", "m_tau_max",
    "true_m_max", "m_tau_ge_mmax_ratio",
    "error_trimmed_mean",
    "relative_error_trimmed_mean",
    "linf_error_trimmed_mean",
    "error_p50_trimmed_mean", "error_p90_trimmed_mean",
    "error_p95_trimmed_mean", "error_p99_trimmed_mean",
    "msg_per_user_trimmed_mean", "msg_per_user_full_mean",
    "fe1_utility_parameter", "fe1_modular_size", "fe1_big_prime",
    "fe1_mu", "fe1_sample_prob", "fe1_collision_probability",
    "fe1_n_eff", "fe1_epsilon", "fe1_delta",
    "elapsed_sec",
]


def save_results_csv(all_results: list[dict], path: str) -> None:
    """Save results to CSV (one row per protocol × epsilon combination)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            # Build row with safe defaults for missing keys
            row = {col: r.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(row)


def save_results_json(all_results: list[dict], path: str) -> None:
    """Save results to JSON (strips raw data)."""
    serialisable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "_raw"}
        serialisable.append(sr)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)


def _fmt_eps_token(eps: float) -> str:
    """Format epsilon token for filenames (ASCII-safe)."""
    s = f"{eps:g}"
    return s.replace(".", "p")


def build_standard_output_path(
    path: str,
    M: int,
    m_max: int,
    n: int,
    U: int,
    eps_token: str,
    include_timestamp: bool = True,
) -> str:
    """Build standardized output filename.

    Format:
        M{M}_mmax{m_max}_n{n}_U{U}_eps{eps}_time{YYYYMMDD_HHMMSS}

    Notes:
    - Directory is taken from ``path``.
    - Extension is taken from ``path`` if present; defaults to ``.json``.
    """
    out = Path(path)
    ext = out.suffix if out.suffix else ".json"
    ts_part = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else "NA"
    stem = f"M{M}_mmax{m_max}_n{n}_U{U}_eps{eps_token}_time{ts_part}"
    return str(out.with_name(stem + ext))


# ======================================================================
# Main
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="User-level Shuffle-DP Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset source (REQUIRED)
    p.add_argument(
        "--dataset", type=str, required=True,
        help="Path to a CSV dataset file (produced by generate_data.py). "
             "n, M, U are read from the file header.",
    )

    # Protocol selection
    p.add_argument(
        "--protocols", nargs="+",
        default=list(PROTOCOL_REGISTRY.keys()),
        choices=list(PROTOCOL_REGISTRY.keys()),
        help="Protocols to run.",
    )

    # Privacy & protocol parameters
    p.add_argument(
        "--epsilon", type=float, nargs="+", default=[1.0],
        help="Privacy budget epsilon (can sweep multiple values).",
    )
    p.add_argument("--delta", type=float, default=None, help="Privacy delta (default: 1/n^2).")
    p.add_argument("--beta", type=float, default=0.1, help="Failure probability.")
    p.add_argument("--gamma", type=float, default=0.3, help="GKMPS gamma parameter.")

    # Experiment control
    p.add_argument("--times", type=int, default=50, help="Number of independent trials per setting.")
    p.add_argument(
        "--trim", type=float, default=0.2,
        help="Fraction to trim from each end for trimmed mean (0.2 = remove top/bottom 20%%).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--output", type=str, default=None,
        help="Save results to file. Directory and extension are used; filename is standardized.",
    )
    p.add_argument(
        "--no_output_timestamp", action="store_true",
        help="Use 'timeNA' in output filename instead of a real timestamp.",
    )
    p.add_argument(
        "--random_select_times", type=int, default=10,
        help="Number of random m_tau selections per trial (baseline_random_tau only).",
    )
    p.add_argument(
        "--simulate", action="store_true",
        help="Enable Simulator fast paths. Depending on the protocol, this may "
             "use streaming faithful simulation or an analytical simulator.",
    )
    p.add_argument(
        "--no_simulate", action="store_true",
        help="Deprecated compatibility flag. Faithful execution is already the default.",
    )
    p.add_argument(
        "--analytical", action="store_true",
        help="Use the fast binomial FE1 simulator (no real messages). "
             "Faster but still approximate. Implies --simulate.",
    )

    # Base protocol selection (FE1, GKMPS)
    p.add_argument(
        "--base_protocol", type=str, default=None,
        choices=list(QUERY_PROTOCOL_REGISTRY.keys()),
        help="Query protocol for Round 2 (FE1=frequency estimation, GKMPS=sum estimation). "
             "If omitted, uses counting-only mode.",
    )
    p.add_argument(
        "--fe_c", type=float, default=1.0,
        help="FE1 utility parameter c (controls hash bucket size).",
    )
    p.add_argument(
        "--fe_workers", type=int, default=None,
        help="FE1 analyzer multiprocessing workers (default: auto).",
    )
    p.add_argument(
        "--fe_use_mu_search", action="store_true",
        help="Enable FE1 mu_search (disabled by default; uses theoretical mu otherwise).",
    )
    p.add_argument(
        "--quick_fe1", action="store_true",
        help="Convenience preset: FE1 + all four protocols + epsilon=1.0 + times=50 + trim=0.2.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    raw_argv = sys.argv[1:]
    def _has_opt(opt: str) -> bool:
        return any(a == opt or a.startswith(opt + "=") for a in raw_argv)

    if args.quick_fe1:
        if not _has_opt("--protocols"):
            args.protocols = ["two_round", "one_round", "baseline_clip_M", "baseline_random_tau"]
        if not _has_opt("--base_protocol"):
            args.base_protocol = "FE1"
        if not _has_opt("--epsilon"):
            args.epsilon = [1.0]
        if not _has_opt("--times"):
            args.times = 50
        if not _has_opt("--trim"):
            args.trim = 0.2

    np.random.seed(args.seed)

    # Load dataset from CSV
    ds = UserLevelDataset.load_csv(args.dataset)
    n = ds.n
    M = ds.M
    U = ds.U
    datasets = ds.to_protocol_input()
    stats = ds.statistics()

    # Default delta = 1/n^2
    delta = args.delta if args.delta is not None else 1.0 / (n * n)

    # Build base protocol factory if specified
    bp_factory = None
    bp_name = args.base_protocol or "none"
    use_simulate = args.simulate and not args.no_simulate
    # --analytical implies --simulate and forces analytical noise (no real messages)
    if getattr(args, 'analytical', False):
        use_simulate = True

    if args.base_protocol == "FE1":
        # Real domain is [0, U]. Use dummy value U+1 and project it out.
        fe_use_analytical = getattr(args, 'analytical', False)
        bp_factory = FE1Protocol(
            B=U + 2,
            real_B=U + 1,
            dummy_value=U + 1,
            c=args.fe_c,
            use_mu_search=args.fe_use_mu_search,
            use_analytical=fe_use_analytical,
            seed=args.seed,
            workers=args.fe_workers,
        )
        print(
            "  Base protocol: FE1 "
            f"(B={U + 2}, real_B={U + 1}, dummy={U + 1}, c={args.fe_c}, "
            f"use_mu_search={args.fe_use_mu_search}, "
            f"use_analytical={fe_use_analytical})"
        )
    elif args.base_protocol == "GKMPS":
        bp_factory = GKMPSSumProtocol(domain=U, gamma=args.gamma)
        print(f"  Base protocol: GKMPS (domain={U}, gamma={args.gamma})")

    print_dataset_stats(stats, source=os.path.basename(args.dataset))
    print(f"\n  Trials: {args.times}  |  Trim: {args.trim:.0%}  |  Seed: {args.seed}")
    if getattr(args, 'analytical', False):
        exec_mode_str = "analytical (binomial simulator, no real messages)"
    elif use_simulate:
        exec_mode_str = "faithful via Simulator (real messages, batched)"
    else:
        exec_mode_str = "faithful (per-record LocalRandomizer)"
    print(f"  Execution mode: {exec_mode_str}")

    # Run experiments (sweep over epsilon if multiple values)
    all_results: list[dict] = []

    for eps in args.epsilon:
        if len(args.epsilon) > 1:
            print_header(f"epsilon = {eps}")

        for pname in args.protocols:
            result = run_protocol(
                protocol_name=pname,
                datasets=datasets,
                n=n,
                M=M,
                U=U,
                epsilon=eps,
                delta=delta,
                beta=args.beta,
                gamma=args.gamma,
                times=args.times,
                trim_frac=args.trim,
                use_simulate=use_simulate,
                base_protocol=bp_factory,
                random_select_times=args.random_select_times,
            )
            # Attach experiment metadata
            result["base_protocol"] = bp_name
            if getattr(args, 'analytical', False):
                result["execution_mode"] = "analytical"
            elif use_simulate:
                result["execution_mode"] = "faithful_batched"
            else:
                result["execution_mode"] = "faithful"
            result["use_simulate"] = use_simulate
            result["n"] = n
            result["M"] = M
            result["U"] = U
            result["epsilon"] = eps
            result["delta"] = delta
            result["beta"] = args.beta
            result["dataset_source"] = os.path.basename(args.dataset)

            print_protocol_result(result)
            all_results.append(result)

    # Save results
    if args.output:
        eps_token = (
            _fmt_eps_token(args.epsilon[0])
            if len(args.epsilon) == 1
            else "multi"
        )
        output_path = build_standard_output_path(
            args.output,
            M=M,
            m_max=int(stats.get("m_max", 0)),
            n=n,
            U=U,
            eps_token=eps_token,
            include_timestamp=not args.no_output_timestamp,
        )

        if output_path.endswith(".csv"):
            save_results_csv(all_results, output_path)
        else:
            save_results_json(all_results, output_path)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
