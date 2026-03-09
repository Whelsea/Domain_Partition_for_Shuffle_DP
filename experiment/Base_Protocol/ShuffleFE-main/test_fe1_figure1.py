#!/usr/bin/env python3
"""
Reproduce Figure 1 from Luo et al. (2022):
  "Frequency Estimation in the Shuffle Model with Almost a Single Message"

Parameters: n = 10^5, B = 2^24, ε = 1, δ = n^{-2}
Test: c = 1, 2, 3

C++ ground truth (from result/ directory, AOL dataset):
  c=1: b=8685,  q=16777259, mu=97.96, msgs/user=9.508,  90%err=17.47,  Linf=57.54
  c=3: b=65,    q=16777259, mu=97.96, msgs/user=1.064,  90%err=67.45,  Linf=222.0

Usage:
  python3 test_fe1_figure1.py
"""

import math
import sys
import time
import numpy as np

# Import from FE1.py in the same directory
from FE1 import FE1Baseline, _process_chunk_jit


def generate_zipf_data(n: int, B: int, alpha: float = 1.5, seed: int = 42) -> np.ndarray:
    """Generate Zipf-distributed data in [1, B]."""
    rng = np.random.default_rng(seed)
    values = rng.zipf(alpha, size=n)
    values = np.clip(values, 1, B)
    return values.astype(np.int64)


def analyze_single_thread(U: np.ndarray, V: np.ndarray, W: np.ndarray,
                          B: int, q: int, b: int, n: int,
                          rho: float, pcol: float) -> np.ndarray:
    """Single-thread analyzer using numba JIT (avoids multiprocessing overhead)."""
    if U.dtype != np.int64:
        U = U.astype(np.int64, copy=False)
    if V.dtype != np.int64:
        V = V.astype(np.int64, copy=False)
    if W.dtype != np.int64:
        W = W.astype(np.int64, copy=False)

    rounds_base = q // b + 1

    # Warm up numba JIT
    _process_chunk_jit(U[:1], V[:1], W[:1], 0, 1, B, q, b, 1)

    # Process all messages in one call
    freq_counts = _process_chunk_jit(U, V, W, 0, len(U), B, q, b, rounds_base)

    freq = freq_counts.astype(np.float64)
    freq = (freq - n * rho / b - n * pcol) / (1.0 - pcol)
    return freq


def check_error(freqvec: np.ndarray, data: np.ndarray, B: int) -> dict:
    """Compute error metrics matching C++ CheckError."""
    realvec = np.zeros(B + 1, dtype=np.float64)
    for v in data:
        realvec[v] += 1

    errors = np.abs(realvec[1:] - freqvec[1:])  # domain [1, B]
    errors_sorted = np.sort(errors)

    return {
        "50%": errors_sorted[int(round(0.50 * B)) - 1],
        "90%": errors_sorted[int(round(0.90 * B)) - 1],
        "95%": errors_sorted[int(round(0.95 * B)) - 1],
        "99%": errors_sorted[int(round(0.99 * B)) - 1],
        "100%": errors_sorted[-1],  # Linf
    }


def run_figure1(c_values=None):
    if c_values is None:
        c_values = [1, 2, 3]

    n = 100_000
    B = 1 << 24  # 2^24 = 16,777,216
    epsilon = 1.0
    delta = 1.0 / (n * n)  # n^{-2}

    print("=" * 70)
    print("FE1 Figure 1 Reproduction (Luo et al., 2022)")
    print(f"n = {n:,}, B = {B:,} (2^24), ε = {epsilon}, δ = {delta:.2e}")
    print("=" * 70)

    # Generate Zipf data (paper uses AOL + Zipf; we use Zipf for reproducibility)
    data = generate_zipf_data(n, B, alpha=1.5, seed=42)
    unique_vals = len(np.unique(data))
    print(f"Data: Zipf(1.5), range [{data.min()}, {data.max()}], "
          f"{unique_vals:,} unique values")

    results = []

    for c in c_values:
        print(f"\n{'─' * 70}")
        print(f"  c = {c}")
        print(f"{'─' * 70}")

        # Initialize FE1
        fe = FE1Baseline(n=n, B=B, epsilon=epsilon, delta=delta,
                         c=c, beta=0.1, use_mu_search=True, seed=42)

        # Print parameters
        print(f"  Parameters:")
        print(f"    b = {fe.b}")
        print(f"    q = {fe.q}")
        print(f"    mu = {fe.mu:.4f}")
        print(f"    sample_prob (ρ) = {fe.sample_prob:.6f}")
        print(f"    collision_prob = {fe.collision_prob:.8f}")
        print(f"    expected msgs/user = {1 + fe.sample_prob:.4f}")

        # Randomize
        t0 = time.time()
        _ = fe.randomize_all(data.tolist(), shuffle=True)
        U, V, W = fe.to_numpy_messages()
        t1 = time.time()
        actual_msgs_per_user = len(U) / n
        print(f"  Randomization: {t1 - t0:.2f}s")
        print(f"    total messages = {len(U):,}")
        print(f"    actual msgs/user = {actual_msgs_per_user:.4f}")

        # Analyze (single-thread numba — avoids macOS multiprocessing issues)
        print(f"  Analyzing (numba single-thread, B={B:,} elements)...")
        sys.stdout.flush()
        t2 = time.time()
        freqvec = analyze_single_thread(
            U, V, W,
            B=fe.B, q=fe.q, b=fe.b, n=fe.n,
            rho=fe.sample_prob, pcol=fe.collision_prob,
        )
        t3 = time.time()
        print(f"  Analysis: {t3 - t2:.2f}s")

        # Compute errors
        errs = check_error(freqvec, data, B)
        print(f"  Error metrics:")
        print(f"    50% error  = {errs['50%']:.2f}")
        print(f"    90% error  = {errs['90%']:.2f}")
        print(f"    95% error  = {errs['95%']:.2f}")
        print(f"    99% error  = {errs['99%']:.2f}")
        print(f"    100% error = {errs['100%']:.2f}  (Linf)")

        results.append({
            "c": c,
            "b": fe.b,
            "q": fe.q,
            "mu": fe.mu,
            "msgs_user": actual_msgs_per_user,
            "expected_msgs_user": 1 + fe.sample_prob,
            **errs,
        })

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY — Python FE1 (Zipf data)")
    print(f"{'=' * 70}")
    header = f"{'c':>3} | {'msgs/user':>10} | {'90% err':>10} | {'100% err':>10}"
    sep = f"{'-' * 3}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}"
    print(header)
    print(sep)
    for r in results:
        print(f"{r['c']:3d} | {r['msgs_user']:10.4f} | {r['90%']:10.2f} | {r['100%']:10.2f}")

    # C++ ground truth
    print(f"\n{'=' * 70}")
    print("REFERENCE — C++ FE1 (AOL data, from result/ directory)")
    print(f"{'=' * 70}")
    print(header)
    print(sep)
    print(f"  1 |     9.5080 |      17.47 |      57.54")
    print(f"  3 |     1.0637 |      67.45 |     222.00")
    print(f"\nNote: Error magnitudes may differ due to Zipf vs AOL data distribution,")
    print(f"but parameters (b, q, mu, msgs/user) should match C++ exactly.")


if __name__ == "__main__":
    run_figure1()
