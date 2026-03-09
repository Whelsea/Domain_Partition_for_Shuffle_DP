#!/usr/bin/env python3
"""
Verify that BaselineClipM.run_streaming() produces statistically
equivalent results to the original run() method.

Runs both methods side-by-side on the same dataset with M=128
(small enough for the original run() to work) and compares:
- m_tau (should be identical)
- mean error (should be close, stochastic)
- total messages (should be close/identical)
"""

import os
import sys
import numpy as np

# Ensure we can import from static/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_clip_M import BaselineClipM
from query_protocols import FE1Protocol


def generate_simple_dataset(n, m_max, U, alpha=6.0, shift=1.36, seed=42):
    """Generate a simple Zipfian dataset."""
    rng = np.random.default_rng(seed)
    support = np.arange(1, m_max + 1, dtype=float)
    weights = (support + shift) ** (-alpha)
    weights /= weights.sum()
    contributions = rng.choice(support, size=n, p=weights).astype(int)

    records = []
    for m_i in contributions:
        user_records = rng.integers(0, U + 1, size=m_i).tolist()
        records.append(user_records)
    return records


def compute_true_freq(records, U):
    """Ground truth frequency vector."""
    freq = np.zeros(U + 1, dtype=float)
    for user in records:
        for v in user:
            freq[v] += 1
    return freq


def run_comparison(n, M, U, epsilon, delta, beta, n_trials=5, seed_base=0):
    print(f"\n{'='*70}")
    print(f"Comparing run() vs run_streaming()")
    print(f"  n={n}, M={M}, U={U}, eps={epsilon}, delta={delta:.2e}, beta={beta}")
    print(f"  n_eff = n*M = {n*M}")
    print(f"{'='*70}")

    datasets = generate_simple_dataset(n, m_max=min(M, 64), U=U, seed=seed_base)
    true_freq = compute_true_freq(datasets, U)
    actual_mmax = max(len(r) for r in datasets)
    print(f"  actual m_max = {actual_mmax}")

    factory = FE1Protocol(
        B=U + 2, real_B=U + 1, dummy_value=U + 1,
        c=1.0, use_mu_search=False, use_analytical=False,
    )

    errors_original = []
    errors_streaming = []
    msgs_original = []
    msgs_streaming = []

    for trial in range(n_trials):
        print(f"\n  Trial {trial+1}/{n_trials}:")

        # --- Original run() ---
        proto_orig = BaselineClipM(n=n, M=M, epsilon=epsilon, delta=delta, beta=beta)
        np.random.seed(1000 + trial)  # Not perfect seeding but ok for quick test
        m_tau_o, qr_o, nmsg_o = proto_orig.run(
            datasets, base_protocol=factory, use_simulate=True,
            status_logger=lambda msg: print(f"    [orig] {msg}") if "finish" in msg or "start" in msg.lower()[:15] else None,
        )
        err_o = np.mean(np.abs(qr_o - true_freq))
        errors_original.append(err_o)
        msgs_original.append(nmsg_o)
        print(f"    Original:  m_tau={m_tau_o}, mae={err_o:.2f}, msgs={nmsg_o}")

        # --- Streaming run_streaming() ---
        proto_stream = BaselineClipM(n=n, M=M, epsilon=epsilon, delta=delta, beta=beta)
        np.random.seed(2000 + trial)
        m_tau_s, qr_s, nmsg_s = proto_stream.run_streaming(
            datasets, base_protocol=factory, use_simulate=True,
            status_logger=lambda msg: print(f"    [stream] {msg}") if "finish" in msg or "start" in msg.lower()[:15] else None,
        )
        err_s = np.mean(np.abs(qr_s - true_freq))
        errors_streaming.append(err_s)
        msgs_streaming.append(nmsg_s)
        print(f"    Streaming: m_tau={m_tau_s}, mae={err_s:.2f}, msgs={nmsg_s}")

        assert m_tau_o == m_tau_s == M, f"m_tau mismatch: {m_tau_o} vs {m_tau_s}"

    print(f"\n  Summary over {n_trials} trials:")
    print(f"    Original  MAE: mean={np.mean(errors_original):.2f} ± {np.std(errors_original):.2f}")
    print(f"    Streaming MAE: mean={np.mean(errors_streaming):.2f} ± {np.std(errors_streaming):.2f}")
    print(f"    Original  msgs: {msgs_original}")
    print(f"    Streaming msgs: {msgs_streaming}")
    print(f"    MAE ratio (streaming/original): {np.mean(errors_streaming)/np.mean(errors_original):.3f}")
    print(f"    => Both methods produce similar error levels: {'YES' if abs(np.mean(errors_streaming)/np.mean(errors_original) - 1) < 0.5 else 'NEEDS INVESTIGATION'}")


if __name__ == "__main__":
    # Small test: M=128, n=1000
    run_comparison(
        n=1000, M=128, U=100,
        epsilon=4.0, delta=1e-6, beta=0.1,
        n_trials=5,
    )

    print("\n\n" + "="*70)
    print("All comparisons done!")
