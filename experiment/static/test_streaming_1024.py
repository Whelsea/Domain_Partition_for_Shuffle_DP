#!/usr/bin/env python3
"""
Test run_streaming with M=1024 — settings that would OOM with run().
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_clip_M import BaselineClipM
from query_protocols import FE1Protocol


def generate_simple_dataset(n, m_max, U, alpha=5.0, shift=1.36, seed=42):
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
    freq = np.zeros(U + 1, dtype=float)
    for user in records:
        for v in user:
            freq[v] += 1
    return freq


def run_test(n, M, U, epsilon, delta, beta):
    print(f"\n{'='*70}")
    print(f"M={M}, n={n}, U={U}, eps={epsilon}")
    print(f"n_eff = {n*M:,}")
    print(f"all_records would be {n*M:,} items (~{n*M*36/1e9:.2f} GB)")
    print(f"{'='*70}")

    datasets = generate_simple_dataset(n, m_max=1024, U=U, alpha=5.0, seed=42)
    true_freq = compute_true_freq(datasets, U)
    actual_mmax = max(len(r) for r in datasets)
    total_records = sum(len(r) for r in datasets)
    print(f"actual m_max = {actual_mmax}")
    print(f"total real records = {total_records:,}")
    print(f"total padded records = {n*M:,}")

    factory = FE1Protocol(
        B=U + 2, real_B=U + 1, dummy_value=U + 1,
        c=1.0, use_mu_search=False, use_analytical=False,
    )

    proto = BaselineClipM(n=n, M=M, epsilon=epsilon, delta=delta, beta=beta)
    t0 = time.time()
    m_tau, qr, nmsg = proto.run_streaming(
        datasets, base_protocol=factory, use_simulate=True,
        status_logger=lambda msg: print(f"  {msg}"),
    )
    elapsed = time.time() - t0

    err = np.mean(np.abs(qr - true_freq))
    linf = np.max(np.abs(qr - true_freq))

    print(f"\nResults:")
    print(f"  m_tau = {m_tau}")
    print(f"  MAE = {err:.2f}")
    print(f"  Linf = {linf:.2f}")
    print(f"  Total messages = {nmsg:,}")
    print(f"  Elapsed = {elapsed:.1f}s")


if __name__ == "__main__":
    # Test 1: M=1024, n=10K (moderate)
    run_test(
        n=10_000, M=1024, U=100,
        epsilon=4.0, delta=1e-8, beta=0.1,
    )

    # Test 2: M=1024, n=100K (large — would OOM with run())
    run_test(
        n=100_000, M=1024, U=100,
        epsilon=4.0, delta=1e-10, beta=0.1,
    )
