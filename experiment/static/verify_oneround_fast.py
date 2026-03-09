#!/usr/bin/env python3
"""Verify that OneRoundProtocol.run_fast() is statistically equivalent
to the original _run_simulate_path() by comparing m_tau distributions
and error metrics over many trials."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'simulated_data'))

import contextlib, io
import numpy as np
from dataset import UserLevelDataset
from one_round_protocol import OneRoundProtocol
from query_protocols import FE1Protocol

DATASET = "../dataset/simulated_data/data/n1000_U100_zipf_mmax1024_seed42.csv"
EPS = 4.0
TRIALS = 3  # original _run_simulate_path is slow (all L levels)

ds = UserLevelDataset.load_csv(DATASET)
datasets = ds.to_protocol_input()
n = ds.n
M = ds.M
U = ds.U
delta = 1 / (n ** 2)
beta = 0.01

print(f"n={n}, M={M}, U={U}, m_max={ds.m_max}, delta={delta:.2e}")

# True histogram
domain_size = U + 1
true_hist = np.zeros(domain_size)
for D_i in datasets:
    for v in D_i:
        true_hist[v] += 1

def make_bp():
    return FE1Protocol(B=U+2, real_B=U+1, dummy_value=U+1, c=1.0, use_mu_search=False)

proto = OneRoundProtocol(n=n, M=M, epsilon=EPS, delta=delta, beta=beta)

fast_mtaus, orig_mtaus = [], []
fast_errors, orig_errors = [], []
fast_nmsg, orig_nmsg = [], []

for t in range(TRIALS):
    # --- run_fast ---
    bp = make_bp()
    with contextlib.redirect_stdout(io.StringIO()):
        m1, qr1, nm1 = proto.run_fast(datasets, base_protocol=bp)
    fast_mtaus.append(m1)
    fast_nmsg.append(nm1)
    if qr1 is not None:
        err1 = np.mean(np.abs(np.array(qr1[:domain_size]) - true_hist))
        fast_errors.append(err1)

    # --- original run (simulator path) ---
    bp2 = make_bp()
    with contextlib.redirect_stdout(io.StringIO()):
        m2, qr2, nm2 = proto.run(datasets, base_protocol=bp2, use_simulate=True)
    orig_mtaus.append(m2)
    orig_nmsg.append(nm2)
    if qr2 is not None:
        err2 = np.mean(np.abs(np.array(qr2[:domain_size]) - true_hist))
        orig_errors.append(err2)

    print(f"  trial {t+1:2d}: fast m_tau={m1:4d} err={fast_errors[-1]:10.1f} nmsg={nm1:>10d}"
          f"  |  orig m_tau={m2:4d} err={orig_errors[-1]:10.1f} nmsg={nm2:>10d}")

print("\n=== Summary ===")
print(f"  run_fast:  m_tau mean={np.mean(fast_mtaus):.1f}  err mean={np.mean(fast_errors):.1f}  nmsg mean={np.mean(fast_nmsg):.0f}")
print(f"  run(sim):  m_tau mean={np.mean(orig_mtaus):.1f}  err mean={np.mean(orig_errors):.1f}  nmsg mean={np.mean(orig_nmsg):.0f}")
print(f"  m_tau match rate: {sum(a==b for a,b in zip(fast_mtaus,orig_mtaus))/TRIALS*100:.0f}% (expect ~100% if counting is stable)")
print(f"  err ratio (fast/orig): {np.mean(fast_errors)/np.mean(orig_errors):.4f}")
print(f"  nmsg ratio (fast/orig): {np.mean(fast_nmsg)/np.mean(orig_nmsg):.4f} (fast should be ~1/L of orig)")
