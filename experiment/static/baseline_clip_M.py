"""
Baseline 1: Global Padding (Clip-to-M).

Naive approach from Section 3 (Attempt 1): set m_tau = M for all users.
All users pad/clip datasets to M records.  No privacy budget is spent
on threshold estimation, so the full (eps, delta) goes to query evaluation.

Per-record budget:  eps / M,  delta / M,  n_eff = n * M.

This baseline demonstrates the utility degradation when using the global
maximum M instead of an instance-specific threshold m_tau ≈ m_max(D).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Any, Callable

import numpy as np

def _resolve_bp(bp, n_eff, eps, delta, beta, status_logger=None):
    """Resolve a query protocol factory into a configured instance."""
    if bp is not None and hasattr(bp, 'create'):
        bp = bp.create(n_eff, eps, delta, beta)
    if bp is not None and hasattr(bp, "set_status_logger"):
        bp.set_status_logger(status_logger)
    return bp


def _log_progress(
    status_logger: Callable[[str], None] | None,
    label: str,
    completed: int,
    total: int,
    next_pct: int,
    step_pct: int = 10,
) -> int:
    if status_logger is None or total <= 0:
        return next_pct
    pct = int((completed * 100) / total)
    while next_pct <= 100 and pct >= next_pct:
        status_logger(f"{label}: {next_pct}% ({completed}/{total})")
        next_pct += step_pct
    return next_pct


class BaselineClipM:
    """Baseline: Global Padding.

    Set m_tau = M unconditionally.  No threshold estimation round.
    Full privacy budget allocated to query evaluation.
    """

    def __init__(
        self,
        n: int,
        M: int,
        epsilon: float,
        delta: float,
        beta: float,
        gamma: float = 0.3,
    ) -> None:
        """
        Args:
            n:       Number of users.
            M:       Global upper bound on records per user.
            epsilon: Total privacy budget (epsilon).
            delta:   Total privacy budget (delta).
            beta:    Total failure probability.
            gamma:   GKMPS noise allocation parameter (default 0.3).
        """
        assert n > 0, "n must be positive"
        assert M > 0, "M must be positive"
        assert epsilon > 0, "epsilon must be positive"
        assert 0 < delta < 1, "delta must be in (0, 1)"
        assert 0 < beta < 1, "beta must be in (0, 1)"

        self.n = n
        self.M = M
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def standardize_dataset(D_i: list, m_tau: int, dummy: Any = 0) -> list:
        """Clip or pad user dataset to exactly *m_tau* records."""
        if len(D_i) > m_tau:
            return D_i[:m_tau]
        if len(D_i) < m_tau:
            return list(D_i) + [dummy] * (m_tau - len(D_i))
        return list(D_i)

    def per_record_budget(self) -> tuple[float, float, int]:
        """Per-record privacy budget.

        No counting round needed, so the full (eps, delta) is available
        for query evaluation.

        Returns:
            (eps_rec, delta_rec, n_eff)
        """
        eps_rec = self.epsilon / self.M
        delta_rec = self.delta / self.M
        n_eff = self.n * self.M
        return eps_rec, delta_rec, n_eff

    # ------------------------------------------------------------------
    # Query evaluation (shared logic)
    # ------------------------------------------------------------------

    def _evaluate_query(
        self,
        base_protocol: Any,
        all_records: list,
        eps_rec: float,
        delta_rec: float,
        beta: float,
        n_eff: int,
        use_simulate: bool,
        status_logger: Callable[[str], None] | None = None,
    ) -> tuple[Any, int]:
        """Run query evaluation on standardised records.

        Supports three base_protocol styles via duck typing:
          1. GKMPS-style Simulator  (use_simulate=True, has ``Simulator``)
          2. BaseProtocol interface  (``randomizer(x, eps, delta, n)``)
          3. GKMPS-style faithful   (``LocalRandomizer(value)``)
        """
        # --- Simulate path (GKMPS.Simulator) ---
        if use_simulate and hasattr(base_protocol, "Simulator"):
            if status_logger is not None:
                status_logger(
                    f"Baseline Clip-M simulate path: standardized_records={len(all_records)}"
                )
            with contextlib.redirect_stdout(io.StringIO()):
                nmsg, dp_result = base_protocol.Simulator(all_records)
            return dp_result, nmsg

        # --- BaseProtocol-style (budget as arguments) ---
        if hasattr(base_protocol, "randomizer"):
            all_messages: list = []
            next_pct = 10
            if status_logger is not None:
                status_logger(
                    f"Baseline Clip-M faithful randomizer: records={len(all_records)}"
                )
            for idx, record in enumerate(all_records, start=1):
                all_messages.extend(
                    base_protocol.randomizer(record, eps_rec, delta_rec, n_eff)
                )
                next_pct = _log_progress(
                    status_logger, "Baseline Clip-M faithful randomizer",
                    idx, len(all_records), next_pct
                )
            if status_logger is not None:
                status_logger(
                    "Baseline Clip-M faithful shuffle: skip explicit shuffle "
                    "(analyzer uses message multiset only)"
                )
                status_logger("Baseline Clip-M faithful analyzer: start")
            result = base_protocol.analyzer(
                all_messages, eps_rec, delta_rec, beta, n_eff
            )
            if status_logger is not None:
                status_logger("Baseline Clip-M faithful analyzer: finish")
            return result, len(all_messages)

        # --- GKMPS-style faithful (budget in constructor) ---
        if hasattr(base_protocol, "LocalRandomizer"):
            all_messages = []
            next_pct = 10
            if status_logger is not None:
                status_logger(
                    f"Baseline Clip-M faithful randomizer: records={len(all_records)}"
                )
            for idx, record in enumerate(all_records, start=1):
                all_messages.extend(base_protocol.LocalRandomizer(record))
                next_pct = _log_progress(
                    status_logger, "Baseline Clip-M faithful randomizer",
                    idx, len(all_records), next_pct
                )
            if status_logger is not None:
                status_logger(
                    "Baseline Clip-M faithful shuffle: skip explicit shuffle "
                    "(analyzer uses message multiset only)"
                )
                status_logger("Baseline Clip-M faithful analyzer: start")
            result = base_protocol.Analyzer(all_messages)
            if status_logger is not None:
                status_logger("Baseline Clip-M faithful analyzer: finish")
            return result, len(all_messages)

        raise TypeError(
            "base_protocol must expose randomizer (BaseProtocol) "
            "or LocalRandomizer (GKMPS) interface"
        )

    # ==================================================================
    # Full Protocol
    # ==================================================================

    def run(
        self,
        datasets: list[list],
        base_protocol: Any | None = None,
        use_simulate: bool = True,
        status_logger: Callable[[str], None] | None = None,
    ) -> tuple[int, Any, int]:
        """Execute the global padding baseline.
        
        Args:
            datasets:      List of n user datasets.
            base_protocol: Record-level shuffle-DP protocol P_Q, or factory
                          with ``create(n_eff, eps, delta, beta)`` method.
            use_simulate:  Use Simulator when available (faster).
        
        Returns:
            (m_tau, query_result, nmessages).  m_tau is always M.
            query_result is None when base_protocol is None.
        """
        assert len(datasets) == self.n, (
            f"Expected {self.n} users, got {len(datasets)}"
        )
        
        m_tau = self.M

        if base_protocol is None:
            return m_tau, None, 0

        def _log(msg: str) -> None:
            if status_logger is not None:
                status_logger(msg)
        
        eps_rec, delta_rec, n_eff = self.per_record_budget()
        
        # Resolve factory → instance with per-record budget
        bp = _resolve_bp(
            base_protocol, n_eff, eps_rec, delta_rec, self.beta,
            status_logger=status_logger,
        )
        dummy = getattr(bp, "padding_value", 0)
        
        # Standardise all user datasets to m_tau = M records
        all_records: list = []
        next_pct = 10
        _log(f"Baseline Clip-M standardize: users={len(datasets)}, m_tau={m_tau}")
        for idx, D_i in enumerate(datasets, start=1):
            D_std = self.standardize_dataset(D_i, m_tau, dummy=dummy)
            all_records.extend(D_std)
            next_pct = _log_progress(
                status_logger, "Baseline Clip-M standardize", idx, len(datasets), next_pct
            )
        
        dp_result, nmsg = self._evaluate_query(
            bp, all_records,
            eps_rec, delta_rec, self.beta, n_eff, use_simulate,
            status_logger=status_logger,
        )
        return m_tau, dp_result, nmsg

    # ==================================================================
    # Streaming Faithful Protocol  (memory-efficient for large n × M)
    # ==================================================================

    def run_streaming(
        self,
        datasets: list[list],
        base_protocol: Any | None = None,
        use_simulate: bool = True,
        status_logger: Callable[[str], None] | None = None,
    ) -> tuple[int, Any, int]:
        """Execute clip-M with streaming record generation.

        Statistically identical to ``run()``, but never materialises
        the full ``all_records`` list (n × M items).  Instead, records
        are generated per-user and fed directly into FE1's streaming
        faithful message generation — bucket counts are accumulated
        incrementally.

        Memory: O(M + batch_messages) instead of O(n × M).

        Only supports FE1-style base protocols (``_FE1Instance``).
        Falls back to ``run()`` for other protocol types.

        Args / Returns: same as ``run()``.
        """
        assert len(datasets) == self.n, (
            f"Expected {self.n} users, got {len(datasets)}"
        )

        m_tau = self.M

        if base_protocol is None:
            return m_tau, None, 0

        def _log(msg: str) -> None:
            if status_logger is not None:
                status_logger(msg)

        eps_rec, delta_rec, n_eff = self.per_record_budget()

        # Resolve factory → instance
        bp = _resolve_bp(
            base_protocol, n_eff, eps_rec, delta_rec, self.beta,
            status_logger=status_logger,
        )
        dummy = getattr(bp, "padding_value", 0)

        # Check if bp is an FE1 instance with the internal _fe attribute
        if not hasattr(bp, "_fe"):
            _log("Baseline Clip-M streaming: not FE1, falling back to run()")
            return self.run(
                datasets, base_protocol, use_simulate,
                status_logger=status_logger,
            )

        # --- Streaming faithful FE1 path ---
        _log(
            f"Baseline Clip-M streaming faithful: "
            f"n={self.n}, M={self.M}, n_eff={n_eff}, "
            f"eps_rec={eps_rec:.6g}, delta_rec={delta_rec:.6g}"
        )

        _FE1_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir, "Base_Protocol", "ShuffleFE-main",
        )
        if _FE1_DIR not in sys.path:
            sys.path.insert(0, _FE1_DIR)
        from FE1 import _process_chunk_jit

        fe = bp._fe
        n    = fe.n          # = n_eff
        B    = fe.B
        q    = fe.q
        b    = fe.b
        rho  = fe.sample_prob
        pcol = fe.collision_prob
        fixed_send     = fe.send_fixed_messages
        remaining_prob = fe.remaining_prob
        rounds_base    = q // b + 1

        rng = np.random.default_rng()
        raw_counts = np.zeros(B + 1, dtype=np.int64)
        total_msgs = 0

        # Warm up numba JIT
        _tiny = np.zeros(1, dtype=np.int64)
        _process_chunk_jit(_tiny, _tiny, _tiny, 0, 0, B, q, b, rounds_base)

        next_pct = 10
        for user_idx, D_i in enumerate(datasets, start=1):
            # Pad / clip to M records (in-memory: only M items)
            D_std = self.standardize_dataset(D_i, m_tau, dummy=dummy)
            batch_vals = np.asarray(D_std, dtype=np.int64) + 1  # 0-indexed → 1-indexed
            bn = len(batch_vals)

            # ---- Real messages (vectorised) ----
            u_real = rng.integers(1, q, size=bn, dtype=np.int64)
            v_real = rng.integers(1, q + 1, size=bn, dtype=np.int64)
            w_real = ((u_real * batch_vals + v_real) % q) % b

            # ---- Dummy messages (vectorised) ----
            sends = np.full(bn, fixed_send, dtype=np.int64)
            if remaining_prob > 0:
                sends += (rng.random(bn) < remaining_prob).astype(np.int64)
            total_dummies = int(sends.sum())

            if total_dummies > 0:
                u_dummy = rng.integers(1, q, size=total_dummies, dtype=np.int64)
                v_dummy = rng.integers(1, q + 1, size=total_dummies, dtype=np.int64)
                w_dummy = rng.integers(0, b, size=total_dummies, dtype=np.int64)

                U = np.concatenate([u_real, u_dummy])
                V = np.concatenate([v_real, v_dummy])
                W = np.concatenate([w_real, w_dummy])
            else:
                U, V, W = u_real, v_real, w_real

            # ---- Count bucket matches (numba JIT) ----
            batch_counts = _process_chunk_jit(
                U, V, W, 0, len(U), B, q, b, rounds_base,
            )
            raw_counts += batch_counts
            total_msgs += len(U)

            del U, V, W, batch_vals, D_std

            next_pct = _log_progress(
                status_logger, "Baseline Clip-M streaming faithful",
                user_idx, self.n, next_pct
            )

        # ---- Debias (same formula as FE1's analyzer) ----
        _log("Baseline Clip-M streaming faithful: debias counts")
        freq_vec = raw_counts.astype(np.float64)
        freq_vec = (freq_vec - n * rho / b - n * pcol) / (1.0 - pcol)
        freq_vec[0] = 0.0

        # ---- Project to real domain ----
        freq_result = bp._project_freq(freq_vec)
        _log(f"Baseline Clip-M streaming faithful: finish, total_messages={total_msgs}")

        return m_tau, freq_result, total_msgs
