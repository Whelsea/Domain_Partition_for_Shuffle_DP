"""
Baseline 2: Random Threshold Selection.

Instead of estimating m_tau via domain partitioning, randomly select
m_tau from the candidate set {1, 2, 4, 8, ..., M} (powers of 2).

Repeats ``random_select_times`` (K) independent runs, each with a
freshly sampled random m_tau.  Here K is used only to reduce evaluation
variance (average over random tau choices), not as protocol composition.
Each sub-run uses the full baseline per-record budget:
    per-record:   eps / m_tau,  delta / m_tau

Final query result is the mean over K runs.
"""

from __future__ import annotations

import contextlib
import io
import math
import random as pyrandom
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


class BaselineRandomTau:
    """Baseline: Random m_tau selection.

    Randomly pick m_tau from {1, 2, 4, 8, ..., M} for each sub-run.
    Repeat ``random_select_times`` times, then average query results.
    K is an evaluation knob (stability), not privacy composition.
    """

    def __init__(
        self,
        n: int,
        M: int,
        epsilon: float,
        delta: float,
        beta: float,
        gamma: float = 0.3,
        random_select_times: int = 10,
    ) -> None:
        """
        Args:
            n:                   Number of users.
            M:                   Global upper bound on records per user.
            epsilon:             Total privacy budget (epsilon).
            delta:               Total privacy budget (delta).
            beta:                Total failure probability.
            gamma:               GKMPS noise allocation parameter.
            random_select_times: Number of independent random m_tau runs (K).
        """
        assert n > 0, "n must be positive"
        assert M > 0, "M must be positive"
        assert epsilon > 0, "epsilon must be positive"
        assert 0 < delta < 1, "delta must be in (0, 1)"
        assert 0 < beta < 1, "beta must be in (0, 1)"
        assert random_select_times >= 1, "random_select_times must be >= 1"

        self.n = n
        self.M = M
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.random_select_times = random_select_times

        # Candidate m_tau values: {1, 2, 4, 8, ..., M}
        # (powers of 2, matching the geometric subdomains)
        self.candidates: list[int] = (
            [2**j for j in range(int(math.log2(M)) + 1)] if M > 1 else [1]
        )
        # Ensure M itself is included even if not a perfect power of 2
        if M not in self.candidates:
            self.candidates.append(M)

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def per_record_budget(
        self, m_tau: int
    ) -> tuple[float, float, int]:
        """Per-record budget for a given m_tau within one run.

        eps_rec   = eps / m_tau
        delta_rec = delta / m_tau
        n_eff     = n * m_tau

        Returns:
            (eps_rec, delta_rec, n_eff)
        """
        eps_rec = self.epsilon / m_tau
        delta_rec = self.delta / m_tau
        n_eff = self.n * m_tau
        return eps_rec, delta_rec, n_eff

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

    # ------------------------------------------------------------------
    # Query evaluation (shared logic, same as BaselineClipM)
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_query(
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
                    f"Baseline Random-Tau simulate path: standardized_records={len(all_records)}"
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
                    f"Baseline Random-Tau faithful randomizer: records={len(all_records)}"
                )
            for idx, record in enumerate(all_records, start=1):
                all_messages.extend(
                    base_protocol.randomizer(record, eps_rec, delta_rec, n_eff)
                )
                next_pct = _log_progress(
                    status_logger, "Baseline Random-Tau faithful randomizer",
                    idx, len(all_records), next_pct
                )
            if status_logger is not None:
                status_logger(
                    "Baseline Random-Tau faithful shuffle: skip explicit shuffle "
                    "(analyzer uses message multiset only)"
                )
                status_logger("Baseline Random-Tau faithful analyzer: start")
            result = base_protocol.analyzer(
                all_messages, eps_rec, delta_rec, beta, n_eff
            )
            if status_logger is not None:
                status_logger("Baseline Random-Tau faithful analyzer: finish")
            return result, len(all_messages)

        # --- GKMPS-style faithful (budget in constructor) ---
        if hasattr(base_protocol, "LocalRandomizer"):
            all_messages: list = []
            next_pct = 10
            if status_logger is not None:
                status_logger(
                    f"Baseline Random-Tau faithful randomizer: records={len(all_records)}"
                )
            for idx, record in enumerate(all_records, start=1):
                all_messages.extend(base_protocol.LocalRandomizer(record))
                next_pct = _log_progress(
                    status_logger, "Baseline Random-Tau faithful randomizer",
                    idx, len(all_records), next_pct
                )
            if status_logger is not None:
                status_logger(
                    "Baseline Random-Tau faithful shuffle: skip explicit shuffle "
                    "(analyzer uses message multiset only)"
                )
                status_logger("Baseline Random-Tau faithful analyzer: start")
            result = base_protocol.Analyzer(all_messages)
            if status_logger is not None:
                status_logger("Baseline Random-Tau faithful analyzer: finish")
            return result, len(all_messages)

        raise TypeError(
            "base_protocol must expose randomizer (BaseProtocol) "
            "or LocalRandomizer (GKMPS) interface"
        )

    # ------------------------------------------------------------------
    # Single sub-run
    # ------------------------------------------------------------------

    def _run_single(
        self,
        datasets: list[list],
        m_tau: int,
        base_protocol: Any,
        use_simulate: bool,
        status_logger: Callable[[str], None] | None = None,
    ) -> tuple[Any, int]:
        """Run one query evaluation with the given m_tau."""
        eps_rec, delta_rec, n_eff = self.per_record_budget(m_tau)
        beta_per_run = self.beta
        
        # Resolve factory → instance with per-record budget
        bp = _resolve_bp(
            base_protocol, n_eff, eps_rec, delta_rec, beta_per_run,
            status_logger=status_logger,
        )
        dummy = getattr(bp, "padding_value", 0)
        
        # Standardise all datasets to m_tau
        all_records: list = []
        next_pct = 10
        if status_logger is not None:
            status_logger(
                f"Baseline Random-Tau standardize: users={len(datasets)}, m_tau={m_tau}"
            )
        for idx, D_i in enumerate(datasets, start=1):
            D_std = self.standardize_dataset(D_i, m_tau, dummy=dummy)
            all_records.extend(D_std)
            next_pct = _log_progress(
                status_logger, "Baseline Random-Tau standardize",
                idx, len(datasets), next_pct
            )
        
        dp_result, nmsg = self._evaluate_query(
            bp, all_records,
            eps_rec, delta_rec, beta_per_run, n_eff, use_simulate,
            status_logger=status_logger,
        )
        return dp_result, nmsg

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
        """Execute the random threshold selection baseline.

        For each of the K = ``random_select_times`` sub-runs:
          1. Sample m_tau uniformly from {1, 2, 4, 8, ..., M}.
          2. Standardise user datasets to m_tau records.
          3. Run query via base_protocol with per-record budget
             eps / m_tau.

        Returns the median of selected m_tau values and the mean of
        all K query results.

        Args:
            datasets:      List of n user datasets.
            base_protocol: Record-level shuffle-DP protocol P_Q.
                           Accepts BaseProtocol (budget-as-args) or
                           GKMPS (budget-in-constructor) style objects.
            use_simulate:  Use GKMPS.Simulator when available (faster).

        Returns:
            (m_tau, query_result, nmessages).
            m_tau:         Median of the K randomly selected thresholds.
            query_result:  Mean of all K query results (None if no P_Q).
        """
        assert len(datasets) == self.n, (
            f"Expected {self.n} users, got {len(datasets)}"
        )

        K = self.random_select_times
        selected_taus: list[int] = []
        query_results: list[Any] = []
        nmessages_list: list[int] = []

        def _log(msg: str) -> None:
            if status_logger is not None:
                status_logger(msg)

        for run_idx in range(K):
            m_tau = pyrandom.choice(self.candidates)
            selected_taus.append(m_tau)
            _log(f"Baseline Random-Tau subrun {run_idx + 1}/{K}: sampled m_tau={m_tau}")

            if base_protocol is not None:
                qr, nmsg = self._run_single(
                    datasets, m_tau, base_protocol, use_simulate,
                    status_logger=status_logger,
                )
                query_results.append(qr)
                nmessages_list.append(nmsg)

        # Representative m_tau: median of selected values
        m_tau_out = int(np.median(selected_taus))

        # Average query result across K runs
        if query_results:
            if isinstance(query_results[0], np.ndarray):
                # Frequency vectors: element-wise average
                query_result: Any = np.mean(query_results, axis=0)
            else:
                query_result = float(np.mean(query_results))
        else:
            query_result = None
        # K is for variance reduction only; report per-subrun communication.
        total_nmsg = int(round(float(np.mean(nmessages_list)))) if nmessages_list else 0

        return m_tau_out, query_result, total_nmsg
