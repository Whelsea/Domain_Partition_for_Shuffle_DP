"""
Two-round protocol for user-level shuffle-DP.

Implements Algorithms 1-4 from Section 4.1:
  - Algorithm 1: Randomizer for Estimating m_tau (Round 1)
  - Algorithm 2: Analyzer for Estimating m_tau (Round 1)
  - Algorithm 3: Randomizer for Query Q (Round 2)
  - Algorithm 4: Analyzer for Query Q (Round 2)

The counting protocol P_cnt uses GKMPS [Ghazi et al., ICML 2021].
The query protocol P_Q is pluggable via ``baseline`` parameter:
  - baseline.LocalRandomizer(value) -> list of messages
  - baseline.Analyzer(messages)     -> aggregated result
  - baseline.Simulator(values)      -> (nmessages, dp_result)

Concrete wrappers (e.g., TwoRound_GKMPS) instantiate the baseline
with per-record privacy budget after m_tau is determined.
"""

from __future__ import annotations

import contextlib
import io
import math
from typing import Any, Callable
import numpy as np

from GKMPS import GKMPS

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


class TwoRoundProtocol:
    """Two-round protocol for user-level shuffle-DP (Section 4.1).

    Round 1 — Estimate clipping threshold m_tau via domain partitioning.
              Uses GKMPS counting protocol on geometric subdomains I_j.
    Round 2 — Standardize user contributions to m_tau records, then evaluate
              query using a pluggable record-level shuffle-DP protocol P_Q.

    Privacy: (eps, delta)-DP total, split (eps/2, delta/2) per round.
    Failure probability: beta, split beta/2 per round.
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
            n: Number of users.
            M: Global upper bound on records per user (assume power of 2).
            epsilon: Total privacy budget (epsilon).
            delta: Total privacy budget (delta).
            beta: Total failure probability.
            gamma: GKMPS noise allocation parameter (default 0.3).
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

        # Number of subdomains: log_2(M) + 1
        # Convention: log(0) := -1,  2^{-1} := 0.
        self.num_subdomains = math.ceil(math.log2(M)) + 1 if M > 1 else 1

        # Round 1 privacy budget
        self.eps_r1 = epsilon / 2.0
        self.delta_r1 = delta / 2.0
        self.beta_r1 = beta / 2.0

        # Round 2 privacy budget (per-record depends on m_tau)
        self.eps_r2 = epsilon / 2.0
        self.delta_r2 = delta / 2.0
        self.beta_r2 = beta / 2.0

        # Threshold for counting  (Algorithm 2, line 4)
        #   T = (2 / eps) * ln( 2 * (logM + 1) / beta )
        self.threshold = (2.0 / epsilon) * math.log(
            2.0 * self.num_subdomains / beta
        )

        # Shared GKMPS instance for bit counting (domain = 1, values in {0,1})
        self._gkmps_cnt = GKMPS(
            n=self.n,
            domain=1,
            epsilon=self.eps_r1,
            delta=self.delta_r1,
            gamma=self.gamma,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _subdomain_range(self, j: int) -> tuple[int, int]:
        """Return (low, high) for subdomain I_j = [2^{j-1}+1, 2^j].

        Convention:  I_0 = [2^{-1}+1, 2^0] = [1, 1] = {1}.
        """
        if j == 0:
            return (1, 1)
        return (2 ** (j - 1) + 1, 2**j)

    def _in_subdomain(self, m_i: int, j: int) -> int:
        """Indicator  I(m_i in I_j)."""
        low, high = self._subdomain_range(j)
        return 1 if low <= m_i <= high else 0

    def round2_per_record_budget(self, m_tau: int) -> tuple[float, float, int]:
        """Compute per-record privacy budget for Round 2.

        eps_rec   = eps / (2 * m_tau)
        delta_rec = delta / (2 * m_tau)
        n_eff     = n * m_tau
        """
        eps_rec = self.epsilon / (2.0 * m_tau)
        delta_rec = self.delta / (2.0 * m_tau)
        n_eff = self.n * m_tau
        return eps_rec, delta_rec, n_eff

    # ==================================================================
    # Round 1 — Estimate m_tau  (Algorithms 1 & 2)
    # ==================================================================

    def round1_randomizer(self, m_i: int) -> dict[int, list]:
        """Algorithm 1: Randomizer for Estimating m_tau.

        For each subdomain j = 0 .. logM:
            C_i^{(j)} = R_cnt( I(m_i in I_j) ;  eps/2, delta/2, n )
            Send C_i^{(j)} to shuffler S_cnt^{(j)}.

        Args:
            m_i: Number of records for user i.

        Returns:
            Dict  j -> list of messages  C_i^{(j)}.
        """
        result: dict[int, list] = {}
        for j in range(self.num_subdomains):
            indicator = self._in_subdomain(m_i, j)
            result[j] = self._gkmps_cnt.LocalRandomizer(indicator)
        return result

    def round1_analyzer(
        self, per_subdomain_messages: dict[int, list]
    ) -> int:
        """Algorithm 2: Analyzer for Estimating m_tau.

        For each subdomain j:
            Q_cnt^{(j)} = A_cnt( Z^{(j)} ;  eps/2, delta/2, beta/(2(logM+1)), n )
            if Q_cnt^{(j)} > T :  m_tau = 2^j

        Args:
            per_subdomain_messages: j -> list of shuffled messages Z^{(j)}.

        Returns:
            m_tau (>= 1).
        """
        m_tau = 0
        for j in range(self.num_subdomains):
            messages = per_subdomain_messages.get(j, [])
            noisy_count = self._gkmps_cnt.Analyzer(messages)
            if noisy_count > self.threshold:
                m_tau = 2**j
        return max(m_tau, 1)

    def round1_simulate(self, user_contributions: list[int]) -> int:
        """Centralized simulation of Round 1 (efficient).

        Equivalent to running LocalRandomizer per user, shuffling, then
        Analyzer — but uses GKMPS.Simulator for each subdomain instead.

        Args:
            user_contributions: list of m_i (length n).

        Returns:
            m_tau (>= 1).
        """
        m_tau = 0
        for j in range(self.num_subdomains):
            indicators = [
                self._in_subdomain(m_i, j) for m_i in user_contributions
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                _, noisy_count = self._gkmps_cnt.Simulator(indicators)
            if noisy_count > self.threshold:
                m_tau = 2**j
        return max(m_tau, 1)

    def estimate_m_tau(
        self, datasets: list[list], use_simulate: bool = True
    ) -> int:
        """Run Round 1 only: estimate clipping threshold m_tau.

        Args:
            datasets: List of n user datasets.
            use_simulate: True = efficient; False = faithful per-user.

        Returns:
            m_tau (>= 1).
        """
        assert len(datasets) == self.n, (
            f"Expected {self.n} users, got {len(datasets)}"
        )
        user_contributions = [len(D_i) for D_i in datasets]

        if use_simulate:
            return self.round1_simulate(user_contributions)

        per_subdomain: dict[int, list] = {
            j: [] for j in range(self.num_subdomains)
        }
        for m_i in user_contributions:
            user_msgs = self.round1_randomizer(m_i)
            for j, msgs in user_msgs.items():
                per_subdomain[j].extend(msgs)
        # No explicit shuffle: GKMPS Analyzer only depends on the message multiset.
        return self.round1_analyzer(per_subdomain)

    # ==================================================================
    # Round 2 — Query Evaluation  (Algorithms 3 & 4)
    # ==================================================================

    @staticmethod
    def standardize_dataset(
        D_i: list, m_tau: int, dummy: Any = 0
    ) -> list:
        """Clip or pad user dataset to exactly m_tau records.

        If |D_i| > m_tau : truncate to first m_tau records.
        If |D_i| < m_tau : pad with dummy (default 0) to length m_tau.
        """
        if len(D_i) > m_tau:
            return D_i[:m_tau]
        if len(D_i) < m_tau:
            return list(D_i) + [dummy] * (m_tau - len(D_i))
        return list(D_i)

    @staticmethod
    def LocalRandomizer(baseline: Any, record: Any) -> list:
        """Algorithm 3 core: call baseline.LocalRandomizer(record).

        Args:
            baseline: Pre-configured record-level shuffle-DP protocol P_Q.
            record: A single data record (or dummy=0 for padding).

        Returns:
            List of messages from the baseline randomizer.
        """
        return baseline.LocalRandomizer(record)

    @staticmethod
    def Analyzer(baseline: Any, shuffled_messages: list) -> Any:
        """Algorithm 4 core: call baseline.Analyzer(shuffled_messages).

        Args:
            baseline: Pre-configured record-level shuffle-DP protocol P_Q.
            shuffled_messages: All shuffled query messages Z.

        Returns:
            Aggregated query result Q_tilde.
        """
        return baseline.Analyzer(shuffled_messages)

    def round2_randomize_user(
        self, baseline: Any, D_i: list, m_tau: int, dummy: Any = 0
    ) -> list:
        """Algorithm 3: Randomizer for Query Q (one user).

        1. Clip / pad D_i to m_tau records.
        2. For each record k = 1..m_tau:
               Y_{i,k} = baseline.LocalRandomizer(D_i[k])
               Send Y_{i,k} to shuffler S_qry.

        Args:
            baseline: Pre-configured baseline P_Q.
            D_i: User i's raw dataset.
            m_tau: Clipping threshold from Round 1.
            dummy: Padding value for short datasets (default 0).

        Returns:
            Flat list of all query messages from this user.
        """
        D_std = self.standardize_dataset(D_i, m_tau, dummy)
        all_msgs: list = []
        for record in D_std:
            msgs = self.LocalRandomizer(baseline, record)
            all_msgs.extend(msgs)
        return all_msgs

    def round2_full(
        self,
        baseline: Any,
        datasets: list[list],
        m_tau: int,
        dummy: Any = 0,
        status_logger: Callable[[str], None] | None = None,
    ) -> tuple[Any, int]:
        """Full Round 2: per-user randomize → analyze.

        Args:
            baseline: Pre-configured baseline P_Q.
            datasets: List of n user datasets.
            m_tau: Clipping threshold from Round 1.
            dummy: Padding value.

        Returns:
            (dp_result, nmessages).
        """
        all_messages: list = []
        next_pct = 10
        if status_logger is not None:
            status_logger(
                f"Round 2 faithful randomizer: users={len(datasets)}, m_tau={m_tau}"
            )
        for idx, D_i in enumerate(datasets, start=1):
            msgs = self.round2_randomize_user(baseline, D_i, m_tau, dummy)
            all_messages.extend(msgs)
            next_pct = _log_progress(
                status_logger, "Round 2 faithful randomizer", idx, len(datasets), next_pct
            )
        nmessages = len(all_messages)
        if status_logger is not None:
            status_logger(
                "Round 2 faithful shuffle: skip explicit shuffle "
                f"(analyzer uses message multiset only, total_messages={nmessages})"
            )
            status_logger("Round 2 faithful analyzer: start")
        dp_result = self.Analyzer(baseline, all_messages)
        if status_logger is not None:
            status_logger("Round 2 faithful analyzer: finish")
        return dp_result, nmessages

    def round2_simulate(
        self,
        baseline: Any,
        datasets: list[list],
        m_tau: int,
        dummy: Any = 0,
        status_logger: Callable[[str], None] | None = None,
    ) -> tuple[Any, int]:
        """Efficient Round 2 simulation via baseline.Simulator.

        Instead of per-user LocalRandomizer + shuffle, directly
        simulates the aggregate noise using baseline.Simulator.

        Args:
            baseline: Pre-configured baseline P_Q.
            datasets: List of n user datasets.
            m_tau: Clipping threshold from Round 1.
            dummy: Padding value.

        Returns:
            (dp_result, nmessages).
        """
        all_values: list = []
        for D_i in datasets:
            D_std = self.standardize_dataset(D_i, m_tau, dummy)
            all_values.extend(D_std)
        if status_logger is not None:
            status_logger(
                f"Round 2 simulate path: standardized_records={len(all_values)}, m_tau={m_tau}"
            )
        with contextlib.redirect_stdout(io.StringIO()):
            nmessages, dp_result = baseline.Simulator(all_values)
        return dp_result, nmessages

    def evaluate_query(
        self,
        baseline: Any,
        datasets: list[list],
        m_tau: int,
        use_simulate: bool = True,
        dummy: Any = 0,
        status_logger: Callable[[str], None] | None = None,
    ) -> tuple[Any, int]:
        """Run Round 2 only: evaluate query Q with given baseline and m_tau.

        The baseline must be pre-configured with per-record budget:
            eps_rec = eps / (2 * m_tau)
            delta_rec = delta / (2 * m_tau)
            n_eff = n * m_tau

        Args:
            baseline: Pre-configured baseline P_Q.
            datasets: List of n user datasets.
            m_tau: Clipping threshold from Round 1.
            use_simulate: True = efficient simulation; False = faithful.
            dummy: Padding value for standardization.

        Returns:
            (dp_result, nmessages).
        """
        if use_simulate:
            return self.round2_simulate(
                baseline, datasets, m_tau, dummy, status_logger=status_logger
            )
        else:
            return self.round2_full(
                baseline, datasets, m_tau, dummy, status_logger=status_logger
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
        """Execute the full two-round protocol.
        
        Args:
            datasets:     List of n user datasets.
            base_protocol: Pre-configured baseline protocol P_Q,
                          OR a query protocol factory with a ``create(n_eff,
                          eps, delta, beta)`` method (see query_protocols.py).
                          None → Round 2 skipped.
            use_simulate: True = efficient centralised simulation;
                          False = faithful per-user LocalRandomizer + shuffle.
            status_logger: Optional callback for progress/status messages.
        
        Returns:
            (m_tau, dp_result, nmessages).
            dp_result is None when base_protocol is None.
        """
        def _log(msg: str) -> None:
            if status_logger is not None:
                status_logger(msg)

        assert len(datasets) == self.n, (
            f"Expected {self.n} users, got {len(datasets)}"
        )
        
        # --- Round 1: estimate m_tau ---
        _log("start Round 1: estimate_m_tau")
        m_tau = self.estimate_m_tau(datasets, use_simulate)
        _log(f"finish estimate_m_tau, m_tau={m_tau}")
        
        # --- Round 2: query evaluation ---
        if base_protocol is None:
            _log("skip Round 2: base_protocol is None")
            return m_tau, None, 0
        
        # Resolve factory → instance with per-record budget
        eps_rec, delta_rec, n_eff = self.round2_per_record_budget(m_tau)
        _log(
            "start Round 2: resolve base_protocol "
            f"(eps_rec={eps_rec:.6g}, delta_rec={delta_rec:.6g}, n_eff={n_eff})"
        )
        bp = _resolve_bp(
            base_protocol, n_eff, eps_rec, delta_rec, self.beta_r2,
            status_logger=status_logger,
        )
        dummy = getattr(bp, "padding_value", 0)
        _log(f"Round 2 padding dummy={dummy}")
        _log("start Round 2: evaluate_query")
        
        dp_result, nmsg = self.evaluate_query(
            bp, datasets, m_tau, use_simulate, dummy=dummy,
            status_logger=status_logger,
        )
        _log(f"finish Round 2: evaluate_query, nmessages={nmsg}")
        return m_tau, dp_result, nmsg


# ======================================================================
# Concrete Protocol Wrappers
# (To be implemented when baselines are decided)
# ======================================================================

# def TwoRound_GKMPS(datasets, n, M, U, epsilon, delta, beta, gamma=0.3):
#     """Two-round protocol with GKMPS as base query protocol P_Q.
#
#     Usage:
#         protocol = TwoRoundProtocol(n, M, epsilon, delta, beta, gamma)
#         m_tau = protocol.estimate_m_tau(datasets)
#         eps_rec, delta_rec, n_eff = protocol.round2_per_record_budget(m_tau)
#         gkmps = GKMPS(n=n_eff, domain=U, epsilon=eps_rec,
#                        delta=delta_rec, gamma=gamma)
#         dp_result, nmsg = protocol.evaluate_query(gkmps, datasets, m_tau)
#         return m_tau, dp_result
#     """
#     pass


# def TwoRound_BBGN(datasets, n, M, U, epsilon, delta, beta):
#     """Two-round protocol with BBGN as base query protocol P_Q.
#
#     Usage:
#         protocol = TwoRoundProtocol(n, M, epsilon, delta, beta)
#         m_tau = protocol.estimate_m_tau(datasets)
#         eps_rec, delta_rec, n_eff = protocol.round2_per_record_budget(m_tau)
#         bbgn = BBGN(n=n_eff, U=U, epsilon=eps_rec, delta=delta_rec)
#         dp_result, nmsg = protocol.evaluate_query(bbgn, datasets, m_tau)
#         return m_tau, dp_result
#     """
#     pass
