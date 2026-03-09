"""
One-round (non-interactive) protocol for user-level shuffle-DP.

Implements Algorithms 5-6 from Section 4.2:
  - Algorithm 5: One-round Randomizer
  - Algorithm 6: One-round Analyzer

Key difference from two-round:
  Users prepare counting messages AND query responses for ALL log M + 1
  subdomains simultaneously.  The analyzer determines m_tau from counting,
  then selectively aggregates query responses from subdomain log(m_tau).

Privacy budget split:
  - eps/2, delta/2  for counting  (parallel composition across disjoint subdomains)
  - eps/2, delta/2  for query, further divided by (logM + 1) subdomains
    and 2^j records within each subdomain  (basic composition)
"""

from __future__ import annotations

import contextlib
import io
import math
import random as pyrandom
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


def _resolve_dummy_value(bp, default=0):
    """Resolve padding dummy from base protocol (factory or instance)."""
    return getattr(bp, "padding_value", default)


class OneRoundProtocol:
    """One-round protocol for user-level shuffle-DP (Section 4.2).

    Users compute all messages in a single shot; the analyzer then
    identifies m_tau and selects the corresponding query responses.
    Compared with the two-round protocol the only overhead is an extra
    log M factor in the query-evaluation budget.
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
        self.num_subdomains = math.ceil(math.log2(M)) + 1 if M > 1 else 1
        # Shorthand
        self.L = self.num_subdomains  # L = logM + 1

        # Counting budget:  eps/2, delta/2  (same as two-round)
        self.eps_cnt = epsilon / 2.0
        self.delta_cnt = delta / 2.0

        # beta' = beta / (2 * (logM + 1))
        self.beta_prime = beta / (2.0 * self.L)

        # Threshold:  (2/eps) * ln(1/beta')  =  (2/eps) * ln(2*(logM+1)/beta)
        self.threshold = (2.0 / epsilon) * math.log(1.0 / self.beta_prime)

        # GKMPS instance for bit counting
        self._gkmps_cnt = GKMPS(
            n=self.n,
            domain=1,
            epsilon=self.eps_cnt,
            delta=self.delta_cnt,
            gamma=self.gamma,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _subdomain_range(self, j: int) -> tuple[int, int]:
        """I_j = [2^{j-1}+1, 2^j],  I_0 = {1}."""
        if j == 0:
            return (1, 1)
        return (2 ** (j - 1) + 1, 2**j)

    def _in_subdomain(self, m_i: int, j: int) -> int:
        low, high = self._subdomain_range(j)
        return 1 if low <= m_i <= high else 0

    def query_budget(self, j: int) -> tuple[float, float, int]:
        """Per-record query budget for subdomain j.

        eps_q   = eps  / (2 * 2^j * (logM + 1))
        delta_q = delta / (2 * 2^j * (logM + 1))
        n_eff   = n * 2^j

        Returns:
            (eps_q, delta_q, n_eff)
        """
        size_j = 2**j
        eps_q = self.epsilon / (2.0 * size_j * self.L)
        delta_q = self.delta / (2.0 * size_j * self.L)
        n_eff = self.n * size_j
        return eps_q, delta_q, n_eff

    @staticmethod
    def _standardize_user_dataset(D_i: list, target_size: int, dummy: Any) -> list:
        """Clip/pad one user's records to ``target_size``."""
        if len(D_i) < target_size:
            return list(D_i) + [dummy] * (target_size - len(D_i))
        return D_i[:target_size]

    def _collect_standardized_values(
        self,
        datasets: list[list],
        target_size: int,
        dummy: Any,
    ) -> list:
        """Flatten all users' standardised records for one subdomain size."""
        all_values: list = []
        for D_i in datasets:
            D_tilde = self._standardize_user_dataset(D_i, target_size, dummy)
            all_values.extend(D_tilde)
        return all_values

    @staticmethod
    def _estimate_query_messages(
        bp: Any,
        all_values: list | None,
        eps_q: float,
        delta_q: float,
        n_eff: int,
    ) -> float:
        """Estimate expected message count for one query instance.

        Priority:
          1) FE1 wrapper: exact expectation via sample_prob.
          2) GKMPS: EstimateMessageNumber if available.
          3) Generic fallback: needs concrete values.
        """
        if hasattr(bp, "_fe") and hasattr(bp._fe, "sample_prob"):
            return float(n_eff * (1.0 + float(bp._fe.sample_prob)))

        if hasattr(bp, "EstimateMessageNumber"):
            try:
                return float(n_eff * bp.EstimateMessageNumber())
            except TypeError:
                return float(n_eff * bp.EstimateMessageNumber())

        if all_values is None:
            return float("nan")

        if hasattr(bp, "Simulator"):
            with contextlib.redirect_stdout(io.StringIO()):
                nmsg, _ = bp.Simulator(all_values)
            return float(nmsg)

        if hasattr(bp, "randomizer"):
            total = 0
            for record in all_values:
                total += len(bp.randomizer(record, eps_q, delta_q, n_eff))
            return float(total)

        if hasattr(bp, "LocalRandomizer"):
            total = 0
            for record in all_values:
                total += len(bp.LocalRandomizer(record))
            return float(total)

        raise TypeError(
            "base_protocol must expose Simulator / EstimateMessageNumber / "
            "randomizer / LocalRandomizer interface"
        )

    def estimate_one_round_nmessages(
        self,
        datasets: list[list],
        base_protocol: Any,
    ) -> float:
        """Estimate expected query-message count for one-round randomization.

        In one-round, users randomize for *all* candidate thresholds 2^j.
        This quantity should not depend on the selected m_tau.
        """
        total = 0.0
        for j in range(self.num_subdomains):
            eps_q, delta_q, n_eff = self.query_budget(j)
            target_size = 2**j
            bp_j = _resolve_bp(base_protocol, n_eff, eps_q, delta_q, self.beta_prime)
            dummy_j = _resolve_dummy_value(bp_j, default=0)
            est = self._estimate_query_messages(
                bp_j, None, eps_q, delta_q, n_eff
            )
            if math.isnan(est):
                all_values_j = self._collect_standardized_values(
                    datasets, target_size, dummy_j
                )
                est = self._estimate_query_messages(
                    bp_j, all_values_j, eps_q, delta_q, n_eff
                )
            total += est
        return total

    # ==================================================================
    # Algorithm 5 — One-round Randomizer
    # ==================================================================

    def randomizer(
        self,
        D_i: list,
        base_protocol: Any | None = None,
        dummy: Any = 0,
    ) -> tuple[dict[int, list], dict[int, list]]:
        """Algorithm 5: One-round Randomizer (per user).

        For each subdomain j = 0 .. logM:
          1. Counting:
               C_i^{(j)} = R_cnt( I(m_i in I_j) ;  eps/2, delta/2, n )
          2. Query  (if base_protocol provided):
               Pad/clip D_i to size 2^j,
               for k = 1..2^j:
                   Y_{i,k}^{(j)} = R_Q( D_tilde[k] ;  eps_q, delta_q, n_eff )

        Args:
            D_i:           User i's dataset.
            base_protocol: Record-level protocol P_Q (None → skip query).
            dummy:         Padding value for short datasets.

        Returns:
            (cnt_msgs, qry_msgs):
                cnt_msgs  —  j -> list of counting messages C_i^{(j)}
                qry_msgs  —  j -> list of query messages   (empty dict if no P_Q)
        """
        m_i = len(D_i)
        cnt_msgs: dict[int, list] = {}
        qry_msgs: dict[int, list] = {}

        for j in range(self.num_subdomains):
            # --- counting ---
            indicator = self._in_subdomain(m_i, j)
            cnt_msgs[j] = self._gkmps_cnt.LocalRandomizer(indicator)

            # --- query ---
            if base_protocol is not None:
                target_size = 2**j
                eps_q, delta_q, n_eff = self.query_budget(j)

                # Clip or pad to target_size records
                if m_i < target_size:
                    D_tilde = list(D_i) + [dummy] * (target_size - m_i)
                else:
                    D_tilde = D_i[:target_size]

                msgs: list = []
                for record in D_tilde:
                    msgs.extend(
                        base_protocol.randomizer(record, eps_q, delta_q, n_eff)
                    )
                qry_msgs[j] = msgs

        return cnt_msgs, qry_msgs

    # ==================================================================
    # Algorithm 6 — One-round Analyzer
    # ==================================================================

    def analyzer(
        self,
        all_cnt_msgs: dict[int, list],
        all_qry_msgs: dict[int, list],
        base_protocol: Any | None = None,
    ) -> tuple[int, Any]:
        """Algorithm 6: One-round Analyzer.

        Step 1 — Determine m_tau  (same counting logic as two-round):
            For j = 0..logM:
                Q_cnt^{(j)} = A_cnt( Z_cnt^{(j)} )
                if Q_cnt^{(j)} > T :  m_tau = 2^j

        Step 2 — Selective aggregation:
            Q_tilde = A_Q( Z_Q^{(log m_tau)} ;
                           eps/(2*(logM+1)*m_tau),
                           delta/(2*(logM+1)*m_tau),
                           beta', n*m_tau )

        Args:
            all_cnt_msgs: j -> shuffled counting messages Z_cnt^{(j)}.
            all_qry_msgs: j -> shuffled query messages Z_Q^{(j)}.
            base_protocol: Record-level protocol P_Q.

        Returns:
            (m_tau, query_result).  query_result is None if P_Q absent.
        """
        # --- Step 1: determine m_tau ---
        m_tau = 0
        for j in range(self.num_subdomains):
            msgs = all_cnt_msgs.get(j, [])
            noisy_count = self._gkmps_cnt.Analyzer(msgs)
            if noisy_count > self.threshold:
                m_tau = 2**j
        m_tau = max(m_tau, 1)

        # --- Step 2: selective aggregation ---
        if base_protocol is None or not all_qry_msgs:
            return m_tau, None

        j_star = int(math.log2(m_tau)) if m_tau > 1 else 0
        eps_q = self.epsilon / (2.0 * self.L * m_tau)
        delta_q = self.delta / (2.0 * self.L * m_tau)
        n_eff = self.n * m_tau

        qry_msgs = all_qry_msgs.get(j_star, [])
        query_result = base_protocol.analyzer(
            qry_msgs, eps_q, delta_q, self.beta_prime, n_eff
        )

        return m_tau, query_result

    # ==================================================================
    # Efficient simulation
    # ==================================================================

    def simulate_counting(self, user_contributions: list[int]) -> int:
        """Centralized simulation of the counting part (efficient).

        Same output distribution as per-user randomizer + shuffle + analyzer.
        """
        m_tau = 0
        for j in range(self.num_subdomains):
            indicators = [
                self._in_subdomain(m_i, j) for m_i in user_contributions
            ]
            gkmps = GKMPS(
                n=self.n,
                domain=1,
                epsilon=self.eps_cnt,
                delta=self.delta_cnt,
                gamma=self.gamma,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                _, noisy_count = gkmps.Simulator(indicators)
            if noisy_count > self.threshold:
                m_tau = 2**j
        return max(m_tau, 1)

    # ==================================================================
    # Fast Simulator path (analytical / Simulator-based)
    # ==================================================================

    def _run_simulate_path(
        self,
        datasets: list[list],
        query_cfg: dict[int, tuple],
    ) -> tuple[int, Any, int]:
        """Execute one-round protocol using Simulator for query evaluation.

        Instead of calling LocalRandomizer per-record and Analyzer on the
        shuffled messages, this path:
          1. Estimates m_tau via counting (same as faithful path).
          2. For each level j, collects all standardised records and calls
             ``bp_j.Simulator(records_j)`` once.
          3. Returns the result for j* = log2(m_tau).

        This is equivalent to the faithful path when the Simulator produces
        the same distribution (e.g. analytical FE1 simulation).
        """
        # Step 1: estimate m_tau via counting simulation
        user_contributions = [len(D_i) for D_i in datasets]
        m_tau = self.simulate_counting(user_contributions)

        # Step 2: evaluate queries per level using Simulator
        qry_results: dict[int, Any] = {}
        total_nmsg = 0

        for j in range(self.num_subdomains):
            bp_j, eps_q, delta_q, n_eff, dummy_j = query_cfg[j]
            target_size = 2 ** j

            # Collect all records for level j (standardised to target_size)
            records_j: list = []
            for D_i in datasets:
                m_i = len(D_i)
                if m_i < target_size:
                    D_tilde = list(D_i) + [dummy_j] * (target_size - m_i)
                else:
                    D_tilde = D_i[:target_size]
                records_j.extend(D_tilde)

            assert len(records_j) == n_eff, (
                f"Level {j}: expected {n_eff} records, got {len(records_j)}"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                nmsg_j, freq_j = bp_j.Simulator(records_j)
            qry_results[j] = freq_j
            total_nmsg += nmsg_j

        # Step 3: select j* and return
        j_star = int(math.log2(m_tau)) if m_tau > 1 else 0
        query_result = qry_results.get(j_star)
        return m_tau, query_result, total_nmsg

    # ==================================================================
    # Fast faithful path — only generate messages for j*
    # ==================================================================

    def run_fast(
        self,
        datasets: list[list],
        base_protocol: Any | None = None,
        use_simulate: bool = True,
        status_logger: Callable[[str], None] | None = None,
    ) -> tuple[int, Any, int]:
        """Execute one-round protocol, generating messages only for j*.

        Statistically equivalent to the full faithful path because
        each level's query channel is independent.  Privacy budget is
        still divided by L = log M + 1 (basic composition over all
        levels), so the per-record budget is identical.

        Steps:
          1. Counting simulation (GKMPS.Simulator) → determine m_tau.
          2. Resolve j* = log₂(m_tau).
          3. Collect standardised records for level j* only.
          4. Run FE1 streaming faithful (real message generation) on j*.

        Falls back to ``run()`` if base protocol is not FE1-compatible.

        Args / Returns: same as ``run()``.
        """
        assert len(datasets) == self.n, (
            f"Expected {self.n} users, got {len(datasets)}"
        )

        def _log(msg: str) -> None:
            if status_logger is not None:
                status_logger(msg)

        if base_protocol is None:
            user_contributions = [len(D_i) for D_i in datasets]
            m_tau = self.simulate_counting(user_contributions)
            return m_tau, None, 0

        # Step 1: determine m_tau via counting simulation
        _log("One-round fast: step 1 — counting simulation")
        user_contributions = [len(D_i) for D_i in datasets]
        m_tau = self.simulate_counting(user_contributions)
        _log(f"One-round fast: m_tau={m_tau}")

        # Step 2: resolve j*
        j_star = int(math.log2(m_tau)) if m_tau > 1 else 0
        eps_q, delta_q, n_eff = self.query_budget(j_star)
        _log(
            f"One-round fast: step 2 — resolve j*={j_star}, "
            f"eps_q={eps_q:.6g}, delta_q={delta_q:.6g}, n_eff={n_eff}"
        )

        bp_star = _resolve_bp(
            base_protocol, n_eff, eps_q, delta_q, self.beta_prime,
            status_logger=status_logger,
        )
        dummy_star = _resolve_dummy_value(bp_star, default=0)

        # Step 3: run query for j* only via Simulator (streaming faithful)
        target_size = 2 ** j_star
        _log(
            f"One-round fast: step 3 — generate messages for j*={j_star}, "
            f"target_size={target_size}"
        )

        if hasattr(bp_star, "Simulator"):
            # Collect standardised records for j*
            records_star: list = []
            for D_i in datasets:
                m_i = len(D_i)
                if m_i < target_size:
                    D_tilde = list(D_i) + [dummy_star] * (target_size - m_i)
                else:
                    D_tilde = D_i[:target_size]
                records_star.extend(D_tilde)

            assert len(records_star) == n_eff, (
                f"Level {j_star}: expected {n_eff} records, got {len(records_star)}"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                nmsg_star, query_result = bp_star.Simulator(records_star)

            # Estimate total messages across ALL levels (one-round sends
            # messages for every level, not just j*).  Only j* needs real
            # generation for error computation; others just need counts.
            total_nmsg = nmsg_star
            for j in range(self.num_subdomains):
                if j == j_star:
                    continue
                eps_j, delta_j, n_eff_j = self.query_budget(j)
                bp_j = _resolve_bp(
                    base_protocol, n_eff_j, eps_j, delta_j, self.beta_prime,
                )
                est = self._estimate_query_messages(
                    bp_j, None, eps_j, delta_j, n_eff_j,
                )
                if math.isnan(est):
                    # Fallback: materialise records for this level
                    target_j = 2 ** j
                    dummy_j = _resolve_dummy_value(bp_j, default=0)
                    vals_j = self._collect_standardized_values(
                        datasets, target_j, dummy_j,
                    )
                    est = self._estimate_query_messages(
                        bp_j, vals_j, eps_j, delta_j, n_eff_j,
                    )
                total_nmsg += int(est)

            _log(
                f"One-round fast: finish, j*_messages={nmsg_star}, "
                f"total_messages={total_nmsg}"
            )
            return m_tau, query_result, total_nmsg

        # Fallback: full faithful path
        _log("One-round fast: no Simulator on bp, falling back to run()")
        return self.run(
            datasets, base_protocol, use_simulate=False,
            status_logger=status_logger,
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
        """Execute the full one-round protocol.
        
        Args:
            datasets:      List of n user datasets.
            base_protocol: Record-level protocol P_Q, or factory with
                          ``create(n_eff, eps, delta, beta)`` method.
                          None → counting only.
            use_simulate:  True = use the fast per-level Simulator path
                          when available; False = faithful per-record
                          LocalRandomizer + shuffle + Analyzer execution.
        
        Returns:
            (m_tau, query_result, nmessages).
        """
        assert len(datasets) == self.n, (
            f"Expected {self.n} users, got {len(datasets)}"
        )

        def _log(msg: str) -> None:
            if status_logger is not None:
                status_logger(msg)

        # Prepare one query-protocol instance per candidate threshold j.
        query_cfg: dict[int, tuple[Any, float, float, int, Any]] = {}
        if base_protocol is not None:
            for j in range(self.num_subdomains):
                eps_q, delta_q, n_eff = self.query_budget(j)
                bp_j = _resolve_bp(
                    base_protocol, n_eff, eps_q, delta_q, self.beta_prime,
                    status_logger=status_logger,
                )
                dummy_j = _resolve_dummy_value(bp_j, default=0)
                query_cfg[j] = (bp_j, eps_q, delta_q, n_eff, dummy_j)

        # -----------------------------------------------------------
        # Check if we can use the fast Simulator path.
        # When the base protocol exposes Simulator (e.g. analytical FE1),
        # we collect records per level and call Simulator once per level
        # instead of calling LocalRandomizer per-record.
        # -----------------------------------------------------------
        _can_simulate = (
            use_simulate
            and base_protocol is not None
            and query_cfg
            and hasattr(next(iter(query_cfg.values()))[0], "Simulator")
        )

        if _can_simulate:
            return self._run_simulate_path(datasets, query_cfg)

        # Faithful one-round randomizer:
        # generate counting + query messages for every level j.
        all_cnt: dict[int, list] = {j: [] for j in range(self.num_subdomains)}
        all_qry: dict[int, list] = {j: [] for j in range(self.num_subdomains)}

        next_pct = 10
        _log(
            "One-round faithful randomizer: "
            f"users={len(datasets)}, levels={self.num_subdomains}"
        )
        for idx, D_i in enumerate(datasets, start=1):
            m_i = len(D_i)
            for j in range(self.num_subdomains):
                indicator = self._in_subdomain(m_i, j)
                all_cnt[j].extend(self._gkmps_cnt.LocalRandomizer(indicator))

                if base_protocol is None:
                    continue

                bp_j, eps_q, delta_q, n_eff, dummy_j = query_cfg[j]
                target_size = 2**j
                if m_i < target_size:
                    D_tilde = list(D_i) + [dummy_j] * (target_size - m_i)
                else:
                    D_tilde = D_i[:target_size]

                if hasattr(bp_j, "randomizer"):
                    for record in D_tilde:
                        all_qry[j].extend(
                            bp_j.randomizer(record, eps_q, delta_q, n_eff)
                        )
                elif hasattr(bp_j, "LocalRandomizer"):
                    for record in D_tilde:
                        all_qry[j].extend(bp_j.LocalRandomizer(record))
                else:
                    raise TypeError(
                        "base_protocol must expose randomizer(...) "
                        "or LocalRandomizer(...)"
                    )
            next_pct = _log_progress(
                status_logger,
                "One-round faithful randomizer",
                idx,
                len(datasets),
                next_pct,
                step_pct=1,
            )

        # No explicit shuffle: analyzers depend on the message multiset, not order.

        # Analyzer step 1: estimate m_tau.
        _log("One-round analyzer step 1: estimate m_tau from counting channels")
        m_tau = 0
        for j in range(self.num_subdomains):
            noisy_count = self._gkmps_cnt.Analyzer(all_cnt[j])
            if noisy_count > self.threshold:
                m_tau = 2**j
        m_tau = max(m_tau, 1)
        _log(f"One-round analyzer step 1: finish, m_tau={m_tau}")

        if base_protocol is None:
            return m_tau, None, 0

        # Analyzer step 2: evaluate the selected query channel j* = log m_tau.
        j_star = int(math.log2(m_tau)) if m_tau > 1 else 0
        bp_star, eps_q, delta_q, n_eff, _ = query_cfg[j_star]
        qry_msgs = all_qry[j_star]
        _log(
            "One-round analyzer step 2: "
            f"start query channel j*={j_star}, messages={len(qry_msgs)}"
        )
        if hasattr(bp_star, "analyzer"):
            query_result = bp_star.analyzer(
                qry_msgs, eps_q, delta_q, self.beta_prime, n_eff
            )
        elif hasattr(bp_star, "Analyzer"):
            query_result = bp_star.Analyzer(qry_msgs)
        else:
            raise TypeError(
                "base_protocol must expose analyzer(...) or Analyzer(...)"
            )
        _log("One-round analyzer step 2: finish")

        total_nmsg = sum(len(msgs) for msgs in all_qry.values())
        return m_tau, query_result, total_nmsg
