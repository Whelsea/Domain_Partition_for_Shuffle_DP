"""
Unified query protocol factories for Round 2 (record-level query evaluation).

Provides factory classes that create base protocol instances with the correct
per-record privacy budget.  The budget depends on m_tau, which is determined
at runtime by the clipping-threshold protocol (two-round, one-round, etc.).

Factory pattern
---------------
    # In run_experiment.py — create factory (captures static params):
    factory = FE1Protocol(B=U+2, real_B=U+1, dummy_value=U+1, c=1.0)

    # Inside protocol.run() — create instance with runtime budget:
    instance = factory.create(n_eff, eps_rec, delta_rec, beta)
    nmessages, freq_vec = instance.Simulator(all_values)

Supported base protocols
------------------------
FE1     Frequency estimation [Luo et al., CCS 2022].
        Input:  values in [0, U_dummy] (0-indexed, our convention),
                where U_dummy can include one dummy symbol.
        Output: projected freq_vec on the real domain (dummy coordinate removed).

GKMPS   Sum estimation [Ghazi et al., ICML 2021].
        Input:  values (non-negative integers)
        Output: scalar estimated sum.

CLI usage
---------
    python run_experiment.py --dataset data.csv --protocols two_round --base_protocol FE1
    python run_experiment.py --dataset data.csv --protocols baseline_clip_M --base_protocol GKMPS
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Helper: resolve factory OR pre-configured instance
# ---------------------------------------------------------------------------

def resolve_base_protocol(
    factory_or_instance: Any,
    n_eff: int,
    epsilon: float,
    delta: float,
    beta: float,
) -> Any:
    """Resolve a query protocol factory into a configured instance.

    Supports two patterns:
      1. **Factory** — has ``create(n_eff, eps, delta, beta)`` method.
         Called to produce a fresh instance with the given budget.
      2. **Pre-configured instance** — returned as-is (backward compatible).

    Returns None when *factory_or_instance* is None.
    """
    if factory_or_instance is None:
        return None
    if hasattr(factory_or_instance, "create"):
        return factory_or_instance.create(n_eff, epsilon, delta, beta)
    return factory_or_instance


# ---------------------------------------------------------------------------
# Lazy import helpers (avoid loading numba/FE1 at module-import time)
# ---------------------------------------------------------------------------

_BASE_PROTOCOL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "Base_Protocol"
)
_SHUFFLE_FE_DIR = os.path.join(_BASE_PROTOCOL_DIR, "ShuffleFE-main")


def _ensure_fe1_path() -> None:
    if _SHUFFLE_FE_DIR not in sys.path:
        sys.path.insert(0, _SHUFFLE_FE_DIR)


def _ensure_gkmps_path() -> None:
    if _BASE_PROTOCOL_DIR not in sys.path:
        sys.path.insert(0, _BASE_PROTOCOL_DIR)


# ======================================================================
# FE1 — Frequency Estimation  [Luo et al., CCS 2022]
# ======================================================================

class _FE1Instance:
    """A configured FE1 instance with ``Simulator(values)`` interface.

    Created by :class:`FE1Protocol.create`.
    """

    def __init__(
        self,
        B: int,
        n_eff: int,
        epsilon: float,
        delta: float,
        beta: float,
        c: float,
        use_mu_search: bool,
        seed: int | None,
        workers: int | None,
        real_B: int | None,
        dummy_value: int | None,
        use_analytical: bool = False,
    ) -> None:
        _ensure_fe1_path()
        from FE1 import FE1Baseline

        self._B = B
        self._n_eff = n_eff
        self._workers = workers
        self._real_B = real_B
        self.padding_value = dummy_value
        self._use_analytical = use_analytical
        self._status_logger: Callable[[str], None] | None = None

        if self.padding_value is not None and not (0 <= self.padding_value < self._B):
            raise ValueError(
                f"dummy_value must be in [0, {self._B - 1}], got {self.padding_value}"
            )

        # When analytical simulation is requested, always use theoretical
        # mu (no binary-search cap) so the noise scales correctly for any
        # epsilon — including very small per-record epsilon in user-DP.
        actual_use_mu_search = use_mu_search and not use_analytical

        with contextlib.redirect_stdout(io.StringIO()):
            self._fe = FE1Baseline(
                n=n_eff,
                B=B,
                epsilon=epsilon,
                delta=delta,
                c=c,
                beta=beta,
                use_mu_search=actual_use_mu_search,
                seed=seed,
            )

    def set_status_logger(self, status_logger: Callable[[str], None] | None) -> None:
        self._status_logger = status_logger

    def _log(self, msg: str) -> None:
        if self._status_logger is not None:
            self._status_logger(msg)

    # --- GKMPS-compatible Simulator interface ---

    def _project_freq(self, freq_vec: np.ndarray) -> np.ndarray:
        """Project FE1 full-domain estimate onto the real domain."""
        full_result = freq_vec[1 : self._B + 1].copy()
        keep_values = [
            v for v in range(self._B)
            if self.padding_value is None or v != self.padding_value
        ]
        if self._real_B is not None:
            keep_values = keep_values[: self._real_B]
        return full_result[np.array(keep_values, dtype=np.int64)]

    # --- Faithful randomizer/analyzer interface ---

    def LocalRandomizer(self, value: int) -> list[tuple[int, int, int]]:
        """Per-record randomizer for faithful one-round execution."""
        assert 0 <= value < self._B, (
            f"FE1: value {value} outside [0, {self._B - 1}]"
        )
        return self._fe.local_randomizer(value + 1)

    def Analyzer(self, messages: list[tuple[int, int, int]]) -> np.ndarray:
        """Analyzer for faithful one-round execution."""
        self._log(f"FE1 analyzer: receive {len(messages):,} messages")
        freq_vec = self._fe.analyzer(
            messages,
            workers=self._workers,
            progress_logger=self._status_logger,
        )
        self._log("FE1 analyzer: project to real domain")
        return self._project_freq(freq_vec)

    def Simulator(self, values: list[int]) -> tuple[int, np.ndarray]:
        """Run FE1 frequency estimation on a flat list of record values.

        Dispatches to analytical simulation (fast, O(B)) when
        ``_use_analytical`` is True — this avoids generating billions of
        dummy messages when per-record epsilon is small.  Falls back to
        the faithful message-level simulation otherwise.

        Parameters
        ----------
        values : list[int]
            Record values in [0, B-1] (0-indexed, our convention).
            Length must equal n_eff (= n × m_tau after standardisation).
            If a dummy symbol is configured, its coordinate is projected out
            from the returned vector.

        Returns
        -------
        (nmessages, freq_result) where
            nmessages  — total shuffled messages sent
            freq_result — np.ndarray on the projected real domain,
                          freq_result[v] ≈ count of value v in values.
        """
        assert len(values) == self._n_eff, (
            f"FE1: expected {self._n_eff} values, got {len(values)}"
        )
        if values:
            vmin = min(values)
            vmax = max(values)
            assert 0 <= vmin and vmax < self._B, (
                f"FE1: value range [{vmin}, {vmax}] exceeds [0, {self._B - 1}]"
            )

        if self._use_analytical:
            return self._analytical_simulator(values)

        # --- Streaming faithful simulation ---
        # Process users in batches to bound memory at O(batch × msgs_per_user)
        # instead of O(n × msgs_per_user).  Produces statistically identical
        # results to materialising all messages, but scales to large mu.
        return self._streaming_faithful_simulator(values)

    # --- Streaming faithful simulator ---

    def _streaming_faithful_simulator(
        self,
        values: list[int],
        mem_limit_bytes: int = 2_000_000_000,   # 2 GB default
    ) -> tuple[int, np.ndarray]:
        """Faithful FE1 simulation with bounded memory.

        Generates real + dummy messages in user batches, counts bucket
        matches via the same numba JIT as FE1's ``analyzers_speedup_mp``,
        accumulates raw counts, and debiases once at the end.

        Statistically identical to the original randomize-all → analyzer
        path, but never allocates more than *mem_limit_bytes* for the
        message arrays.
        """
        _ensure_fe1_path()
        from FE1 import _process_chunk_jit

        fe   = self._fe
        n    = fe.n
        B    = fe.B
        q    = fe.q
        b    = fe.b
        rho  = fe.sample_prob
        pcol = fe.collision_prob
        fixed_send    = fe.send_fixed_messages
        remaining_prob = fe.remaining_prob
        rounds_base   = q // b + 1

        values_arr = np.asarray(values, dtype=np.int64) + 1   # 0-indexed → 1-indexed

        # Batch size: each message is 3 × int64 = 24 bytes.
        # Expected messages per user ≈ 1 + fixed_send + remaining_prob.
        msgs_per_user = 1 + fixed_send + (1 if remaining_prob > 0 else 0)
        batch_size = max(1, int(mem_limit_bytes / (msgs_per_user * 24)))
        batch_size = min(batch_size, n)

        rng = np.random.default_rng()
        raw_counts = np.zeros(B + 1, dtype=np.int64)
        total_msgs = 0

        # Warm up numba JIT (first call compiles; do it with a tiny array).
        _tiny = np.zeros(1, dtype=np.int64)
        _process_chunk_jit(_tiny, _tiny, _tiny, 0, 0, B, q, b, rounds_base)

        total_batches = (n + batch_size - 1) // batch_size
        next_pct = 10
        self._log(
            "FE1 simulator: "
            f"faithful batching start, users={n}, batches={total_batches}, "
            f"batch_size={batch_size}"
        )
        for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            bn  = end - start
            batch_vals = values_arr[start:end]

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

            del U, V, W

            if self._status_logger is not None and total_batches > 0:
                pct = int((batch_idx * 100) / total_batches)
                while next_pct <= 100 and pct >= next_pct:
                    self._log(
                        f"FE1 simulator: {next_pct}% ({batch_idx}/{total_batches} batches)"
                    )
                    next_pct += 10

        # ---- Debias (same formula as FE1's analyzer) ----
        self._log("FE1 simulator: debias counts")
        freq_vec = raw_counts.astype(np.float64)
        freq_vec = (freq_vec - n * rho / b - n * pcol) / (1.0 - pcol)
        freq_vec[0] = 0.0

        freq_result = self._project_freq(freq_vec)
        self._log(f"FE1 simulator: finish, total_messages={total_msgs}")
        return total_msgs, freq_result

    # --- Analytical (fast) simulator ---

    def _analytical_simulator(
        self, values: list[int]
    ) -> tuple[int, np.ndarray]:
        """Analytical FE1 simulation — O(B) time, no messages generated.

        Uses the closed-form noise distribution of FE1's estimator so
        that we can simulate at any epsilon (including very small ones)
        without actually creating dummy messages.

        Noise model per coordinate j (FE1's debiased estimator):
            hat{f}_j = (cnt_j - n*rho/b - n*pcol) / (1 - pcol)
        where cnt_j ~ Binom(n, pcol) + Binom(n*rho, 1/b) + f_j
        (approximately).  By CLT the estimator noise is Gaussian with

            Var(hat{f}_j) = [n*pcol*(1-pcol)
                             + n*rho*(1/b)*(1-1/b)] / (1-pcol)^2
        """
        import math as _math

        fe = self._fe
        n   = fe.n
        b   = fe.b
        rho = fe.sample_prob
        pcol = fe.collision_prob

        # 1. True frequency vector (1-indexed, [1..B])
        true_freq = np.zeros(fe.B + 1, dtype=np.float64)
        for v in values:
            true_freq[v + 1] += 1          # 0-indexed → 1-indexed

        # 2. Noise variance per coordinate
        var_collision = n * pcol * (1.0 - pcol)
        var_dummy     = n * rho  * (1.0 / b) * (1.0 - 1.0 / b)
        denom = (1.0 - pcol) ** 2
        if denom < 1e-15:
            # pcol ≈ 1 → estimator is degenerate (b too small).
            # Fall back to very-large-noise regime.
            sigma = 1e12
        else:
            var_total = (var_collision + var_dummy) / denom
            sigma = _math.sqrt(max(var_total, 0.0))

        # 3. Add Gaussian noise
        rng = np.random.default_rng()
        noise = rng.normal(0.0, sigma, size=fe.B + 1)
        freq_vec = true_freq + noise
        freq_vec[0] = 0.0                  # coordinate 0 unused in FE1

        # 4. Expected message count
        nmessages = int(round(n * (1.0 + rho)))

        # 5. Project to real domain
        freq_result = self._project_freq(freq_vec)
        return nmessages, freq_result

    @property
    def theta(self) -> float:
        """FE1 error bound θ (per-query)."""
        return self._fe.get_theta_fe1()


class FE1Protocol:
    """Factory for FE1 frequency estimation base protocol.

    Captures static parameters (domain size B, utility param c).
    Call :meth:`create` with runtime budget to get a configured instance.

    Parameters
    ----------
    B : int
        FE1 total domain size (including optional dummy value).
        For real domain [0, U] with dummy U+1, set ``B = U + 2``.
    real_B : int, optional
        Number of real-domain coordinates to return after projection.
        For [0, U], set ``real_B = U + 1``.
    dummy_value : int, optional
        0-indexed dummy value to project out from FE1 output.
    c : float
        Utility parameter controlling hash bucket size b = n / log^c(n).
        c=1 → optimal error, ~O(1) msgs/user.
        c=3 → ~1+o(1) msgs/user, O(log²n) error.
    use_mu_search : bool
        Use binary search for μ (True) or theoretical bound (False).
    seed : int, optional
        Random seed for reproducibility.
    workers : int, optional
        Multiprocessing workers for the analyzer.
        None = auto (cpu_count - 1).
    """

    name = "FE1"

    def __init__(
        self,
        B: int,
        real_B: int | None = None,
        dummy_value: int | None = None,
        c: float = 1.0,
        use_mu_search: bool = True,
        use_analytical: bool = False,
        seed: int | None = None,
        workers: int | None = None,
    ) -> None:
        self.B = B
        self.real_B = real_B
        self.padding_value = dummy_value
        self.c = c
        self.use_mu_search = use_mu_search
        self.use_analytical = use_analytical
        self.seed = seed
        self.workers = workers
        self._create_calls = 0
        self._created_params: list[dict[str, Any]] = []

    def create(
        self, n_eff: int, epsilon: float, delta: float, beta: float
    ) -> _FE1Instance:
        """Create a configured FE1 instance with the given budget."""
        seed = None
        if self.seed is not None:
            # Deterministic but different stream for each create() call.
            seed = self.seed + self._create_calls
        self._create_calls += 1

        inst = _FE1Instance(
            B=self.B,
            n_eff=n_eff,
            epsilon=epsilon,
            delta=delta,
            beta=beta,
            c=self.c,
            use_mu_search=self.use_mu_search,
            seed=seed,
            workers=self.workers,
            real_B=self.real_B,
            dummy_value=self.padding_value,
            use_analytical=self.use_analytical,
        )
        self._created_params.append(
            {
                "n_eff": int(n_eff),
                "epsilon": float(epsilon),
                "delta": float(delta),
                "utility_parameter": float(inst._fe.c),
                "modular_size": int(inst._fe.b),
                "big_prime": int(inst._fe.q),
                "mu": float(inst._fe.mu),
                "sample_prob": float(inst._fe.sample_prob),
                "collision_probability": float(inst._fe.collision_prob),
            }
        )
        return inst

    def consume_created_params(self) -> list[dict[str, Any]]:
        """Return and clear FE1 create() parameter logs."""
        out = self._created_params
        self._created_params = []
        return out


# ======================================================================
# GKMPS — Sum Estimation  [Ghazi et al., ICML 2021]
# ======================================================================

class GKMPSSumProtocol:
    """Factory for GKMPS sum estimation base protocol.

    Parameters
    ----------
    domain : int
        Value domain size (max value).
    gamma : float
        Noise allocation parameter.
    """

    name = "GKMPS"

    def __init__(self, domain: int, gamma: float = 0.3) -> None:
        self.domain = domain
        self.gamma = gamma

    def create(
        self, n_eff: int, epsilon: float, delta: float, beta: float
    ) -> Any:
        """Create a configured GKMPS instance."""
        _ensure_gkmps_path()
        from GKMPS import GKMPS

        return GKMPS(
            n=n_eff,
            domain=self.domain,
            epsilon=epsilon,
            delta=delta,
            gamma=self.gamma,
        )


# ======================================================================
# Registry
# ======================================================================

QUERY_PROTOCOL_REGISTRY: dict[str, type] = {
    "FE1": FE1Protocol,
    "GKMPS": GKMPSSumProtocol,
}
