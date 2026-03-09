"""
UserLevelDataset + DatasetGenerator — canonical data module for user-level shuffle-DP.

Convention (follows the paper §2):
    [U] = {0, 1, ..., U}           — domain has U+1 distinct values
    D_i = (x_{i,1}, ..., x_{i,m_i}) — user i's records, m_i ≤ M
    Each x_{i,k} ∈ {0, 1, ..., U}   — a single integer record value

This module contains:
    UserLevelDataset   — data container (no privacy logic)
    DatasetGenerator   — two-stage simulated dataset generation

Typical flow:
    dataset = DatasetGenerator.generate(n=1000, M=2**20, m_max=1024, U=100, seed=42)
    dataset.save_csv("data/n1000_U100_zipf_mmax1024_seed42.csv")

    dataset = UserLevelDataset.load_csv("data/n1000_U100_zipf_mmax1024_seed42.csv")
    proto.run(dataset.to_protocol_input(), base_protocol=...)

Supported contribution distributions (Stage 1)
-----------------------------------------------
uniform_fixed   Every user has exactly *m_each* records.
uniform_random  m_i ~ Uniform({low, …, high}), clipped to [1, m_max_target].
zipf            Standard Zipf(alpha), clipped to [1, m_max_target].
                If ``shift`` is provided, sample from the finite-support law
                on ``{1, ..., m_max_target}`` with
                ``Pr[m_i = x] ∝ (x + shift)^(-alpha)``.
geometric       m_i ~ Geometric(p), clipped to [1, m_max_target].
gaussian        m_i ~ N(mu, sigma) rounded & clipped to [1, m_max_target].
one_heavy       One user with m_heavy records, rest with m_rest.
mixed           n_heavy heavy users, rest light users.

Supported value distributions (Stage 2)
----------------------------------------
uniform         x ~ Uniform({0, 1, …, U}).
zipf            x ~ Zipf(alpha), mapped into {0, …, U}.
gaussian        x ~ N(mu, sigma), discretised & clipped to {0, …, U}.
constant        x = value  (useful for unit tests / counting-only runs).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


DEFAULT_CONTRIB_ZIPF_ALPHA = 6.0
DEFAULT_CONTRIB_ZIPF_SHIFT = 1.36
DEFAULT_VALUE_ZIPF_ALPHA = 1.5


# ======================================================================
# UserLevelDataset — data container
# ======================================================================


@dataclass
class UserLevelDataset:
    """A user-level dataset for shuffle-DP experiments.

    Attributes
    ----------
    records : list[list[int]]
        ``records[i]`` is the list of record values for user *i*.
        Each value is an integer in {0, 1, …, U}.
        ``len(records) == n``.
        ``len(records[i]) == m_i`` (variable, bounded by M).
    n : int
        Number of users.
    M : int
        Global upper bound on records per user.
        Required by all protocols for budget computation.
    U : int
        Domain upper bound.  Record values live in {0, 1, …, U}.
        Domain size = U + 1.
    metadata : dict
        Free-form metadata (generation parameters, source, timestamps …).
    """

    records: list[list[int]]
    n: int
    M: int
    U: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if len(self.records) != self.n:
            raise ValueError(
                f"records has {len(self.records)} users but n={self.n}"
            )

    def validate(self, strict: bool = False) -> list[str]:
        """Check data integrity.  Returns list of warnings.

        When *strict* is True, raises ValueError on the first problem.
        """
        warnings: list[str] = []

        for i, rec in enumerate(self.records):
            if len(rec) > self.M:
                msg = f"User {i}: m_i={len(rec)} > M={self.M}"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
            for k, v in enumerate(rec):
                if not isinstance(v, (int, np.integer)):
                    msg = f"User {i}, record {k}: non-integer value {v!r}"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
                elif v < 0 or v > self.U:
                    msg = (
                        f"User {i}, record {k}: value {v} "
                        f"outside domain [0, {self.U}]"
                    )
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
        return warnings

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def contributions(self) -> list[int]:
        """Number of records per user: ``[m_1, m_2, …, m_n]``."""
        return [len(r) for r in self.records]

    @property
    def m_max(self) -> int:
        """Actual maximum contribution in this dataset."""
        return max((len(r) for r in self.records), default=0)

    @property
    def total_records(self) -> int:
        """Total number of records across all users."""
        return sum(len(r) for r in self.records)

    @property
    def domain_size(self) -> int:
        """Number of distinct domain values: U + 1."""
        return self.U + 1

    def statistics(self) -> dict[str, Any]:
        """Summary statistics (safe for JSON serialisation)."""
        c = self.contributions
        return {
            "n": self.n,
            "M": self.M,
            "U": self.U,
            "domain_size": self.domain_size,
            "m_max": int(max(c)) if c else 0,
            "m_min": int(min(c)) if c else 0,
            "m_mean": float(np.mean(c)) if c else 0.0,
            "m_median": float(np.median(c)) if c else 0.0,
            "total_records": int(sum(c)),
        }

    # ------------------------------------------------------------------
    # Protocol integration
    # ------------------------------------------------------------------

    def to_protocol_input(self) -> list[list[int]]:
        """Return records in the ``list[list[int]]`` format all protocols accept.

        Equivalent to ``self.records`` — provided for API clarity so
        call sites read ``dataset.to_protocol_input()`` rather than
        relying on internal attribute names.
        """
        return self.records

    # ------------------------------------------------------------------
    # Persistence  (JSON — human-readable, inspectable)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save dataset to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "n": self.n,
            "M": self.M,
            "U": self.U,
            "records": self.records,
            "metadata": self.metadata,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> UserLevelDataset:
        """Load dataset from a JSON file."""
        with open(path) as fh:
            d = json.load(fh)
        return cls(
            records=d["records"],
            n=d["n"],
            M=d["M"],
            U=d["U"],
            metadata=d.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Persistence  (CSV — lightweight, standard, real-data compatible)
    # ------------------------------------------------------------------

    def save_csv(self, path: str) -> None:
        """Save dataset to a CSV file.

        Format::

            # n=1000,M=64,U=100
            # actual_m_max=57,m_mean=3.142000
            # source=simulated,contrib_dist=zipf,value_dist=uniform,seed=42
            86,77,64,90,78
            96
            44
            10,7,82,51,62

        - Lines starting with ``#`` are metadata.
        - First metadata line **must** contain ``n``, ``M``, ``U``.
        - Each subsequent line is one user's records (comma-separated ints).
        - An empty line represents a user with 0 records.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        stats = self.statistics()
        with open(path, "w") as fh:
            # Line 1: essential parameters
            fh.write(f"# n={self.n},M={self.M},U={self.U}\n")
            # Line 2: dataset summary statistics
            fh.write(
                "# "
                f"actual_m_max={stats['m_max']},"
                f"m_mean={stats['m_mean']:.6f}\n"
            )
            # Line 3: metadata (generation info)
            if self.metadata:
                meta_parts = [f"{k}={v}" for k, v in self.metadata.items()
                              if not isinstance(v, (dict, list))]
                if meta_parts:
                    fh.write(f"# {','.join(meta_parts)}\n")
            # Data lines: one user per line
            for user_recs in self.records:
                if user_recs:
                    fh.write(",".join(str(v) for v in user_recs) + "\n")
                else:
                    fh.write("\n")

    @classmethod
    def load_csv(cls, path: str) -> UserLevelDataset:
        """Load dataset from a CSV file produced by :meth:`save_csv`.

        Reads ``n``, ``M``, ``U`` from the ``# n=...,M=...,U=...`` header.
        Each non-comment line is parsed as one user's comma-separated records.
        """
        metadata: dict[str, Any] = {}
        n_hdr: int | None = None
        M_hdr: int | None = None
        U_hdr: int | None = None
        records: list[list[int]] = []

        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("#"):
                    # Parse metadata from comment lines
                    content = line[1:].strip()
                    for part in content.split(","):
                        part = part.strip()
                        if "=" in part:
                            k, v = part.split("=", 1)
                            k, v = k.strip(), v.strip()
                            if k == "n":
                                n_hdr = int(v)
                            elif k == "M":
                                M_hdr = int(v)
                            elif k == "U":
                                U_hdr = int(v)
                            else:
                                metadata[k] = v
                    continue
                # Data line
                line = line.strip()
                if line == "":
                    records.append([])
                else:
                    records.append([int(x) for x in line.split(",")])

        if n_hdr is None or M_hdr is None or U_hdr is None:
            raise ValueError(
                f"CSV header missing required fields. "
                f"Expected '# n=...,M=...,U=...' but got n={n_hdr}, M={M_hdr}, U={U_hdr}"
            )

        # Validate n matches actual record count
        if len(records) != n_hdr:
            raise ValueError(
                f"Header says n={n_hdr} but file has {len(records)} data lines"
            )

        return cls(
            records=records,
            n=n_hdr,
            M=M_hdr,
            U=U_hdr,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_raw(
        cls,
        records: list[list[int]],
        M: int,
        U: int,
        metadata: dict[str, Any] | None = None,
    ) -> UserLevelDataset:
        """Create from pre-existing records (e.g. preprocessed real data).

        ``n`` is inferred from ``len(records)``.
        """
        return cls(
            records=records,
            n=len(records),
            M=M,
            U=U,
            metadata=metadata or {"source": "raw"},
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UserLevelDataset(n={self.n}, M={self.M}, U={self.U}, "
            f"m_max={self.m_max}, total={self.total_records})"
        )

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        s = self.statistics()
        lines = [
            f"UserLevelDataset",
            f"  users       (n) : {s['n']}",
            f"  max bound   (M) : {s['M']}",
            f"  domain      (U) : {s['U']}  (domain_size = {s['domain_size']})",
            f"  m_max           : {s['m_max']}",
            f"  m_min           : {s['m_min']}",
            f"  m_mean          : {s['m_mean']:.2f}",
            f"  m_median        : {s['m_median']:.1f}",
            f"  total records   : {s['total_records']}",
        ]
        if self.metadata:
            lines.append(f"  metadata        : {self.metadata}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Canonical file naming
    # ------------------------------------------------------------------

    @staticmethod
    def canonical_filename(
        n: int,
        M: int,
        U: int,
        m_max: int | None = None,
        contrib_dist: str = "zipf",
        seed: int | None = None,
    ) -> str:
        """Generate a canonical filename from dataset parameters.

        Examples::

            n1000_U100_zipf_mmax64_seed42.csv
            n1000_U100_zipf_mmax1024_seed42.csv
            n500_U50_gaussian_mmax32_seed0.csv
            n1000_U100_zipf_mmax64.csv          # no seed
        """
        effective_m_max = M if m_max is None else m_max
        name = f"n{n}_U{U}_{contrib_dist}_mmax{effective_m_max}"
        if seed is not None:
            name += f"_seed{seed}"
        return name + ".csv"


# ======================================================================
# Stage 1 — contribution count samplers
# ======================================================================

def _sample_contributions(
    n: int,
    m_max: int,
    distribution: str,
    rng: np.random.Generator,
    **params: Any,
) -> np.ndarray:
    """Return an array of *n* contribution counts, each in [1, m_max]."""

    if distribution == "uniform_fixed":
        m_each = min(params.get("m_each", 5), m_max)
        return np.full(n, m_each, dtype=int)

    if distribution == "uniform_random":
        lo = max(params.get("low", 1), 1)
        hi = min(params.get("high", m_max), m_max)
        return rng.integers(lo, hi + 1, size=n)

    if distribution == "zipf":
        alpha = params.get("alpha", 1.5)
        shift = params.get("shift", 0.0)
        if shift != 0.0:
            return _sample_shifted_zipf(n, m_max, alpha, shift, rng)
        m = rng.zipf(alpha, size=n)
        return np.clip(m, 1, m_max).astype(int)

    if distribution == "geometric":
        p = params.get("p", 0.3)
        m = rng.geometric(p, size=n)
        return np.clip(m, 1, m_max).astype(int)

    if distribution == "gaussian":
        mu = params.get("mu", m_max / 2)
        sigma = params.get("sigma", m_max / 4)
        m = rng.normal(mu, sigma, size=n)
        m = np.round(m).astype(int)
        return np.clip(m, 1, m_max).astype(int)

    if distribution == "one_heavy":
        m_heavy = min(params.get("m_heavy", m_max), m_max)
        m_rest = max(min(params.get("m_rest", 1), m_max), 1)
        counts = np.full(n, m_rest, dtype=int)
        counts[-1] = m_heavy          # last user is the heavy one
        return counts

    if distribution == "mixed":
        n_heavy = params.get("n_heavy", max(1, n // 10))
        m_heavy = min(params.get("m_heavy", 10), m_max)
        m_rest = max(min(params.get("m_rest", 1), m_max), 1)
        counts = np.full(n, m_rest, dtype=int)
        counts[:n_heavy] = m_heavy
        return counts

    raise ValueError(f"Unknown contribution distribution: {distribution!r}")


def _sample_shifted_zipf(
    n: int,
    m_max: int,
    alpha: float,
    shift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` values from ``Pr[X=x] ∝ (x + shift)^(-alpha)`` on ``[1, m_max]``."""

    if alpha <= 0:
        raise ValueError(f"zipf alpha must be positive, got {alpha}")
    if shift <= -1.0:
        raise ValueError(f"zipf shift must satisfy shift > -1, got {shift}")

    support = np.arange(1, m_max + 1, dtype=np.float64)
    shifted = support + shift
    if np.any(shifted <= 0):
        raise ValueError(f"zipf shift produced non-positive support values: shift={shift}")

    # Build the CDF once, then sample via inverse transform. This is faster
    # than materialising per-user weights when n is large.
    log_weights = -alpha * np.log(shifted)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    cdf = np.cumsum(weights / np.sum(weights))

    uniforms = rng.random(n)
    return np.searchsorted(cdf, uniforms, side="right").astype(int) + 1


# ======================================================================
# Stage 2 — record value samplers
# ======================================================================

def _sample_values(
    count: int,
    U: int,
    distribution: str,
    rng: np.random.Generator,
    **params: Any,
) -> list[int]:
    """Return a list of *count* integer values, each in {0, …, U}."""

    if count == 0:
        return []

    if distribution == "uniform":
        # Uniform over {0, 1, ..., U}
        return rng.integers(0, U + 1, size=count).tolist()

    if distribution == "zipf":
        alpha = params.get("alpha", 1.5)
        raw = rng.zipf(alpha, size=count)
        # Zipf naturally produces values ≥ 1.  Shift to 0-indexed and clip.
        return np.clip(raw - 1, 0, U).astype(int).tolist()

    if distribution == "gaussian":
        mu = params.get("mu", U / 2)
        sigma = params.get("sigma", U / 6)
        raw = rng.normal(mu, sigma, size=count)
        raw = np.round(raw).astype(int)
        return np.clip(raw, 0, U).astype(int).tolist()

    if distribution == "constant":
        v = min(int(params.get("value", 1)), U)
        return [v] * count

    raise ValueError(f"Unknown value distribution: {distribution!r}")


# ======================================================================
# DatasetGenerator — two-stage generation
# ======================================================================

class DatasetGenerator:
    """Two-stage simulated dataset generator.

    Usage
    -----
    >>> ds = DatasetGenerator.generate(
    ...     n=1000, M=2**20, m_max=1024, U=100,
    ...     contrib_dist="zipf",   contrib_params={"alpha": 1.5},
    ...     value_dist="uniform",
    ...     seed=42,
    ... )
    >>> ds
    UserLevelDataset(n=1000, M=1048576, U=100, m_max=..., total=...)
    """

    # Expose samplers so callers can enumerate supported distributions
    CONTRIB_DISTRIBUTIONS = (
        "uniform_fixed",
        "uniform_random",
        "zipf",
        "geometric",
        "gaussian",
        "one_heavy",
        "mixed",
    )
    VALUE_DISTRIBUTIONS = (
        "uniform",
        "zipf",
        "gaussian",
        "constant",
    )

    @classmethod
    def generate(
        cls,
        n: int,
        M: int,
        U: int,
        m_max: int | None = None,
        contrib_dist: str = "zipf",
        value_dist: str = "uniform",
        contrib_params: dict[str, Any] | None = None,
        value_params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> UserLevelDataset:
        """Generate a simulated user-level dataset.

        Parameters
        ----------
        n : int
            Number of users.
        M : int
            Global public upper bound on records per user.
        m_max : int, optional
            Target maximum contribution in generated data (private statistic).
            Actual generated counts satisfy ``m_i <= m_max <= M``.
            If omitted, defaults to ``M``.
        U : int
            Domain upper bound.  Record values in {0, …, U}.
        contrib_dist : str
            Distribution name for contribution counts (Stage 1).
        value_dist : str
            Distribution name for record values (Stage 2).
        contrib_params : dict, optional
            Extra parameters forwarded to the Stage 1 sampler.
        value_params : dict, optional
            Extra parameters forwarded to the Stage 2 sampler.
        seed : int, optional
            Random seed for full reproducibility.

        Returns
        -------
        UserLevelDataset
        """
        _cp = contrib_params or {}
        _vp = value_params or {}
        if contrib_dist == "zipf":
            _cp = dict(_cp)
            _cp.setdefault("alpha", DEFAULT_CONTRIB_ZIPF_ALPHA)
            _cp.setdefault("shift", DEFAULT_CONTRIB_ZIPF_SHIFT)
        if value_dist == "zipf":
            _vp = dict(_vp)
            _vp.setdefault("alpha", DEFAULT_VALUE_ZIPF_ALPHA)
        rng = np.random.default_rng(seed)
        target_m_max = M if m_max is None else m_max

        if target_m_max <= 0:
            raise ValueError(f"m_max must be positive, got {target_m_max}")
        if target_m_max > M:
            raise ValueError(f"m_max must satisfy m_max <= M, got m_max={target_m_max}, M={M}")

        # Stage 1 — contribution counts
        m_values = _sample_contributions(n, target_m_max, contrib_dist, rng, **_cp)

        # Stage 2 — record values per user
        records: list[list[int]] = []
        for m_i in m_values:
            user_recs = _sample_values(int(m_i), U, value_dist, rng, **_vp)
            records.append(user_recs)

        metadata: dict[str, Any] = {
            "source": "simulated",
            "contrib_dist": contrib_dist,
            "contrib_params": _cp,
            "m_max_target": target_m_max,
            "value_dist": value_dist,
            "value_params": _vp,
            "seed": seed,
        }

        return UserLevelDataset(
            records=records,
            n=n,
            M=M,
            U=U,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Preset recipes (from experiment outline)
    # ------------------------------------------------------------------

    @classmethod
    def zipf_uniform(
        cls,
        n: int = 1000,
        M: int = 64,
        U: int = 100,
        alpha: float = DEFAULT_CONTRIB_ZIPF_ALPHA,
        shift: float = DEFAULT_CONTRIB_ZIPF_SHIFT,
        seed: int | None = None,
    ) -> UserLevelDataset:
        """Zipf contributions + Uniform values (common default)."""
        return cls.generate(
            n=n, M=M, U=U,
            contrib_dist="zipf", contrib_params={"alpha": alpha, "shift": shift},
            value_dist="uniform",
            seed=seed,
        )

    @classmethod
    def gaussian_gaussian(
        cls,
        n: int = 1000,
        M: int = 64,
        U: int = 100,
        contrib_mu: float = 10.0,
        contrib_sigma: float = 5.0,
        value_mu: float | None = None,
        value_sigma: float | None = None,
        seed: int | None = None,
    ) -> UserLevelDataset:
        """Gaussian contributions + Gaussian values."""
        return cls.generate(
            n=n, M=M, U=U,
            contrib_dist="gaussian",
            contrib_params={"mu": contrib_mu, "sigma": contrib_sigma},
            value_dist="gaussian",
            value_params={
                "mu": value_mu if value_mu is not None else U / 2,
                "sigma": value_sigma if value_sigma is not None else U / 6,
            },
            seed=seed,
        )

    @classmethod
    def uniform_uniform(
        cls,
        n: int = 1000,
        M: int = 64,
        U: int = 100,
        m_each: int = 5,
        seed: int | None = None,
    ) -> UserLevelDataset:
        """Fixed-count contributions + Uniform values."""
        return cls.generate(
            n=n, M=M, U=U,
            contrib_dist="uniform_fixed", contrib_params={"m_each": m_each},
            value_dist="uniform",
            seed=seed,
        )
