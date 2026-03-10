import math
import time
import numpy as np
from typing import Callable, List, Tuple
from collections import Counter
from multiprocessing import Pool, cpu_count, set_start_method

# macOS/Windows: spawn；Linux: fork
try:
    set_start_method("spawn")
except RuntimeError:
    pass

# ====================== Numba：Speedup ======================
import numba

@numba.njit
def mod_pow(base: int, exp: int, mod: int) -> int:
    res = 1
    b = base % mod
    e = exp
    while e > 0:
        if e & 1:
            res = (res * b) % mod
        b = (b * b) % mod
        e >>= 1
    return res

@numba.njit
def _process_chunk_jit(U, V, W, start, end, B, q, b, rounds_base):
    local = np.zeros(B + 1, dtype=np.int64)
    for i in range(start, end):
        u = U[i]
        v = V[i]
        w = W[i]
        invu = mod_pow(u, q - 2, q)
        start_id = (invu * (w - v + q)) % q
        step     = (invu * b) % q
        id_ = start_id
        rounds = ((q - 1 - w) // b) + 1
        for _ in range(rounds):
            if 1 <= id_ <= B:
                local[id_] += 1
            id_ += step
            if id_ >= q:
                id_ -= q
    return local

def _mp_worker(args):
    return _process_chunk_jit(*args)


def _log_progress(
    progress_logger: Callable[[str], None] | None,
    label: str,
    completed: int,
    total: int,
    next_pct: int,
    step_pct: int = 10,
) -> int:
    if progress_logger is None or total <= 0:
        return next_pct
    pct = int((completed * 100) / total)
    while next_pct <= 100 and pct >= next_pct:
        progress_logger(f"{label}: {next_pct}% ({completed}/{total})")
        next_pct += step_pct
    return next_pct

def analyzers_speedup_mp(U: np.ndarray, V: np.ndarray, W: np.ndarray,
                         B: int, q: int, b: int, n: int, rho: float, pcol: float,
                         workers: int | None = None,
                         progress_logger: Callable[[str], None] | None = None) -> np.ndarray:
    assert U.shape == V.shape == W.shape
    if U.dtype != np.int64: U = U.astype(np.int64, copy=False)
    if V.dtype != np.int64: V = V.astype(np.int64, copy=False)
    if W.dtype != np.int64: W = W.astype(np.int64, copy=False)

    M = U.shape[0]
    rounds_base = q // b + 1

    # Warm-up JIT (Avoid the first compilation of child processes)
    _process_chunk_jit(U[:1], V[:1], W[:1], 0, 1, B, q, b, 1)

    if workers is None:
        workers = max(1, cpu_count() - 1)

    if progress_logger is not None:
        progress_logger(
            "FE1 analyzer: "
            f"total_messages={M}, workers={workers}, rounds_base={rounds_base}"
        )

    freq_counts = np.zeros(B + 1, dtype=np.int64)

    if workers <= 1:
        chunk_count = min(max(10, 1), M)
        bounds = np.linspace(0, M, chunk_count + 1, dtype=np.int64)
        tasks = []
        for i in range(chunk_count):
            start = int(bounds[i]); end = int(bounds[i + 1])
            if start < end:
                tasks.append((U, V, W, start, end, int(B), int(q), int(b), int(rounds_base)))

        next_pct = 10
        for idx, task in enumerate(tasks, start=1):
            freq_counts += _mp_worker(task)
            next_pct = _log_progress(
                progress_logger, "FE1 analyzer", idx, len(tasks), next_pct
            )
    else:
        bounds = np.linspace(0, M, workers + 1, dtype=np.int64)
        tasks = []
        for i in range(workers):
            start = int(bounds[i]); end = int(bounds[i + 1])
            if start < end:
                tasks.append((U, V, W, start, end, int(B), int(q), int(b), int(rounds_base)))

        next_pct = 10
        with Pool(processes=workers) as pool:
            for idx, part in enumerate(pool.imap_unordered(_mp_worker, tasks), start=1):
                freq_counts += part
                next_pct = _log_progress(
                    progress_logger, "FE1 analyzer", idx, len(tasks), next_pct
                )

    if progress_logger is not None:
        progress_logger("FE1 analyzer: debias counts")

    freq = freq_counts.astype(np.float64)
    freq = (freq - n * rho / b - n * pcol) / (1.0 - pcol)
    if progress_logger is not None:
        progress_logger("FE1 analyzer: finish")
    return freq

# ====================== search μ  ======================
def next_prime_at_least(m: int) -> int:
    if m <= 2:
        return 2
    def is_prime(p: int) -> bool:
        small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for tp in small:
            if p == tp:
                return True
            if p % tp == 0:
                return False
        d, s = p - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 7, 61]:
            if a % p == 0:
                continue
            x = pow(a, d, p)
            if x == 1 or x == p - 1:
                continue
            ok = False
            for _ in range(s - 1):
                x = (x * x) % p
                if x == p - 1:
                    ok = True
                    break
            if not ok:
                return False
        return True
    q = max(2, m)
    while not is_prime(q):
        q += 1
    return q

def mu_search(n: int, epsilon: float, delta: float) -> float:
    epow = math.exp(epsilon)
    def checker(p: float) -> bool:
        prob = np.empty(n + 1, dtype=np.float64)
        accprob = np.empty(n + 1, dtype=np.float64)
        tp2 = np.empty(n + 1, dtype=np.float64)
        tp2[0] = 1.0
        for i in range(1, n + 1):
            tp2[i] = tp2[i - 1] * (1.0 - p)
        C = 1.0
        prob[0] = C * tp2[n]
        for i in range(1, n + 1):
            C = C * (n - (i - 1)) * p / i
            prob[i] = C * tp2[n - i]
        accprob[n] = prob[n]
        for i in range(n - 1, -1, -1):
            accprob[i] = accprob[i + 1] + prob[i]
        pro = 0.0
        for x2 in range(0, n + 1):
            x1 = int(math.ceil(epow * x2 - 1.0))
            if x1 < 0:
                x1 = 0
            if x1 >= n:
                break
            pro += prob[x2] * accprob[x1]
        return pro <= delta
    le, ri = 0.0, 1000.0 / n
    while le + 0.1 / n < ri:
        mi = (le + ri) * 0.5
        if checker(mi):
            ri = mi
        else:
            le = mi
    # The optimal mu is ri * n.
    return ri * n

    # Note on small epsilon: The optimal mu derived from the binary search (ri * n)
    # may exhibit poor performance (or stability) when epsilon is small.
    # Therefore, we use the theoretical bound presented in the FE1 paper as a robust and stable alternative.
    # return 32 * math.log(2 / delta) / (epsilon ** 2)


# ======================  FE1  ======================
class FE1Baseline:
    def __init__(self, n: int, B: int, epsilon: float, delta: float, c: float, beta: float,
                 use_mu_search: bool = True, seed: int | None = None):
        self.n = int(n)
        self.B = int(B)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.c = float(c)
        self.beta = beta

        if use_mu_search:
            self.mu = mu_search(self.n, self.epsilon, self.delta)
            self.b = max(2, int(n / (math.log(n) ** self.c)))
        else:
            self.mu = 32.0 * math.log(2.0 / self.delta) / (self.epsilon ** 2)
            self.b = int(max(2, epsilon ** 2 * n / math.pow(math.log(2.0/self.delta), c)))
        print("eps:", self.epsilon, "delta:",self.delta,"mu:",self.mu,"b:",self.b)
        self.q = next_prime_at_least(max(2, self.B, self.b))
        self.sample_prob = self.mu * (self.b / self.n)
        self.send_fixed_messages = int(math.floor(self.sample_prob))
        self.remaining_prob = self.sample_prob - self.send_fixed_messages

        self.collision_prob = (self.q // self.b) * ((self.q % self.b) + self.q - self.b) / (self.q * (self.q - 1))

        self.messages: List[Tuple[int, int, int]] = []
        self.rng = np.random.default_rng(seed)

    def bits_per_message(self) -> int:
        return math.ceil(math.log2(self.q)) * 2 + math.ceil(math.log2(self.b))

    def local_randomizer(self, x: int) -> List[Tuple[int, int, int]]:
        assert 1 <= x <= self.B
        msgs: List[Tuple[int, int, int]] = []
        u = self.rng.integers(1, self.q)
        v = self.rng.integers(1, self.q + 1)
        w = ((u * x + v) % self.q) % self.b
        msgs.append((int(u), int(v), int(w)))
        send = self.send_fixed_messages + (1 if self.rng.random() < self.remaining_prob else 0)
        if send > 0:
            uu = self.rng.integers(1, self.q, size=send)
            vv = self.rng.integers(1, self.q + 1, size=send)
            ww = self.rng.integers(0, self.b, size=send)
            msgs.extend([(int(uu[i]), int(vv[i]), int(ww[i])) for i in range(send)])
        return msgs

    def randomize_all(self, values: List[int], shuffle: bool = True) -> List[Tuple[int, int, int]]:
        self.messages = []
        for v in values:
            # Data is already 1-indexed [1, B], use directly (consistent with C++ implementation)
            x = v
            self.messages.extend(self.local_randomizer(x))
        if shuffle:
            self.rng.shuffle(self.messages)
        return self.messages

    def analyzer_single_vectorized(self, query_id: int,
                                   messages_np: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        U, V, W = messages_np
        q = self.q; b = self.b; n = self.n
        pcol = self.collision_prob; rho = self.sample_prob
        cnt = np.sum(((U * query_id + V) % q) % b == W)
        return (cnt - n * rho / b - n * pcol) / (1.0 - pcol)

    def to_numpy_messages(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.array(self.messages, dtype=np.int64)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    def get_theta_fe1(self):
        """
        theta: error bound (can be adjust to different settings)
        """
        b = self.b
        pcol = self.collision_prob
        rho = self.sample_prob

        # expectation of noise
        mu_noise1 = self.n * pcol
        mu_noise2 = (self.n * 2 * math.floor(rho)) * (1 / b)
        mu_noise3 = (self.n * 2) * ((rho - math.floor(rho)) / b)
        mu = mu_noise1 + mu_noise2 + mu_noise3

        term1 = 3 * math.log(2 * self.B / self.beta)
        term2 = math.sqrt(3 * math.log(2 * self.B / self.beta) * mu) / (1 - pcol)

        bias = 0
        theta = max(term1, term2) + bias

        return theta

    def analyzer(
        self,
        messages: List[Tuple[int, int, int]],
        workers: int | None = None,
        progress_logger: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """
        Public Analyzer method: Aggregates messages and uses the speedup function
        to estimate the frequency vector for the current group size (self.n).
        """
        # 1. Handle empty message list
        if not messages:
            return np.zeros(self.B + 1)

        # 2. Convert list of tuples to numpy arrays for the speedup function
        if progress_logger is not None:
            progress_logger(
                f"FE1 analyzer: convert {len(messages):,} messages to numpy"
            )
        arr = np.array(messages, dtype=np.int64)
        U, V, W = arr[:, 0], arr[:, 1], arr[:, 2]
        if progress_logger is not None:
            progress_logger("FE1 analyzer: start aggregate counting")

        # 3. Call the external multi-process analyzer
        # We rely on the FE1 object's internal n, rho, pcol which are set
        # based on the group size/parameters in the HSDP structure.
        freq_vec = analyzers_speedup_mp(
            U, V, W,
            B=self.B, q=self.q, b=self.b,
            n=self.n, rho=self.sample_prob, pcol=self.collision_prob,
            workers=workers,  # Pass through the workers parameter
            progress_logger=progress_logger,
        )

        return freq_vec


# ====================== test code non-speedup vs speedup ======================
def run_once(n=100_000, B=(1 << 22), epsilon=1.0, delta=None, c=1.0,
             seed=1234, num_queries=10_000, workers=4, verbose=True):
    if delta is None:
        delta = 1.0 / (n * n)

    rng = np.random.default_rng(seed)
    values = rng.integers(0, B, size=n, dtype=np.int64)

    fe = FE1Baseline(n=n, B=B, epsilon=epsilon, delta=delta, c=c, beta=0.1, use_mu_search=True, seed=seed)
    t0 = time.time()
    _ = fe.randomize_all(values.tolist(), shuffle=True)
    U, V, W = fe.to_numpy_messages()
    t1 = time.time()

    rho = fe.sample_prob
    k = int(math.floor(rho))
    frac = rho - k
    total = len(U)
    successes = total - n - n * k
    exp_successes = n * frac
    std_successes = math.sqrt(n * frac * (1.0 - frac)) if 0 < frac < 1 else 0.0
    z = (successes - exp_successes) / std_successes if std_successes > 0 else 0.0
    if verbose:
        print(f"[c={c}] rho={rho:.6f}, msgs/user(theory)={1+rho:.6f}, msgs/user(sim)={total/n:.6f}")
        print(f"[c={c}] successes={successes} (exp={exp_successes:.2f}, std={std_successes:.2f}, z={z:.2f})")
        print(f"[c={c}] q={fe.q}, b={fe.b}, pcol={fe.collision_prob:.7g}")
        print(f"[c={c}] randomization time: {t1 - t0:.2f}s, total messages={total}")


    used = set(int(v) + 1 for v in values)
    queries = []
    while len(queries) < num_queries:
        cand = rng.integers(1, B + 1, size=num_queries, dtype=np.int64)
        cand = [int(x) for x in cand if x not in used]
        queries.extend(cand)
    queries = np.array(queries[:num_queries], dtype=np.int64)
    cnt = Counter(int(v) + 1 for v in values)
    g_true = np.array([cnt.get(int(x), 0) for x in queries], dtype=np.int64)

    t2 = time.time()
    g_hat_non = np.array([fe.analyzer_single_vectorized(int(x), (U, V, W)) for x in queries],
                         dtype=np.float64)
    t3 = time.time()
    err_non = np.abs(g_hat_non - g_true)
    non_speedup_sec = t3 - t2
    if verbose:
        print(f"[non-speedup] {len(queries)} queries in {non_speedup_sec:.2f}s, "
              f"q95={np.quantile(err_non,0.95):.2f}, max={err_non.max():.2f}")

    # speedup：全域
    t4 = time.time()
    freq_all = analyzers_speedup_mp(U, V, W, B=fe.B, q=fe.q, b=fe.b,
                                    n=fe.n, rho=fe.sample_prob, pcol=fe.collision_prob,
                                    workers=workers)
    t5 = time.time()
    speedup_sec = t5 - t4

    realvec = np.zeros(B + 1, dtype=np.int64)
    for v in values:
        realvec[v + 1] += 1
    err_all = np.abs(freq_all - realvec)
    if verbose:
        print(f"[speedup • FULL DOMAIN] analyzed {B} ids in {speedup_sec:.2f}s, "
              f"q95={np.quantile(err_all[1:],0.95):.2f}, max={err_all[1:].max():.2f}")
        if non_speedup_sec > 0:
            print(f"[ratio] speedup/non-speedup time = {speedup_sec/non_speedup_sec:.2f}x")

    return {"non_speedup_sec": non_speedup_sec, "speedup_sec": speedup_sec}

if __name__ == "__main__":
    _ = run_once(n=2**17, B=2**17,epsilon=4, c=1.0, seed=1, workers=4)
