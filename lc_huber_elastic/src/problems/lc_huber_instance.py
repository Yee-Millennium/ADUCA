"""Instance generation utilities for the linearly constrained Huber problem.

We consider the linearly constrained convex problem

    minimize    h_δ(u)
    subject to  A u = b

where h_δ is the radial Huber loss

    h_δ(u) = 0.5 ||u||^2,                 if ||u|| <= δ
           = δ ||u|| - 0.5 δ^2,           otherwise.

We solve the associated saddle / variational inequality on the Lagrangian
L(u,v) = h_δ(u) + <Au - b, v>, with operator

    F(u,v) = ( ∇h_δ(u) + A^T v ,  b - A u ).

This module provides a small baseline instance (0), three dense ADUCA-favorable
elastic-net scenarios (EN-1..EN-3), and five smaller easy dense variants
(EN-4..EN-8). The dense scenarios are lognormal-heterogeneous Gaussian matrices
with selective l1, designed to favor ADUCA-style block updates.
Each scenario specifies:
  - problem size (n, m)
  - a dense/structured pattern for A
  - sparse u_bar for feasibility and b = A u_bar
  - recommended elastic-net (lambda1, lambda2) and ADUCA block sizes (metadata)

Implementation notes
--------------------
* The generator returns (A, b, u_bar, x0). We set b = A u_bar so the constraint
  is feasible.
* Optional row/column scaling is supported to create heterogeneity.
* make_lc_huber_problem(...) also computes and stores optval_huber for logging:
  since h_δ(u) is increasing in ||u||, the optimum of the constrained problem is
  attained at the minimum-norm feasible solution u⋆ = argmin ||u|| s.t. Au=b.
* For large instances, exact optval_huber and spectral-norm estimation can be
  expensive. make_lc_huber_problem exposes fast auto defaults that skip the
  optval solve and use a cheap ||A|| upper bound when A is very large.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.problems.GMVI_func import GMVIProblem
from src.problems.lc_huber_g_func import ZeroGFunc, ElasticNetGFunc
from src.problems.lc_huber_opr_func import LCHuberOprFunc


@dataclass(frozen=True)
class LCHuberScenario:
    """Container for a reproducible scenario definition."""

    # Dimensions
    n: int = 100
    m: int = 20

    # Huber parameter
    delta: float = 0.1

    # Fraction of nonzeros in u_bar (used only to construct b = A u_bar)
    sparsity: float = 0.1

    # Elastic-net parameters (metadata only; not used unless passed explicitly).
    lambda1: float = 0.0
    lambda2: float = 0.0

    # Recommended block sizes (metadata only).
    block_size_u: Optional[int] = None
    block_size_v: Optional[int] = None
    total_blocks: Optional[int] = None

    # A entry scale:
    # - If None and A_kind is "dense"/"gaussian": use default std = 1/n.
    # - Otherwise, you should set A_std explicitly in the scenario.
    A_std: Optional[float] = None

    # A structure.
    #  - "dense": i.i.d. Gaussian
    #  - "sparse_mask": i.i.d. Gaussian masked by Bernoulli(density)
    #  - "sparse_degree": each column has exactly degree_per_col nonzeros
    #  - "banded": each row touches a contiguous band of width band_width
    #
    # Additional dense structured kinds:
    #  - "block_dominant_dense": dense with strong diagonal blocks and weak dense off-diagonal noise
    #  - "banded_plus_noise_dense": banded rows plus weak dense background noise (still dense overall)
    #  - "clustered_dense": columns clustered around a small number of dense centers (high coherence)
    #  - "lowrank_plus_noise_dense": low-rank matrix with large condition number plus dense noise
    A_kind: str = "dense"

    # Sparsity controls for structured A.
    A_density: float = 1.0
    degree_per_col: Optional[int] = None
    band_width: Optional[int] = None

    # --- Dense structured parameters
    # Block dominant
    num_blocks: Optional[int] = None
    offdiag_strength: float = 0.03  # multiplies A_std for the weak dense background
    diag_block_strength: float = 1.0  # multiplies the diag-block std

    # Banded+noise (dense overall)
    dense_noise_strength: float = 0.02  # multiplies A_std for the dense background
    band_strength: float = 1.0  # multiplies the default band std 1/sqrt(band_width)

    # Clustered columns
    num_clusters: Optional[int] = None
    cluster_assign: str = "contiguous"  # {"contiguous","random"}
    cluster_noise: float = 0.02  # relative noise amplitude inside each cluster (multiplies N(0,1))

    # Low-rank + noise
    low_rank: Optional[int] = None
    low_rank_cond: float = 1e6
    low_rank_noise: float = 0.05  # multiplies A_std for the dense noise background
    low_rank_sigma_max: float = 1.0

    # Optional column/row scaling to create heterogeneous Lipschitz blocks.
    col_scaling: Optional[str] = None  # {"none","lognormal","powerlaw"}
    col_scale_strength: float = 0.0
    row_scaling: Optional[str] = None  # {"none","lognormal"}
    row_scale_strength: float = 0.0

    # Optional: stiff-but-inactive columns (large columns avoided by u_bar)
    stiff_inactive_frac: float = 0.0
    stiff_inactive_scale: float = 1.0


# -----------------------------------------------------------------------------
# Scenario bank
# -----------------------------------------------------------------------------
# 1–3 correspond to EN-1..EN-3 (dense ADUCA-favorable elastic-net scenarios).
# 4–8 correspond to EN-4..EN-8 (small easy dense variants).
# 0 is a small dense baseline that is easy to inspect.
SCENARIOS: Dict[int, LCHuberScenario] = {
    # Instance 0: Small dense baseline
    0: LCHuberScenario(
        n=100,
        m=20,
        delta=0.1,
        sparsity=0.1,
        A_kind="dense",
        lambda1=1e-3,
        lambda2=1e-2,
        block_size_u=5,
        block_size_v=2,
        total_blocks=30,
    ),

    # EN-1: Dense + extreme heterogeneity at larger scale
    1: LCHuberScenario(
        n=12000,
        m=1200,
        delta=0.05,
        sparsity=0.01,
        A_std=1.0 / np.sqrt(1200.0),
        A_kind="dense",
        col_scaling="lognormal",
        col_scale_strength=3.0,
        lambda1=4e-2,
        lambda2=1e-5,
        block_size_u=240,
        block_size_v=60,
        total_blocks=80,
    ),

    # EN-2: Easy medium-scale dense + extreme heterogeneity
    2: LCHuberScenario(
        n=2400,
        m=240,
        delta=0.05,
        sparsity=0.01,
        A_std=1.0 / np.sqrt(240.0),
        A_kind="dense",
        col_scaling="lognormal",
        col_scale_strength=3.0,
        lambda1=4e-2,
        lambda2=1e-5,
        block_size_u=60,
        block_size_v=24,
        total_blocks=50,
    ),

    # EN-3: Easy dense + heterogeneity with mild row scaling
    3: LCHuberScenario(
        n=3000,
        m=300,
        delta=0.05,
        sparsity=0.015,
        A_std=1.0 / np.sqrt(300.0),
        A_kind="dense",
        col_scaling="lognormal",
        col_scale_strength=2.6,
        row_scaling="lognormal",
        row_scale_strength=0.4,
        lambda1=3e-2,
        lambda2=1e-5,
        block_size_u=75,
        block_size_v=30,
        total_blocks=50,
    ),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _make_scaling_vector(rng: np.random.Generator, n: int, kind: Optional[str], strength: float) -> np.ndarray:
    """Return a positive scaling vector of length n."""
    kind = (kind or "none").lower()
    if strength <= 0.0 or kind in ("none",):
        return np.ones(n)

    if kind == "lognormal":
        # log s ~ N(0, strength^2)
        return np.exp(strength * rng.standard_normal(n))

    if kind == "powerlaw":
        # Heavy-tailed positive scales. We normalize by mean to avoid exploding norms.
        alpha = 1.0 / max(strength, 1e-12) + 1.0
        u = rng.uniform(0.0, 1.0, size=n)
        s = np.power(np.maximum(u, 1e-12), -1.0 / alpha)
        return s / np.mean(s)

    raise ValueError(f"Unknown scaling kind: {kind}")


def _partition_indices(n: int, num_parts: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays (starts, ends) partitioning [0,n) into num_parts contiguous blocks."""
    num_parts = int(max(1, num_parts))
    bounds = np.linspace(0, n, num_parts + 1).astype(int)
    starts = bounds[:-1]
    ends = bounds[1:]
    return starts, ends


def _generate_A_dense(rng: np.random.Generator, m: int, n: int, std: float) -> np.ndarray:
    return rng.normal(loc=0.0, scale=std, size=(m, n))


def _generate_A_sparse_mask(
    rng: np.random.Generator, m: int, n: int, std: float, density: float
) -> np.ndarray:
    if not (0.0 < density <= 1.0):
        raise ValueError("A_density must be in (0,1] for sparse_mask")
    A = rng.normal(loc=0.0, scale=std, size=(m, n))
    mask = rng.uniform(0.0, 1.0, size=(m, n)) < float(density)
    A *= mask

    # Ensure no completely-zero rows/columns (helps numerics).
    row_nnz = np.count_nonzero(A, axis=1)
    empty_rows = np.where(row_nnz == 0)[0]
    if empty_rows.size > 0:
        cols = rng.integers(0, n, size=empty_rows.size)
        A[empty_rows, cols] = rng.normal(loc=0.0, scale=std, size=empty_rows.size)

    col_nnz = np.count_nonzero(A, axis=0)
    empty_cols = np.where(col_nnz == 0)[0]
    if empty_cols.size > 0:
        rows = rng.integers(0, m, size=empty_cols.size)
        A[rows, empty_cols] = rng.normal(loc=0.0, scale=std, size=empty_cols.size)
    return A


def _generate_A_sparse_degree(
    rng: np.random.Generator, m: int, n: int, std: float, degree_per_col: int
) -> np.ndarray:
    d = int(max(1, degree_per_col))
    A = np.zeros((m, n), dtype=float)
    for j in range(n):
        rows = rng.choice(m, size=min(d, m), replace=False)
        A[rows, j] = rng.normal(loc=0.0, scale=std, size=rows.size)

    # Ensure every row has at least one nonzero.
    row_nnz = np.count_nonzero(A, axis=1)
    empty_rows = np.where(row_nnz == 0)[0]
    for i in empty_rows:
        j = int(rng.integers(0, n))
        A[i, j] = rng.normal(loc=0.0, scale=std)
    return A


def _generate_A_banded(rng: np.random.Generator, m: int, n: int, std: float, band_width: int) -> np.ndarray:
    w = int(max(1, band_width))
    A = np.zeros((m, n), dtype=float)
    if m == 1:
        centers = [n // 2]
    else:
        centers = [int(round(i * (n - 1) / (m - 1))) for i in range(m)]

    half = w // 2
    for i, c in enumerate(centers):
        j0 = max(0, c - half)
        j1 = min(n, j0 + w)
        j0 = max(0, j1 - w)
        cols = np.arange(j0, j1)
        A[i, cols] = rng.normal(loc=0.0, scale=std, size=cols.size)
    return A


def _generate_A_block_dominant_dense(
    rng: np.random.Generator,
    *,
    m: int,
    n: int,
    base_std: float,
    num_blocks: int,
    offdiag_strength: float,
    diag_block_strength: float,
) -> np.ndarray:
    """Dense matrix with strong diagonal blocks and weak dense off-diagonal noise.

    Construction:
      A = N(0, offdiag_strength * base_std) dense background
      For each block b: add N(0, diag_std_b) on the diagonal block (rows b, cols b),
      where diag_std_b = diag_block_strength / sqrt(col_block_size_b).

    Off-diagonal noise uses offdiag_strength * base_std; diagonal blocks use 1/sqrt(block_n).
    """
    nb = int(max(1, num_blocks))
    # Dense background (off-diagonal and also fills diagonal initially)
    A = rng.normal(loc=0.0, scale=float(offdiag_strength) * float(base_std), size=(m, n))

    r_starts, r_ends = _partition_indices(m, nb)
    c_starts, c_ends = _partition_indices(n, nb)

    for b in range(nb):
        r0, r1 = int(r_starts[b]), int(r_ends[b])
        c0, c1 = int(c_starts[b]), int(c_ends[b])
        block_n = max(1, c1 - c0)
        diag_std = float(diag_block_strength) * (1.0 / np.sqrt(float(block_n)))
        A[r0:r1, c0:c1] += rng.normal(loc=0.0, scale=diag_std, size=(r1 - r0, c1 - c0))

    return A


def _generate_A_banded_plus_noise_dense(
    rng: np.random.Generator,
    *,
    m: int,
    n: int,
    base_std: float,
    band_width: int,
    dense_noise_strength: float,
    band_strength: float,
) -> np.ndarray:
    """Dense matrix with a strong banded component plus weak dense noise background.

    A = dense_noise + band, where
      dense_noise_ij ~ N(0, dense_noise_strength * base_std)
      band entries add N(0, band_strength / sqrt(band_width)) on a contiguous window per row.

    With base_std=1/sqrt(n), dense noise std is ε/sqrt(n), matching Scenario B.
    """
    w = int(max(1, band_width))
    A = rng.normal(loc=0.0, scale=float(dense_noise_strength) * float(base_std), size=(m, n))

    if m == 1:
        centers = [n // 2]
    else:
        centers = [int(round(i * (n - 1) / (m - 1))) for i in range(m)]

    half = w // 2
    band_std = float(band_strength) * (1.0 / np.sqrt(float(w)))
    for i, c in enumerate(centers):
        j0 = max(0, c - half)
        j1 = min(n, j0 + w)
        j0 = max(0, j1 - w)
        cols = np.arange(j0, j1)
        A[i, cols] += rng.normal(loc=0.0, scale=band_std, size=cols.size)

    return A


def _generate_A_clustered_dense(
    rng: np.random.Generator,
    *,
    m: int,
    n: int,
    base_std: float,
    num_clusters: int,
    cluster_assign: str,
    cluster_noise: float,
) -> np.ndarray:
    """Dense column-clustered matrix with high coherence.

    Column model:
        A[:,j] = base_std * (g_{c(j)} + η ξ_j),
    where g_c, ξ_j ~ N(0, I_m). With base_std = 1/sqrt(m) this matches Scenario C.
    """
    K = int(max(1, num_clusters))
    assign_mode = (cluster_assign or "contiguous").lower()
    # Cluster centers (dense)
    G = rng.normal(loc=0.0, scale=1.0, size=(K, m))

    if assign_mode == "random":
        cluster_ids = rng.integers(0, K, size=n)
    else:
        # contiguous assignment
        # cluster size approx n/K
        cluster_ids = (np.arange(n) * K) // n
        cluster_ids = np.clip(cluster_ids, 0, K - 1)

    # Build A column-wise in a vectorized way.
    # centers: (n,m) then transpose to (m,n) to match A shape.
    centers = G[cluster_ids, :]  # (n,m)
    noise = rng.normal(loc=0.0, scale=1.0, size=(n, m))
    A = float(base_std) * (centers + float(cluster_noise) * noise)
    return A.T  # (m,n)


def _generate_A_lowrank_plus_noise_dense(
    rng: np.random.Generator,
    *,
    m: int,
    n: int,
    base_std: float,
    rank: int,
    cond: float,
    noise_strength: float,
    sigma_max: float,
) -> np.ndarray:
    """Dense low-rank + noise matrix with controlled condition number.

    A = U diag(sigma) V^T + noise_strength * N(0, base_std)
    where U∈R^{m×r}, V∈R^{n×r} have orthonormal columns and
    sigma spans [sigma_max, sigma_max/cond] geometrically.
    """
    r = int(max(1, min(rank, m, n)))
    cond = float(max(cond, 1.0))

    U0 = rng.normal(size=(m, r))
    V0 = rng.normal(size=(n, r))
    U, _ = np.linalg.qr(U0, mode="reduced")
    V, _ = np.linalg.qr(V0, mode="reduced")

    sigma = float(sigma_max) * np.geomspace(1.0, 1.0 / cond, num=r)
    # (U * sigma) @ V^T is U diag(sigma) V^T
    A_lr = (U * sigma.reshape(1, -1)) @ V.T

    noise = rng.normal(loc=0.0, scale=float(base_std), size=(m, n))
    A = A_lr + float(noise_strength) * noise
    return A


def _generate_A(
    rng: np.random.Generator,
    *,
    m: int,
    n: int,
    std: float,
    A_kind: str,
    A_density: float,
    degree_per_col: Optional[int],
    band_width: Optional[int],
    # dense structured
    num_blocks: Optional[int],
    offdiag_strength: float,
    diag_block_strength: float,
    dense_noise_strength: float,
    band_strength: float,
    num_clusters: Optional[int],
    cluster_assign: str,
    cluster_noise: float,
    low_rank: Optional[int],
    low_rank_cond: float,
    low_rank_noise: float,
    low_rank_sigma_max: float,
) -> np.ndarray:
    """Generate an (m,n) matrix A according to the chosen structure."""
    kind = str(A_kind or "dense").lower()

    if kind in {"dense", "gaussian"}:
        return _generate_A_dense(rng, m, n, std)

    if kind in {"sparse_mask", "mask"}:
        return _generate_A_sparse_mask(rng, m, n, std, float(A_density))

    if kind in {"sparse_degree", "degree"}:
        d = degree_per_col
        if d is None:
            d = max(1, int(round(float(A_density) * m)))
        return _generate_A_sparse_degree(rng, m, n, std, int(d))

    if kind in {"banded", "local"}:
        w = band_width
        if w is None:
            w = max(1, int(round(float(A_density) * n)))
        return _generate_A_banded(rng, m, n, std, int(w))

    # --- Dense structured kinds
    if kind in {"block_dominant_dense", "block_dominant"}:
        nb = num_blocks if num_blocks is not None else 10
        return _generate_A_block_dominant_dense(
            rng,
            m=m,
            n=n,
            base_std=std,
            num_blocks=int(nb),
            offdiag_strength=float(offdiag_strength),
            diag_block_strength=float(diag_block_strength),
        )

    if kind in {"banded_plus_noise_dense", "banded_noise_dense", "banded_plus_noise"}:
        w = int(band_width if band_width is not None else max(1, int(round(0.02 * n))))
        return _generate_A_banded_plus_noise_dense(
            rng,
            m=m,
            n=n,
            base_std=std,
            band_width=w,
            dense_noise_strength=float(dense_noise_strength),
            band_strength=float(band_strength),
        )

    if kind in {"clustered_dense", "clustered"}:
        K = int(num_clusters if num_clusters is not None else 20)
        return _generate_A_clustered_dense(
            rng,
            m=m,
            n=n,
            base_std=std,
            num_clusters=K,
            cluster_assign=str(cluster_assign or "contiguous"),
            cluster_noise=float(cluster_noise),
        )

    if kind in {"lowrank_plus_noise_dense", "lowrank_dense", "lowrank_plus_noise"}:
        r = int(low_rank if low_rank is not None else min(m, n, 50))
        return _generate_A_lowrank_plus_noise_dense(
            rng,
            m=m,
            n=n,
            base_std=std,
            rank=r,
            cond=float(low_rank_cond),
            noise_strength=float(low_rank_noise),
            sigma_max=float(low_rank_sigma_max),
        )

    raise ValueError(f"Unknown A_kind: {A_kind}")


def _huber_value_from_r2(r2: float, delta: float) -> float:
    r2 = max(float(r2), 0.0)
    r = float(np.sqrt(r2))
    if r <= delta:
        return 0.5 * r2
    return delta * r - 0.5 * delta * delta


def _min_norm_opt_r2(A: np.ndarray, b: np.ndarray, *, jitter0: float = 0.0) -> Tuple[float, float]:
    """Return (||u⋆||², jitter) for u⋆ = argmin ||u|| s.t. Au=b.

    With full row rank, u⋆ = A^T (A A^T)^{-1} b and ||u⋆||² = b^T (A A^T)^{-1} b.

    Numerically, we solve (AAT + jitter I) y = b and estimate
      ||u⋆||² ≈ y^T b - jitter ||y||²
    (exact when AAT y = b - jitter y).
    """
    m = A.shape[0]
    AAT = A @ A.T

    jitter = float(jitter0)
    if jitter <= 0.0:
        diag_mean = float(np.mean(np.abs(np.diag(AAT)))) if m > 0 else 1.0
        jitter = 1e-12 * max(1.0, diag_mean)

    for _ in range(10):
        try:
            y = np.linalg.solve(AAT + jitter * np.eye(m), b)
            r2 = float(y @ b - jitter * (y @ y))
            return max(r2, 0.0), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0

    # Robust fallback (can be expensive): pseudo-inverse of AAT.
    y = np.linalg.pinv(AAT) @ b
    r2 = float(y @ (AAT @ y))
    return max(r2, 0.0), float("nan")


def _estimate_global_L(
    A: np.ndarray,
    *,
    method: str = "auto",
    size_threshold: int = 5_000_000,
) -> float:
    """Estimate 1 + ||A||_2 with a cheap upper bound for large A.

    method:
      - "auto": spectral norm for small A, 1/inf upper bound for large A
      - "spectral": exact (SVD-based) spectral norm
      - "one_inf": sqrt(||A||_1 * ||A||_inf) upper bound
      - "fro": Frobenius norm upper bound
    """
    method = str(method or "auto").lower()
    if method == "auto":
        method = "spectral" if A.size <= int(size_threshold) else "one_inf"

    if method in {"spectral", "svd", "exact"}:
        sigma = float(np.linalg.norm(A, ord=2))
    elif method in {"one_inf", "oneinf", "bound"}:
        norm1 = float(np.linalg.norm(A, ord=1))
        norminf = float(np.linalg.norm(A, ord=np.inf))
        sigma = float(np.sqrt(norm1 * norminf))
    elif method in {"fro", "frobenius"}:
        sigma = float(np.linalg.norm(A, ord="fro"))
    else:
        raise ValueError(f"Unknown lipschitz method: {method}")

    return 1.0 + sigma


def generate_lc_huber_data(
    *,
    n: int = 100,
    m: int = 20,
    delta: float = 0.1,
    sparsity: float = 0.1,
    seed: int = 0,
    A_std: Optional[float] = None,
    A_kind: str = "dense",
    A_density: float = 1.0,
    degree_per_col: Optional[int] = None,
    band_width: Optional[int] = None,
    # dense structured:
    num_blocks: Optional[int] = None,
    offdiag_strength: float = 0.03,
    diag_block_strength: float = 1.0,
    dense_noise_strength: float = 0.02,
    band_strength: float = 1.0,
    num_clusters: Optional[int] = None,
    cluster_assign: str = "contiguous",
    cluster_noise: float = 0.02,
    low_rank: Optional[int] = None,
    low_rank_cond: float = 1e6,
    low_rank_noise: float = 0.05,
    low_rank_sigma_max: float = 1.0,
    # scaling:
    col_scaling_kind: Optional[str] = None,
    col_scale_strength: float = 0.0,
    row_scaling_kind: Optional[str] = None,
    row_scale_strength: float = 0.0,
    # stiff-but-inactive columns
    stiff_inactive_frac: float = 0.0,
    stiff_inactive_scale: float = 1.0,
    # optional torch-based dense generation
    generate_device: Optional[str] = None,
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate (A, b, u_bar, x0) for the LC-Huber problem.

    If generate_device is set and A_kind is dense/gaussian, we try to draw A via
    torch on that device, then move it back to NumPy for downstream code.
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive")
    if not (0.0 < sparsity <= 1.0):
        raise ValueError("sparsity must be in (0, 1]")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if not (0.0 <= float(stiff_inactive_frac) < 1.0):
        raise ValueError("stiff_inactive_frac must be in [0,1)")
    if float(stiff_inactive_scale) <= 0.0:
        raise ValueError("stiff_inactive_scale must be positive")

    rng = np.random.default_rng(seed)

    # Default std uses 1/n scaling when A_std is None.
    std = (1.0 / n) if A_std is None else float(A_std)

    A = None
    A_backend = "numpy"
    A_device_used = "cpu"
    kind = str(A_kind or "dense").lower()
    if generate_device is not None and kind in {"dense", "gaussian"}:
        try:
            import torch

            device = torch.device(str(generate_device))
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            A_torch = torch.randn((m, n), generator=gen, device=device, dtype=torch.float64) * float(std)
            A = A_torch.cpu().numpy()
            A_backend = "torch"
            A_device_used = str(device)
        except Exception:
            A = None
            A_backend = "numpy_fallback"
            A_device_used = "cpu"

    if A is None:
        A = _generate_A(
            rng,
            m=m,
            n=n,
            std=std,
            A_kind=A_kind,
            A_density=A_density,
            degree_per_col=degree_per_col,
            band_width=band_width,
            num_blocks=num_blocks,
            offdiag_strength=offdiag_strength,
            diag_block_strength=diag_block_strength,
            dense_noise_strength=dense_noise_strength,
            band_strength=band_strength,
            num_clusters=num_clusters,
            cluster_assign=cluster_assign,
            cluster_noise=cluster_noise,
            low_rank=low_rank,
            low_rank_cond=low_rank_cond,
            low_rank_noise=low_rank_noise,
            low_rank_sigma_max=low_rank_sigma_max,
        )

    # Optional heterogeneity: scale columns/rows.
    col_scale = _make_scaling_vector(rng, n, col_scaling_kind, col_scale_strength)
    row_scale = _make_scaling_vector(rng, m, row_scaling_kind, row_scale_strength)
    A = (row_scale[:, None] * A) * col_scale[None, :]

    # Optional: stiff-but-inactive columns (large columns avoided by u_bar)
    stiff_cols = np.array([], dtype=int)
    if float(stiff_inactive_frac) > 0.0 and float(stiff_inactive_scale) != 1.0:
        k_stiff = int(round(float(stiff_inactive_frac) * float(n)))
        k_stiff = max(1, min(k_stiff, n))
        stiff_cols = rng.choice(n, size=k_stiff, replace=False)
        A[:, stiff_cols] *= float(stiff_inactive_scale)

    # Build sparse u_bar.
    k = max(1, int(round(float(sparsity) * n)))
    if stiff_cols.size == 0:
        available = np.arange(n)
    else:
        mask = np.ones(n, dtype=bool)
        mask[stiff_cols] = False
        available = np.nonzero(mask)[0]
    if available.size == 0:
        raise ValueError("stiff_inactive_frac is too large; no columns left for u_bar support")
    k = min(k, int(available.size))
    support = rng.choice(available, size=k, replace=False)
    u_bar = np.zeros(n)
    u_bar[support] = rng.uniform(low=0.0, high=1.0, size=k)

    b = A @ u_bar

    u0 = rng.standard_normal(n)
    v0 = rng.standard_normal(m)
    x0 = np.concatenate((u0, v0), axis=0)

    if return_meta:
        return A, b, u_bar, x0, {"A_backend": A_backend, "A_generate_device": A_device_used}
    return A, b, u_bar, x0


def make_lc_huber_problem(
    scenario: int = 0,
    *,
    seed: int = 0,
    override: Optional[dict] = None,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    lipschitz_method: str = "auto",
    lipschitz_size_threshold: int = 5_000_000,
    compute_optval: str | bool = "auto",
    optval_size_threshold: int = 1_000_000,
    optval_device: Optional[str] = None,
    generate_device: Optional[str] = None,
) -> Tuple[GMVIProblem, np.ndarray, dict]:
    """Create a GMVIProblem instance and initial point x0."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario id {scenario}. Available: {sorted(SCENARIOS.keys())}")

    s = SCENARIOS[scenario]
    params = {
        "n": s.n,
        "m": s.m,
        "delta": s.delta,
        "sparsity": s.sparsity,
        "A_std": s.A_std,
        "A_kind": s.A_kind,
        "A_density": s.A_density,
        "degree_per_col": s.degree_per_col,
        "band_width": s.band_width,
        "num_blocks": s.num_blocks,
        "offdiag_strength": s.offdiag_strength,
        "diag_block_strength": s.diag_block_strength,
        "dense_noise_strength": s.dense_noise_strength,
        "band_strength": s.band_strength,
        "num_clusters": s.num_clusters,
        "cluster_assign": s.cluster_assign,
        "cluster_noise": s.cluster_noise,
        "low_rank": s.low_rank,
        "low_rank_cond": s.low_rank_cond,
        "low_rank_noise": s.low_rank_noise,
        "low_rank_sigma_max": s.low_rank_sigma_max,
        "col_scaling_kind": s.col_scaling,
        "col_scale_strength": s.col_scale_strength,
        "row_scaling_kind": s.row_scaling,
        "row_scale_strength": s.row_scale_strength,
        "stiff_inactive_frac": s.stiff_inactive_frac,
        "stiff_inactive_scale": s.stiff_inactive_scale,
    }
    if override:
        params.update(override)

    A, b, u_bar, x0, gen_meta = generate_lc_huber_data(
        seed=seed,
        generate_device=generate_device,
        return_meta=True,
        **params,
    )

    operator = LCHuberOprFunc(A=A, b=b, delta=float(params["delta"]))
    g = ElasticNetGFunc(n_u=operator.n_u, n_v=operator.n_v, lambda1=lambda1, lambda2=lambda2)
    problem = GMVIProblem(operator_func=operator, g_func=g)

    # Compute a Lipschitz estimate. For large A, use a cheap upper bound.
    L_est = _estimate_global_L(A, method=lipschitz_method, size_threshold=lipschitz_size_threshold)

    # Compute the optimal primal value for logging: min ||u|| s.t. Au=b.
    compute_optval_flag = compute_optval
    if isinstance(compute_optval_flag, str):
        lowered = compute_optval_flag.lower()
        if lowered == "auto":
            compute_optval_flag = A.size <= int(optval_size_threshold)
        elif lowered in {"true", "1", "yes", "y"}:
            compute_optval_flag = True
        elif lowered in {"false", "0", "no", "n"}:
            compute_optval_flag = False
        else:
            raise ValueError("compute_optval must be bool or one of: 'auto', 'true', 'false'")
    if compute_optval_flag:
        optval_backend = "numpy"
        optval_device_used = "cpu"
        if optval_device is not None and str(optval_device).lower() not in ("cpu", "none"):
            try:
                from src.problems.lc_huber_torch_oracle import LCHuberTorchOracle

                oracle = LCHuberTorchOracle(A, b, delta=float(params["delta"]), device=optval_device)
                opt_r2 = float(oracle.opt_r2)
                optval_huber = float(oracle.opt_primal_value)
                opt_jitter = float(oracle.optval_jitter)
                optval_backend = "torch"
                optval_device_used = str(optval_device)
            except Exception:
                opt_r2, opt_jitter = _min_norm_opt_r2(A, b)
                optval_huber = _huber_value_from_r2(opt_r2, float(params["delta"]))
                optval_backend = "numpy_fallback"
                optval_device_used = "cpu"
        else:
            opt_r2, opt_jitter = _min_norm_opt_r2(A, b)
            optval_huber = _huber_value_from_r2(opt_r2, float(params["delta"]))
    else:
        opt_r2 = float("nan")
        opt_jitter = float("nan")
        optval_huber = float("nan")
        optval_backend = "skipped"
        optval_device_used = "cpu"

    info = {
        "scenario": int(scenario),
        "seed": int(seed),
        "n": int(params["n"]),
        "m": int(params["m"]),
        "delta": float(params["delta"]),
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "sparsity": float(params["sparsity"]),
        "A_kind": str(params.get("A_kind", "dense")),
        "A_density": float(params.get("A_density", 1.0) or 1.0),
        "degree_per_col": None if params.get("degree_per_col", None) is None else int(params["degree_per_col"]),
        "band_width": None if params.get("band_width", None) is None else int(params["band_width"]),
        "num_blocks": None if params.get("num_blocks", None) is None else int(params["num_blocks"]),
        "offdiag_strength": float(params.get("offdiag_strength", 0.0) or 0.0),
        "diag_block_strength": float(params.get("diag_block_strength", 0.0) or 0.0),
        "dense_noise_strength": float(params.get("dense_noise_strength", 0.0) or 0.0),
        "band_strength": float(params.get("band_strength", 0.0) or 0.0),
        "num_clusters": None if params.get("num_clusters", None) is None else int(params["num_clusters"]),
        "cluster_assign": params.get("cluster_assign", None),
        "cluster_noise": float(params.get("cluster_noise", 0.0) or 0.0),
        "low_rank": None if params.get("low_rank", None) is None else int(params["low_rank"]),
        "low_rank_cond": float(params.get("low_rank_cond", 0.0) or 0.0),
        "low_rank_noise": float(params.get("low_rank_noise", 0.0) or 0.0),
        "low_rank_sigma_max": float(params.get("low_rank_sigma_max", 0.0) or 0.0),
        "A_std": float(1.0 / params["n"]) if params["A_std"] is None else float(params["A_std"]),
        "col_scaling_kind": params.get("col_scaling_kind", None),
        "col_scale_strength": float(params.get("col_scale_strength", 0.0) or 0.0),
        "row_scaling_kind": params.get("row_scaling_kind", None),
        "row_scale_strength": float(params.get("row_scale_strength", 0.0) or 0.0),
        "stiff_inactive_frac": float(params.get("stiff_inactive_frac", 0.0) or 0.0),
        "stiff_inactive_scale": float(params.get("stiff_inactive_scale", 1.0) or 1.0),
        "stiff_inactive_count": int(
            max(1, min(int(round(float(params.get("stiff_inactive_frac", 0.0) or 0.0) * params["n"])), params["n"]))
        )
        if float(params.get("stiff_inactive_frac", 0.0) or 0.0) > 0.0
        and float(params.get("stiff_inactive_scale", 1.0) or 1.0) != 1.0
        else 0,
        "opt_r2": float(opt_r2),
        "optval_huber": float(optval_huber),
        "optval_jitter": float(opt_jitter) if np.isfinite(opt_jitter) else None,
        # Keep raw arrays for debugging; runners only serialize a small instance summary.
        "A": A,
        "b": b,
        "u_bar": u_bar,
        "x0": x0,
        "L_est": float(L_est),
        "optval_computed": bool(compute_optval_flag),
        "optval_backend": str(optval_backend),
        "optval_device": str(optval_device_used),
        "A_backend": gen_meta.get("A_backend", None) if isinstance(gen_meta, dict) else None,
        "A_generate_device": gen_meta.get("A_generate_device", None) if isinstance(gen_meta, dict) else None,
    }

    return problem, x0, info
