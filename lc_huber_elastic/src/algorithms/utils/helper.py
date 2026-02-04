from __future__ import annotations

from typing import List, Tuple


def prox_residual_torch(x, F_x, g, *, tau: float = 1.0, weights=None) -> float:
    """Compute the (Euclidean) proximal residual.

    Residual definition (composite inclusion):

        r(x) = || x - prox_{tau g}(x - tau F(x)) ||.

    For this benchmark suite we use the conventional choice tau=1 unless
    overridden. `weights` is forwarded to g.prox_opr_torch when the proximal
    map supports a diagonal metric.
    """

    import torch

    with torch.no_grad():
        z = x - float(tau) * F_x
        prox = g.prox_opr_torch(z, tau=float(tau), weights=weights)
        return float(torch.linalg.vector_norm(x - prox).item())


def compute_opt_measure(
    kind: str,
    *,
    x,
    F_x,
    g,
    oracle,
    Au,
    r2,
) -> float:
    """Compute the requested opt-measure.

    kind:
      - "prox_residual": proximal residual (tau=1, Euclidean)
      - "projected_primal_gap": h_δ(u_proj) - h_δ(u⋆)
    """

    kind = str(kind or "prox_residual").lower()
    if kind in {"prox_residual", "prox"}:
        return prox_residual_torch(x, F_x, g, tau=1.0, weights=None)

    if kind in {"projected_primal_gap", "projected_gap"}:
        if oracle is None or Au is None or r2 is None:
            raise ValueError("oracle, Au, and r2 are required for projected_primal_gap")
        return float(oracle.projected_huber_gap_from_state(Au, r2))

    raise ValueError(f"Unknown opt-measure kind: {kind}")


def construct_contiguous_slices(begin: int, end: int, block_size: int) -> List[slice]:
    """Construct contiguous slices on [begin,end) with given block size."""

    if block_size <= 0:
        raise ValueError("block_size must be >= 1")
    if end < begin:
        raise ValueError("end must be >= begin")

    blocks: List[slice] = []
    for start in range(begin, end, block_size):
        stop = min(start + block_size, end)
        blocks.append(slice(start, stop))
    return blocks


def construct_uv_block_slices(
    n_u: int,
    n_v: int,
    block_size_u: int,
    block_size_v: int,
) -> Tuple[List[slice], List[str]]:
    """Construct a two-stage block partition for x = (u, v).

    We create contiguous blocks for u (first) and v (second) so that no block
    crosses the u/v boundary. This simplifies block-slice operator evaluation.

    Returns:
        blocks: list of slices into x
        block_types: list of strings ('u' or 'v') for each block
    """

    if n_u <= 0 or n_v <= 0:
        raise ValueError("n_u and n_v must be positive")

    blocks_u = construct_contiguous_slices(0, n_u, block_size_u)
    blocks_v = construct_contiguous_slices(n_u, n_u + n_v, block_size_v)

    blocks = blocks_u + blocks_v
    block_types = ["u"] * len(blocks_u) + ["v"] * len(blocks_v)
    return blocks, block_types
