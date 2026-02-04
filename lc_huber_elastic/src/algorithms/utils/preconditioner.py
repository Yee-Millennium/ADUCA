"""Diagonal preconditioners (Λ^{-1}) for the LC-Huber saddle operator.

Problem operator
---------------
For x=(u,v) with u∈R^n, v∈R^m,

    F(x) = (∇h_δ(u) + A^T v,  b - A u).

We build a diagonal scaling directly from Euclidean column/row norms of A:

    Λ_u^{-1}[j] = 1 / ||A_{:,j}||_2,
    Λ_v^{-1}[i] = 1 / ||A_{i,:}||_2,

so that the first n_u coordinates (u) use inverse column norms and the next
n_v coordinates (v) use inverse row norms.

This matches the user's requested normalization and avoids any Lipschitz
estimation logic.
"""

from __future__ import annotations

import numpy as np


def normalizers_numpy(A: np.ndarray, mode: str = "diag_lipschitz", eps: float = 1e-12) -> np.ndarray:
    """Return Λ^{-1} as a NumPy vector for x=(u,v).

    Args:
        A: constraint matrix (m,n)
        mode: 'identity' or 'diag_lipschitz'
        eps: lower bound to avoid division by zero
    """

    mode = str(mode).lower()
    m, n = A.shape
    if mode in {"none", "identity"}:
        return np.ones(n + m, dtype=float)

    if mode in {"diag_lipschitz", "diag", "lipschitz"}:
        col_norm = np.linalg.norm(A, axis=0)
        row_norm = np.linalg.norm(A, axis=1)
        col_norm = np.maximum(col_norm, eps)
        row_norm = np.maximum(row_norm, eps)
        return np.concatenate([1.0 / col_norm, 1.0 / row_norm], axis=0)

    raise ValueError(f"Unknown preconditioner mode: {mode}")


def normalizers_torch(
    A,
    *,
    device,
    dtype,
    mode: str = "diag_lipschitz",
    eps: float = 1e-12,
):
    """Return Λ^{-1} as a torch tensor."""

    import torch

    mode = str(mode).lower()
    A_t = torch.as_tensor(A, dtype=dtype, device=device)
    m, n = A_t.shape

    if mode in {"none", "identity"}:
        return torch.ones(n + m, dtype=dtype, device=device)

    if mode in {"diag_lipschitz", "diag", "lipschitz"}:
        col_norm = torch.linalg.vector_norm(A_t, dim=0).clamp_min(float(eps))
        row_norm = torch.linalg.vector_norm(A_t, dim=1).clamp_min(float(eps))
        return torch.cat([1.0 / col_norm, 1.0 / row_norm], dim=0)

    raise ValueError(f"Unknown preconditioner mode: {mode}")


def recip_normalizers_torch(normalizers, *, eps: float = 0.0):
    """Return Λ from Λ^{-1}."""

    import torch

    if eps <= 0:
        return torch.where(normalizers != 0, 1.0 / normalizers, torch.zeros_like(normalizers))
    return torch.where(normalizers.abs() > eps, 1.0 / normalizers, torch.zeros_like(normalizers))
