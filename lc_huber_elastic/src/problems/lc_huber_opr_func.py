"""Operator for the linearly constrained Huber minimization problem (Figure 3).

Figure 3 problem statement:
    minimize_{u \in R^n}   h_δ(u)
    subject to             A u = b

with the (vector) Huber function
        { 0.5 ||u||^2                    if ||u|| <= δ
h_δ(u)= {
        { δ||u|| - 0.5 δ^2               otherwise

The paper runs algorithms on the Lagrangian saddle function
    L(u, v) = h_δ(u) + <A u - b, v>,
where v \in R^m is the dual multiplier.

Define x = (u, v) \in R^{n+m}. The corresponding monotone saddle operator is

    F(x) = ( ∇_u L(u,v),  -∇_v L(u,v) )
         = ( ∇h_δ(u) + A^T v,   b - A u ).

This implementation uses NumPy and is intended to be used by the GMVI / VI
algorithms implemented elsewhere in this repository.
"""

from __future__ import annotations

import numpy as np


class LCHuberOprFunc:
    """Monotone operator for the Huber equality-constrained saddle formulation."""

    def __init__(self, A: np.ndarray, b: np.ndarray, delta: float = 0.1):
        A = np.asarray(A)
        b = np.asarray(b)
        if A.ndim != 2:
            raise ValueError("A must be a 2D array")
        if b.ndim != 1:
            raise ValueError("b must be a 1D array")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"A and b dimension mismatch: A is {A.shape}, b is {b.shape}")
        if delta <= 0:
            raise ValueError("delta must be positive")

        self.A = A
        self.A_T = A.T
        self.b = b
        self.delta = float(delta)

        self.m, self.n_u = A.shape
        self.n_v = self.m
        self.n = int(self.n_u + self.n_v)

    # -------------------------
    # Huber pieces
    # -------------------------
    def huber_value(self, u: np.ndarray) -> float:
        """Return h_δ(u)."""
        r = float(np.linalg.norm(u))
        if r <= self.delta:
            return 0.5 * r * r
        return self.delta * r - 0.5 * self.delta * self.delta

    def huber_grad(self, u: np.ndarray) -> np.ndarray:
        """Return ∇h_δ(u)."""
        r = float(np.linalg.norm(u))
        if r <= self.delta or r == 0.0:
            return np.asarray(u).copy()
        return (self.delta / r) * np.asarray(u)

    # -------------------------
    # Operator interface
    # -------------------------
    def func_value(self, x: np.ndarray) -> float:
        """Primal objective value h_δ(u) (ignores the equality constraint).

        This is mainly provided for logging/diagnostics; algorithms for the
        saddle-point formulation typically monitor the residual instead.
        """
        x = np.asarray(x)
        if x.shape[0] != self.n:
            raise ValueError(f"x must have length {self.n}")
        u = x[: self.n_u]
        return self.huber_value(u)

    def func_map(self, x: np.ndarray) -> np.ndarray:
        """Return the full operator F(x) = (∇h_δ(u)+A^T v,  b - A u)."""
        x = np.asarray(x)
        if x.shape[0] != self.n:
            raise ValueError(f"x must have length {self.n}")
        u = x[: self.n_u]
        v = x[self.n_u :]
        Fu = self.huber_grad(u) + self.A_T @ v
        Fv = self.b - self.A @ u
        return np.concatenate((Fu, Fv), axis=0)

    def func_map_block(self, block: slice, x: np.ndarray) -> np.ndarray:
        """Return F(x)[block].

        This helper is convenient for block-coordinate implementations.
        """
        F = self.func_map(x)
        return F[block]

    def func_map_block_update(
        self,
        F_store: np.ndarray,
        x: np.ndarray,
        x_prev: np.ndarray | None = None,
        block: slice | None = None,
    ) -> np.ndarray:
        """Update cached operator values after a block update.

        For this problem, the Huber gradient depends on the *global* norm ||u||,
        so even a small u-block update can change many components of F_u.

        For correctness and simplicity, we recompute the full operator.

        Args:
            F_store: array to overwrite with the new F(x)
            x: current full iterate (u, v)
            x_prev: unused (kept for API compatibility)
            block: unused (kept for API compatibility)

        Returns:
            Updated F_store.
        """
        np.copyto(F_store, self.func_map(x))
        return F_store

    # -------------------------
    # Lipschitz helpers
    # -------------------------
    def estimate_global_L(self) -> float:
        """A simple global Lipschitz estimate for F under the Euclidean norm.

        Since ∇h_δ is 1-Lipschitz and the linear coupling contributes ||A||,
        a coarse but safe estimate is
            L <= 1 + ||A||_2.

        For tighter estimates you can compute ||[[I, A^T],[-A, 0]]||_2.
        """
        sigma = float(np.linalg.norm(self.A, ord=2))
        return 1.0 + sigma
