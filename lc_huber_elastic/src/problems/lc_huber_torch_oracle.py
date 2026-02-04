"""Torch oracle for the LC-Huber linearly constrained problem.

We work with x = (u, v) where u∈R^n, v∈R^m and the monotone operator is

    F(x) = (∇h_δ(u) + A^T v,  b - A u).

The Huber gradient for the vector-norm Huber is

    ∇h_δ(u) = u                        if ||u|| ≤ δ,
             δ u / ||u||              otherwise.

For ADUCA's cyclic block updates we want fast *block* evaluations F(x)[sl] at a
partially updated x. This is achieved by maintaining the state

    Au  = A u,
    ATv = A^T v,
    r2  = ||u||²,

and updating these quantities incrementally when blocks of u or v are changed.

Objective / opt-measure
-----------------------
In the LC-Huber benchmark we ultimately care about the primal objective

    minimize_u  h_δ(u)   subject to  A u = b.

Since iterates produced by saddle algorithms are typically infeasible during the
run, a meaningful primal objective diagnostic is to evaluate h_δ on the
*orthogonal projection* of u onto the affine set {u: Au=b}:

    u_proj = argmin_{Au=b} ||u' - u||².

This oracle provides efficient utilities to compute

    gap(u) = h_δ(u_proj) - h_δ(u⋆),

where u⋆ is the (unique) minimum-Euclidean-norm feasible solution. Because h_δ is
strictly increasing in ||u||, u⋆ is also the minimizer of h_δ over {Au=b}.

The projection and u⋆ can be expressed using only m×m linear solves with
AAT := A A^T. We precompute a (possibly jittered) Cholesky factorization of AAT
for repeated solves.
"""

from __future__ import annotations

from typing import Tuple

import torch


class LCHuberTorchOracle:
    """GPU/CPU oracle with incremental state maintenance and primal diagnostics."""

    def __init__(self, A, b, *, delta: float = 0.1, device=None, dtype=None):
        self.device = torch.device("cpu") if device is None else torch.device(str(device))
        self.dtype = torch.float64 if dtype is None else dtype

        self.A = torch.as_tensor(A, dtype=self.dtype, device=self.device)
        self.b = torch.as_tensor(b, dtype=self.dtype, device=self.device)
        self.delta = float(delta)

        if self.A.ndim != 2:
            raise ValueError("A must be 2D")
        if self.b.ndim != 1:
            raise ValueError("b must be 1D")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(f"A and b mismatch: A is {tuple(self.A.shape)}, b is {tuple(self.b.shape)}")
        if self.delta <= 0:
            raise ValueError("delta must be positive")

        self.m, self.n_u = self.A.shape
        self.n_v = self.m
        self.n = int(self.n_u + self.n_v)

        # Precompute AAT factorization for objective diagnostics.
        self._aat_chol, self._aat_jitter = self._factorize_aat()
        self._opt_r2, self._opt_obj = self._compute_opt_primal_obj()

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------
    def split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 1 or x.numel() != self.n:
            raise ValueError(f"x must be a 1D tensor of length {self.n}")
        u = x[: self.n_u]
        v = x[self.n_u :]
        return u, v

    @torch.no_grad()
    def compute_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (Au, ATv, r2) for the current x."""

        u, v = self.split(x)
        r2 = torch.dot(u, u)
        Au = self.A @ u
        ATv = self.A.T @ v
        return Au, ATv, r2

    @torch.no_grad()
    def huber_scale(self, r2: torch.Tensor) -> torch.Tensor:
        """Return scalar scale s where ∇h(u) = s*u.

        s = 1 if ||u|| ≤ δ, else δ/||u||.
        """

        r = torch.sqrt(torch.clamp(r2, min=0.0))
        delta = torch.as_tensor(self.delta, dtype=r.dtype, device=r.device)
        # Avoid division by zero even though r==0 implies we are in the quadratic region.
        inv_r = 1.0 / (r + 1e-24)
        return torch.where(r <= delta, torch.ones_like(r), delta * inv_r)

    # ---------------------------------------------------------------------
    # Huber objective helpers
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def huber_value_from_r2(self, r2: torch.Tensor) -> torch.Tensor:
        """Return h_δ(u) given r2 = ||u||²."""

        r2 = torch.clamp(r2, min=0.0)
        r = torch.sqrt(r2)
        delta = torch.as_tensor(self.delta, dtype=r.dtype, device=r.device)
        val_quad = 0.5 * r2
        val_lin = delta * r - 0.5 * delta * delta
        return torch.where(r <= delta, val_quad, val_lin)

    # ---------------------------------------------------------------------
    # AAT factorization / solves
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _factorize_aat(self) -> Tuple[torch.Tensor, float]:
        """Return (chol(AAT + jitter I), jitter).

        We expect A to have full row rank in the synthetic generators, but this
        routine is defensive and will add a small diagonal jitter if needed.
        """

        AAT = self.A @ self.A.T
        eye = torch.eye(self.m, dtype=self.dtype, device=self.device)

        # dtype-dependent base jitter.
        base_eps = 1e-12 if self.dtype == torch.float64 else 1e-6

        # Scale jitter by a characteristic magnitude of AAT.
        diag_mean = float(torch.mean(torch.diag(AAT)).abs().item()) if self.m > 0 else 1.0
        jitter = float(base_eps * max(1.0, diag_mean))

        # Try Cholesky with progressively larger jitter.
        for _ in range(8):
            try:
                chol = torch.linalg.cholesky(AAT + jitter * eye)
                return chol, jitter
            except RuntimeError:
                jitter *= 10.0

        # If this happens, the instance is numerically problematic.
        raise RuntimeError("Failed to Cholesky-factorize AAT even with jitter.")

    @torch.no_grad()
    def solve_aat(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve (AAT + jitter I) y = rhs using the cached Cholesky factor."""

        if rhs.ndim != 1 or rhs.numel() != self.m:
            raise ValueError(f"rhs must have shape ({self.m},)")
        y = torch.cholesky_solve(rhs.unsqueeze(1), self._aat_chol).squeeze(1)
        return y

    # ---------------------------------------------------------------------
    # Primal objective diagnostics
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _compute_opt_primal_obj(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (||u⋆||², h_δ(u⋆)) for the minimum-norm feasible solution."""

        y = self.solve_aat(self.b)
        # If jitter>0, AAT y = b - jitter y.
        r2 = torch.dot(y, self.b) - float(self._aat_jitter) * torch.dot(y, y)
        r2 = torch.clamp(r2, min=0.0)
        obj = self.huber_value_from_r2(r2)
        return r2, obj

    @property
    def opt_primal_value(self) -> float:
        return float(self._opt_obj.item())

    @property
    def opt_r2(self) -> float:
        return float(self._opt_r2.item())

    @property
    def optval_jitter(self) -> float:
        return float(self._aat_jitter)

    @torch.no_grad()
    def projected_r2_from_state(self, Au: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        """Return ||u_proj||² given cached (Au, r2).

        u_proj is the orthogonal projection of u onto {u: Au=b}.

        Implementation notes
        --------------------
        The projection has the form u_proj = u - A^T y where
            (AAT) y = (Au - b).

        We avoid forming u_proj explicitly (which would cost an n-dimensional
        matmul). Instead we use the identity

            ||u - A^T y||² = ||u||² - 2 (Au)^T y + y^T (AAT) y.

        When we solve a jittered system (AAT + jitter I) y = r, we still compute
        y^T AAT y via y^T(r - jitter y).
        """

        if Au.ndim != 1 or Au.numel() != self.m:
            raise ValueError(f"Au must have shape ({self.m},)")

        r = Au - self.b
        y = self.solve_aat(r)

        # y^T AAT y = y^T (r - jitter y)
        yAATy = torch.dot(y, r) - float(self._aat_jitter) * torch.dot(y, y)

        r2_proj = r2 - 2.0 * torch.dot(Au, y) + yAATy
        return torch.clamp(r2_proj, min=0.0)

    @torch.no_grad()
    def projected_huber_gap_from_state(self, Au: torch.Tensor, r2: torch.Tensor) -> float:
        """Return h_δ(u_proj) - h_δ(u⋆)."""

        r2_proj = self.projected_r2_from_state(Au, r2)
        obj_proj = self.huber_value_from_r2(r2_proj)
        gap = obj_proj - self._opt_obj
        gap = torch.clamp(gap, min=0.0)
        return float(gap.item())

    @torch.no_grad()
    def projected_u_from_state(self, u: torch.Tensor, Au: torch.Tensor) -> torch.Tensor:
        """Euclidean projection of u onto the affine set {A u = b}.

        This is primarily used for diagnostics when the primal objective
        includes non-radial terms (e.g. ℓ_1 regularization), where knowing
        only ||u|| is insufficient.
        """

        if u.ndim != 1 or u.numel() != self.n_u:
            raise ValueError(f"u must have shape ({self.n_u},)")
        if Au.ndim != 1 or Au.numel() != self.m:
            raise ValueError(f"Au must have shape ({self.m},)")

        r = Au - self.b
        y = self.solve_aat(r)
        # u_proj = u - A^T y
        return u - (self.A.T @ y)

    @torch.no_grad()
    def projected_composite_objective(
        self,
        u: torch.Tensor,
        Au: torch.Tensor,
        lambda1: float = 0.0,
        lambda2: float = 0.0,
    ) -> float:
        """Return h_δ(u_proj) + λ1||u_proj||_1 + (λ2/2)||u_proj||^2.

        The projection u_proj is the Euclidean projection of u onto {A u = b}.
        """

        u_proj = self.projected_u_from_state(u, Au)
        r2_proj = torch.dot(u_proj, u_proj)
        obj = self.huber_value_from_r2(r2_proj)
        if lambda1 != 0.0:
            obj = obj + float(lambda1) * torch.sum(torch.abs(u_proj))
        if lambda2 != 0.0:
            obj = obj + 0.5 * float(lambda2) * r2_proj
        return float(obj.item())


    # ---------------------------------------------------------------------
    # Operator evaluation
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def func_map_with_state(
        self, x: torch.Tensor, Au: torch.Tensor, ATv: torch.Tensor, r2: torch.Tensor
    ) -> torch.Tensor:
        """Compute the full operator F(x) using cached state."""

        u, _ = self.split(x)
        scale = self.huber_scale(r2)
        Fu = scale * u + ATv
        Fv = self.b - Au
        return torch.cat((Fu, Fv), dim=0)

    @torch.no_grad()
    def func_map_slice_with_state(
        self,
        x: torch.Tensor,
        Au: torch.Tensor,
        ATv: torch.Tensor,
        r2: torch.Tensor,
        sl: slice,
    ) -> torch.Tensor:
        """Compute F(x)[sl] using cached state.

        This assumes ``sl`` does not cross the u/v boundary.
        """

        start = 0 if sl.start is None else int(sl.start)
        stop = self.n if sl.stop is None else int(sl.stop)

        if stop <= self.n_u:
            # u-block
            scale = self.huber_scale(r2)
            u_slice = x[start:stop]
            return scale * u_slice + ATv[start:stop]

        if start >= self.n_u:
            # v-block (shift indices)
            i0 = start - self.n_u
            i1 = stop - self.n_u
            return self.b[i0:i1] - Au[i0:i1]

        raise ValueError("Block slice crosses u/v boundary; please partition blocks accordingly.")

    # ---------------------------------------------------------------------
    # Incremental state updates
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def update_state_after_block_update_(
        self,
        x: torch.Tensor,
        Au: torch.Tensor,
        ATv: torch.Tensor,
        r2: torch.Tensor,
        sl: slice,
        old_block: torch.Tensor,
        new_block: torch.Tensor,
    ) -> torch.Tensor:
        """Update (Au, ATv, r2) *in place* after overwriting x[sl].

        Args:
            x: full iterate (modified by caller prior to this call)
            Au: cached A u (modified in place)
            ATv: cached A^T v (modified in place)
            r2: cached ||u||^2 (returned updated)
            sl: slice updated
            old_block: x_old[sl] (1D tensor)
            new_block: x_new[sl] (1D tensor)

        Returns:
            Updated r2 tensor.
        """

        start = 0 if sl.start is None else int(sl.start)
        stop = self.n if sl.stop is None else int(sl.stop)

        if stop <= self.n_u:
            # u-update
            dq = new_block - old_block
            # r2 <- r2 + ||new||^2 - ||old||^2
            r2 = r2 + torch.dot(new_block, new_block) - torch.dot(old_block, old_block)
            # Au <- Au + A[:,idx] dq
            Au.add_(self.A[:, start:stop] @ dq)
            return r2

        if start >= self.n_u:
            # v-update
            i0 = start - self.n_u
            i1 = stop - self.n_u
            dv = new_block - old_block
            ATv.add_(self.A[i0:i1, :].T @ dv)
            return r2

        raise ValueError("Block slice crosses u/v boundary; please partition blocks accordingly.")
