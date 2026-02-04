"""LC-Huber g-functions.

The LC-Huber benchmark is a saddle / VI formulation for the linearly
constrained problem

    minimize_u h_\delta(u)   subject to  A u = b,

where h_\delta is the (vector) Huber loss. The saddle operator is implemented
in :mod:`src.problems.lc_huber_opr_func`.

This file provides:

* :class:`ZeroGFunc`: the historical default g \equiv 0.
* :class:`ElasticNetGFunc`: \ell_1 + \ell_2 regularization on the primal
  variable u (and identity on the dual variable v), mirroring the SVM setup.

The proximal operator of the Elastic-Net term is coordinate-wise and supports
an optional diagonal metric (used by normalized / preconditioned variants).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


class ZeroGFunc:
    """Zero function g(x) = 0 with identity proximal map."""

    def __init__(self, n: int):
        self.n = int(n)

    def func_value(self, x: np.ndarray) -> float:
        return 0.0

    def prox_opr(self, u: np.ndarray, tau: float, d: int | None = None) -> np.ndarray:
        # tau and d are ignored; kept for interface compatibility with other benchmarks.
        return u

    # Torch helpers (used by torch algorithms in this repo)
    @torch.no_grad()
    def prox_opr_torch(self, z: torch.Tensor, tau: float, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return z

    @torch.no_grad()
    def prox_block_torch(
        self,
        z_block: torch.Tensor,
        block_type: str,
        tau: float,
        weights_block: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return z_block


@torch.no_grad()
def _soft_threshold_torch(x: torch.Tensor, thresh: torch.Tensor | float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)


def _soft_threshold_numpy(x: np.ndarray, thresh: np.ndarray | float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


@dataclass
class ElasticNetGFunc:
    """Elastic-Net regularizer on u and identity on v.

    g(u,v) = lambda1 * ||u||_1 + (lambda2/2) * ||u||^2.

    Parameters
    ----------
    n_u, n_v:
        Dimensions of u and v.
    lambda1, lambda2:
        Nonnegative regularization strengths.
    """

    n_u: int
    n_v: int
    lambda1: float = 0.0
    lambda2: float = 0.0

    def __post_init__(self) -> None:
        self.n_u = int(self.n_u)
        self.n_v = int(self.n_v)
        self.n = int(self.n_u + self.n_v)
        self.lambda1 = float(self.lambda1)
        self.lambda2 = float(self.lambda2)
        if self.lambda1 < 0.0 or self.lambda2 < 0.0:
            raise ValueError("lambda1 and lambda2 must be nonnegative")

    # -----------------------------
    # Values
    # -----------------------------
    def func_value(self, x: np.ndarray) -> float:
        u = x[: self.n_u]
        val = 0.0
        if self.lambda1 != 0.0:
            val += self.lambda1 * float(np.sum(np.abs(u)))
        if self.lambda2 != 0.0:
            val += 0.5 * self.lambda2 * float(np.dot(u, u))
        return float(val)

    # -----------------------------
    # Proximal maps (NumPy)
    # -----------------------------
    def prox_opr(
        self,
        z: np.ndarray,
        tau: float,
        d: int | None = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Full proximal map.

        Solves, for the u-block,

            argmin_u 0.5 ||u - z_u||_W^2 + tau * [lambda1 ||u||_1 + (lambda2/2)||u||^2]

        with W diagonal if weights is provided (weights = diag(W)). The v-block
        is returned unchanged.
        """
        tau = float(tau)
        if tau <= 0.0 or (self.lambda1 == 0.0 and self.lambda2 == 0.0):
            return z

        out = z.copy()
        zu = out[: self.n_u]

        if weights is None:
            # u = soft(z, tau*lambda1) / (1 + tau*lambda2)
            if self.lambda1 != 0.0:
                zu = _soft_threshold_numpy(zu, tau * self.lambda1)
            if self.lambda2 != 0.0:
                zu = zu / (1.0 + tau * self.lambda2)
            out[: self.n_u] = zu
            return out

        w = weights[: self.n_u]
        w = np.maximum(w, 1e-24)

        # u_i = (w_i/(w_i + tau*lambda2)) * soft(z_i, tau*lambda1 / w_i)
        if self.lambda1 != 0.0:
            zu = _soft_threshold_numpy(zu, (tau * self.lambda1) / w)
        if self.lambda2 != 0.0:
            zu = (w / (w + tau * self.lambda2)) * zu
        out[: self.n_u] = zu
        return out

    # -----------------------------
    # Proximal maps (Torch)
    # -----------------------------
    @torch.no_grad()
    def prox_opr_torch(
        self,
        z: torch.Tensor,
        tau: float,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tau = float(tau)
        if tau <= 0.0 or (self.lambda1 == 0.0 and self.lambda2 == 0.0):
            return z

        out = z.clone()
        zu = out[: self.n_u]

        if weights is None:
            if self.lambda1 != 0.0:
                zu = _soft_threshold_torch(zu, tau * self.lambda1)
            if self.lambda2 != 0.0:
                zu = zu / (1.0 + tau * self.lambda2)
            out[: self.n_u] = zu
            return out

        w = weights[: self.n_u].clamp_min(1e-24)

        if self.lambda1 != 0.0:
            zu = _soft_threshold_torch(zu, (tau * self.lambda1) / w)
        if self.lambda2 != 0.0:
            zu = (w / (w + tau * self.lambda2)) * zu

        out[: self.n_u] = zu
        return out

    @torch.no_grad()
    def prox_block_torch(
        self,
        z_block: torch.Tensor,
        block_type: str,
        tau: float,
        weights_block: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Block proximal map.

        * For u-blocks: apply the elastic-net prox coordinate-wise.
        * For v-blocks: identity.
        """
        block_type = str(block_type).lower()
        tau = float(tau)

        if block_type != "u":
            return z_block

        if tau <= 0.0 or (self.lambda1 == 0.0 and self.lambda2 == 0.0):
            return z_block

        out = z_block

        if weights_block is None:
            if self.lambda1 != 0.0:
                out = _soft_threshold_torch(out, tau * self.lambda1)
            if self.lambda2 != 0.0:
                out = out / (1.0 + tau * self.lambda2)
            return out

        w = weights_block.clamp_min(1e-24)
        if self.lambda1 != 0.0:
            out = _soft_threshold_torch(out, (tau * self.lambda1) / w)
        if self.lambda2 != 0.0:
            out = (w / (w + tau * self.lambda2)) * out
        return out
