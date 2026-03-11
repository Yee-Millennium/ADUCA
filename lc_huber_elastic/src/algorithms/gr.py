"""Golden-Ratio (GR) algorithm (torch) for LC-Huber with optional Elastic-Net g.

We solve the composite monotone inclusion

    0 \in F(x) + \partial g(x),   x=(u,v),

where the LC-Huber saddle operator is

    F(u,v) = (\nabla h_\delta(u) + A^T v,\; b - A u),

and the (optional) regularizer is

    g(u,v) = \lambda_1 ||u||_1 + (\lambda_2/2) ||u||_2^2.

When \lambda_1 = \lambda_2 = 0, this reduces to the historical g \equiv 0 case.

Notes
-----
* The implementation supports a diagonal preconditioner (normalizers) applied
  to the forward step, together with the corresponding diagonal metric in the
  proximal map.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Tuple

import torch
import numpy as np

from src.algorithms.utils.exitcriterion import CheckExitCondition, ExitCriterion
from src.algorithms.utils.preconditioner import normalizers_torch, recip_normalizers_torch
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_uv_block_slices, compute_opt_measure
from src.problems.GMVI_func import GMVIProblem
from src.problems.lc_huber_torch_oracle import LCHuberTorchOracle


def _dtype_from_string(dtype_str: str) -> torch.dtype:
    dtype_str = str(dtype_str).lower()
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


def gr_torch(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, x0=None) -> Tuple[Results, np.ndarray]:
    """Run GR on LC-Huber (torch).

    Parameters
    ----------
    problem:
        GMVIProblem wrapping an LCHuberOprFunc and a g-func (possibly Elastic-Net).
    exit_criterion:
        Stopping criteria.
    parameters:
        Optional keys:
            - L: override global Lipschitz estimate for F
            - device, dtype
            - preconditioner ('identity' or 'diag_lipschitz')
    x0:
        Optional initial point (numpy/torch), length n_u+n_v.

    Returns
    -------
    (results, x_final_torch)
    """

    device = torch.device(str(parameters.get("device", "cpu")))
    dtype = _dtype_from_string(parameters.get("dtype", "float64"))
    beta = float(parameters.get("beta", 0.7))

    operator = problem.operator_func
    g = problem.g_func

    n_u = int(operator.n_u)
    n_v = int(operator.n_v)
    n = int(operator.n)

    block_size_u = int(parameters.get("block_size_u", parameters.get("block_size", 1)))
    block_size_v = int(parameters.get("block_size_v", parameters.get("block_size", 1)))
    blocks, _ = construct_uv_block_slices(n_u, n_v, block_size_u, block_size_v)
    m = len(blocks)

    oracle = LCHuberTorchOracle(operator.A, operator.b, delta=operator.delta, device=device, dtype=dtype)

    # Lipschitz estimate for F
    L = float(parameters.get("L_est", parameters.get("L", operator.estimate_global_L())))
    if L <= 0:
        raise ValueError("L must be positive")
    a = 0.01
    a_prev = a

    rho = beta + beta**2

    # Preconditioner
    prec_mode = str(parameters.get("preconditioner", "identity"))
    normalizers = normalizers_torch(oracle.A, device=device, dtype=dtype, mode=prec_mode)
    normalizers_recip = recip_normalizers_torch(normalizers)
    use_weights = prec_mode != "identity"
    logging.info(f"GR_TORCH using preconditioner mode '{prec_mode}'")

    n = int(operator.n)

    # Initialization
    if x0 is None:
        x_prev = torch.ones(n, dtype=dtype, device=device)
    else:
        x_prev = torch.tensor(x0, dtype=dtype, device=device)
        if x_prev.ndim != 1 or x_prev.numel() != n:
            raise ValueError(f"x0 must be a 1D tensor/array of length {n}")
    v_prev = x_prev.clone()

    # First GR step: x_cur = prox_{a g}(x_prev - a * Λ^{-1} F(x_prev))
    Au_prev, ATv_prev, r2_prev = oracle.compute_state(x_prev)
    F_prev = oracle.func_map_with_state(x_prev, Au_prev, ATv_prev, r2_prev)

    x_forward = x_prev - a * (normalizers * F_prev)
    x_cur = g.prox_opr_torch(x_forward, tau=a, weights=normalizers_recip if use_weights else None)

    # Operator at x_cur (used for the first stepsize computation)
    Au_cur, ATv_cur, r2_cur = oracle.compute_state(x_cur)
    F = oracle.func_map_with_state(x_cur, Au_cur, ATv_cur, r2_cur)

    # Bookkeeping
    results = Results()
    start = time.time()
    k = 0

    opt_kind = str(parameters.get("opt_measure", "prox_residual"))

    # Initial log
    opt_measure = compute_opt_measure(
        opt_kind,
        x=x_prev,
        F_x=F_prev,
        g=g,
        oracle=oracle,
        Au=Au_prev,
        r2=r2_prev,
    )
    logresult(results, k, 0.0, opt_measure, L=L)

    with torch.no_grad():
        while not CheckExitCondition(exit_criterion, k, time.time() - start, opt_measure):
            # Stepsize update (Euclidean norm, matching SVM GR)
            step_1 = rho * a

            dx = x_cur - x_prev
            dF = F - F_prev
            norm_x = torch.sqrt(torch.dot(dx, dx * normalizers_recip)).item()
            norm_F = torch.sqrt(torch.dot(dF, dF * normalizers)).item()
            if norm_x <= 0.0 or norm_F <= 0.0:
                L = 0.0
                step_2 = math.inf
            else:
                L = norm_F / norm_x
                step_2 = 1.0 / (4.0 * (beta**2) * a_prev * (L**2))

            a_prev, a = a, min(step_1, step_2)

            v = (1 - beta) * x_cur + beta * v_prev

            x_forward = v - a * (normalizers * F)
            x_new = g.prox_opr_torch(x_forward, tau=a, weights=normalizers_recip if use_weights else None)

            # Shift
            x_prev = x_cur
            x_cur = x_new
            v_prev = v

            # Refresh operator at the new point
            Au_cur, ATv_cur, r2_cur = oracle.compute_state(x_cur)
            F_prev = F
            F = oracle.func_map_with_state(x_cur, Au_cur, ATv_cur, r2_cur)
            k += 1

            if k % int(exit_criterion.loggingfreq) == 0:
                opt_measure = compute_opt_measure(
                    opt_kind,
                    x=x_cur,
                    F_x=F,
                    g=g,
                    oracle=oracle,
                    Au=Au_cur,
                    r2=r2_cur,
                )
                logresult(results, k, time.time() - start, opt_measure, L=L)
                logging.info(f"GR_TORCH iter {k}: opt_measure={opt_measure:.3e}")

    # Final
    opt_measure = compute_opt_measure(
        opt_kind,
        x=x_cur,
        F_x=F,
        g=g,
        oracle=oracle,
        Au=Au_cur,
        r2=r2_cur,
    )
    logresult(results, k, time.time() - start, opt_measure, L=L)
    return results, x_cur.detach().cpu().numpy()


def gr_torch_normalized(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, x0=None) -> Tuple[Results, np.ndarray]:
    """GR with diagonal preconditioner enabled by default."""
    params = dict(parameters)
    # Default to diag_lipschitz if not specified
    params.setdefault("preconditioner", "diag_lipschitz")
    return gr_torch(problem, exit_criterion, params, x0=x0)


# -----------------------------------------------------------------------------
# Backward-compatible aliases (run_algos imports these names)
# -----------------------------------------------------------------------------
gr = gr_torch
gr_normalized = gr_torch_normalized
