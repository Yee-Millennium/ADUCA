from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.algorithms.utils.exitcriterion import CheckExitCondition, ExitCriterion
from src.algorithms.utils.helper import construct_uv_block_slices, compute_opt_measure
from src.algorithms.utils.preconditioner import normalizers_torch, recip_normalizers_torch
from src.algorithms.utils.results import Results, logresult
from src.problems.lc_huber_torch_oracle import LCHuberTorchOracle


def _dtype_from_string(dtype_str: str) -> torch.dtype:
    dtype_str = str(dtype_str).lower()
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


def pccm_torch(problem, exitcriterion: ExitCriterion, parameters: Dict, x0: Optional[np.ndarray] = None) -> Tuple[Results, np.ndarray]:
    """PCCM (torch), unnormalized."""

    return _pccm_impl(problem, exitcriterion, parameters, x0=x0, normalized=False)


def pccm_torch_normalized(problem, exitcriterion: ExitCriterion, parameters: Dict, x0: Optional[np.ndarray] = None) -> Tuple[Results, np.ndarray]:
    """PCCM (torch) with a diagonal Lipschitz preconditioner."""

    return _pccm_impl(problem, exitcriterion, parameters, x0=x0, normalized=True)


def _pccm_impl(
    problem,
    exitcriterion: ExitCriterion,
    parameters: Dict,
    *,
    x0: Optional[np.ndarray],
    normalized: bool,
) -> Tuple[Results, np.ndarray]:

    device = torch.device(str(parameters.get("device", "cpu")))
    dtype = _dtype_from_string(parameters.get("dtype", "float64"))

    operator = problem.operator_func
    g = problem.g_func

    n = int(operator.n)
    n_u = int(operator.n_u)

    # Blocks (do not cross u/v boundary)
    block_size_u = int(parameters.get("block_size_u", parameters.get("block_size", 1)))
    block_size_v = int(parameters.get("block_size_v", parameters.get("block_size", 1)))
    blocks, block_types = construct_uv_block_slices(n_u, int(operator.n_v), block_size_u, block_size_v)
    m = len(blocks)

    oracle = LCHuberTorchOracle(operator.A, operator.b, delta=float(operator.delta), device=str(device), dtype=dtype)

    # Lipschitz constant and step size
    if parameters.get("L") is not None:
        L = float(parameters["L"])
    else:
        L = float(operator.estimate_L())
    logging.info(f"PCCM: estimated global Lipschitz L = {L:.3e}")
    lipschitz_mult = float(parameters.get("lipschitz_mult", 1.0))
    L_eff = max(1e-12, L * lipschitz_mult)
    if normalized:
        L_eff = L_eff 
    else:
        L_eff = L_eff 
    logging.info(f"PCCM: Using L_eff = {L_eff:.6e} (L={L:.6e}, mult={lipschitz_mult:.3e})")
    a = 1.0 / (2.0 * L_eff)

    # Preconditioner
    if normalized:
        prec_mode = str(parameters.get("preconditioner", "diag_lipschitz"))
    else:
        prec_mode = "identity"

    normalizers = normalizers_torch(operator.A, device=device, dtype=dtype, mode=prec_mode)
    normalizers_recip = recip_normalizers_torch(normalizers)

    # Initialization
    if x0 is None:
        x0 = torch.ones(n, device=device, dtype=dtype)
    else:
        x0 = torch.as_tensor(x0, device=device, dtype=dtype)
        if x0.ndim != 1 or x0.numel() != n:
            raise ValueError(f"x0 must be 1D of length {n}")

    x0_t = x0.clone()
    x = x0_t.clone()

    # State and operator values
    Au, ATv, r2 = oracle.compute_state(x)
    F_store = oracle.func_map_with_state(x, Au, ATv, r2)

    # PCCM buffers
    z = torch.zeros_like(x)
    z_prev = torch.zeros_like(x)

    results = Results()
    start = time.time()
    iteration = 0

    opt_kind = str(parameters.get("opt_measure", "prox_residual"))

    opt_measure = compute_opt_measure(
        opt_kind,
        x=x,
        F_x=F_store,
        g=g,
        oracle=oracle,
        Au=Au,
        r2=r2,
    )
    logresult(results, iteration, 0.0, opt_measure, L=L_eff)

    # Accumulator for the proximal parameter (matches the SVM implementation style)
    A_accum = 0.0

    while not CheckExitCondition(exitcriterion, iteration, time.time() - start, opt_measure):
        A_accum += a
        z_prev.copy_(z)

        for sl, typ in zip(blocks, block_types):
            # Operator evaluation (current x)
            F_block = oracle.func_map_slice_with_state(x, Au, ATv, r2, sl)

            # Dual update
            if normalized:
                z[sl] = z_prev[sl] + a * (normalizers[sl] * F_block)
            else:
                z[sl] = z_prev[sl] + a * F_block

            # Primal proximal update
            old_block = x[sl].clone()
            z_tmp = x0_t[sl] - z[sl]

            if normalized:
                new_block = g.prox_block_torch(z_tmp, block_type=typ, tau=A_accum, weights_block=normalizers_recip[sl])
            else:
                new_block = g.prox_block_torch(z_tmp, block_type=typ, tau=A_accum, weights_block=None)

            x[sl] = new_block
            r2 = oracle.update_state_after_block_update_(x, Au, ATv, r2, sl, old_block, new_block)

        # End epoch
        F_store = oracle.func_map_with_state(x, Au, ATv, r2)
        iteration += 1

        if iteration % int(exitcriterion.loggingfreq) == 0:
            opt_measure = compute_opt_measure(
                opt_kind,
                x=x,
                F_x=F_store,
                g=g,
                oracle=oracle,
                Au=Au,
                r2=r2,
            )
            logresult(results, iteration, time.time() - start, opt_measure, L=L_eff)
            logging.info(f"PCCM iter {iteration}: prox residual = {opt_measure:.3e}")

    opt_measure = compute_opt_measure(
        opt_kind,
        x=x,
        F_x=F_store,
        g=g,
        oracle=oracle,
        Au=Au,
        r2=r2,
    )
    logresult(results, iteration, time.time() - start, opt_measure, L=L_eff)

    return results, x.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Backward-compatible aliases (run_algos imports these names)
# -----------------------------------------------------------------------------
pccm = pccm_torch
pccm_normalized = pccm_torch_normalized
