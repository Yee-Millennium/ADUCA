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
from src.problems.GMVI_func import GMVIProblem
from src.problems.lc_huber_torch_oracle import LCHuberTorchOracle


def _dtype_from_string(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).lower()
    if s == "float32":
        return torch.float32
    if s == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


def _weighted_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # ||x||_{diag(w)} = sqrt( sum_i w_i x_i^2 )
    return torch.sqrt(torch.sum(w * x * x) + 1e-24)


def coder(
    problem: GMVIProblem,
    exitcriterion: ExitCriterion,
    parameters: Dict,
    x0: Optional[np.ndarray] = None,
) -> Tuple[Results, np.ndarray]:
    """CODER (fixed-L) for LC-Huber + ElasticNet in torch."""
    return _coder_impl(problem, exitcriterion, parameters, x0=x0, normalized=False, linesearch=False)


def coder_normalized(
    problem: GMVIProblem,
    exitcriterion: ExitCriterion,
    parameters: Dict,
    x0: Optional[np.ndarray] = None,
) -> Tuple[Results, np.ndarray]:
    """Rescaled CODER (fixed-L) using a diagonal Lipschitz preconditioner."""
    return _coder_impl(problem, exitcriterion, parameters, x0=x0, normalized=True, linesearch=False)


def coder_linesearch(
    problem: GMVIProblem,
    exitcriterion: ExitCriterion,
    parameters: Dict,
    x0: Optional[np.ndarray] = None,
) -> Tuple[Results, np.ndarray]:
    """CODER with epoch-wise linesearch (torch)."""
    return _coder_impl(problem, exitcriterion, parameters, x0=x0, normalized=False, linesearch=True)


def coder_linesearch_normalized(
    problem: GMVIProblem,
    exitcriterion: ExitCriterion,
    parameters: Dict,
    x0: Optional[np.ndarray] = None,
) -> Tuple[Results, np.ndarray]:
    """Rescaled CODER_linesearch using a diagonal Lipschitz preconditioner."""
    return _coder_impl(problem, exitcriterion, parameters, x0=x0, normalized=True, linesearch=True)


def _coder_impl(
    problem: GMVIProblem,
    exitcriterion: ExitCriterion,
    parameters: Dict,
    *,
    x0: Optional[np.ndarray],
    normalized: bool,
    linesearch: bool,
) -> Tuple[Results, np.ndarray]:
    device = torch.device(str(parameters.get("device", "cpu")))
    dtype = _dtype_from_string(parameters.get("dtype", "float64"))

    operator = problem.operator_func
    g = problem.g_func

    n_u = int(operator.n_u)
    n_v = int(operator.n_v)
    n = int(operator.n)

    # Blocks: do not cross u/v boundary.
    block_size_u = int(parameters.get("block_size_u", parameters.get("block_size", 1)))
    block_size_v = int(parameters.get("block_size_v", parameters.get("block_size", 1)))
    blocks, block_types = construct_uv_block_slices(n_u, n_v, block_size_u, block_size_v)
    m = len(blocks)

    oracle = LCHuberTorchOracle(operator.A, operator.b, delta=operator.delta, device=device, dtype=dtype)

    # Global Lipschitz (fixed-L mode uses a = 1/(2L))
    if parameters.get("L") is not None:
        L0 = float(parameters["L"])
    else:
        # Fallback: spectral norm computation via numpy (can be expensive for large dense A)
        L0 = float(operator.estimate_global_L())

    if not (L0 > 0.0):
        raise ValueError("L must be positive")

    # Preconditioner Λ^{-1}
    prec_mode = str(parameters.get("preconditioner", "diag_lipschitz")) if normalized else "identity"
    normalizers = normalizers_torch(oracle.A, device=device, dtype=dtype, mode=prec_mode)
    normalizers_recip = recip_normalizers_torch(normalizers)
    use_weighted_prox = (prec_mode != "identity")

    # Initialization
    if x0 is None:
        x = torch.ones(n, dtype=dtype, device=device)
    else:
        x = torch.tensor(x0, dtype=dtype, device=device)
        if x.ndim != 1 or x.numel() != n:
            raise ValueError(f"x0 must be 1D of length {n}")

    # CODER uses a fixed reference point x_init for dual-averaging prox.
    x_init = x.clone()

    # State and operator
    Au, ATv, r2 = oracle.compute_state(x)
    F_store = oracle.func_map_with_state(x, Au, ATv, r2)

    # Buffers
    z_dual = torch.zeros(n, dtype=dtype, device=device)
    z_dual_prev = torch.zeros_like(z_dual)
    F_prev_epoch = torch.zeros_like(F_store)

    # Delayed operator storage (per-block)
    F_tilde = F_store.clone()
    F_tilde_prev = F_store.clone()

    # Iteration bookkeeping
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
    logresult(results, iteration, 0.0, opt_measure, L=L0)

    # ------------------------------------------------------------
    # Fixed-L CODER
    # ------------------------------------------------------------
    if not linesearch:
        if normalized:
            L0 = L0 
        else:
            L0 = L0 
        logging.info(f"CODER: Using L0 = {L0:.6e}")
        a = 0.0
        a_prev = 0.0
        A_accum = 0.0
        a_base = 1.0 / (2.0 * L0)


        while not CheckExitCondition(exitcriterion, iteration, time.time() - start, opt_measure):
            # epoch constants
            z_dual_prev.copy_(z_dual)
            F_prev_epoch.copy_(F_store)

            a_prev = float(a)
            a = float(a_base)
            A_accum = float(A_accum + a)

            for sl, btype in zip(blocks, block_types):
                F_block = oracle.func_map_slice_with_state(x, Au, ATv, r2, sl)

                # shift delayed operator
                F_tilde_prev[sl] = F_tilde[sl]
                F_tilde[sl] = F_block

                # CODER correction
                if a_prev == 0.0:
                    F_bar = F_block
                else:
                    F_bar = F_block + (a_prev / a) * (F_prev_epoch[sl] - F_tilde_prev[sl])

                # dual update
                if normalized:
                    z_dual[sl] = z_dual_prev[sl] + a * (normalizers[sl] * F_bar)
                else:
                    z_dual[sl] = z_dual_prev[sl] + a * F_bar

                # primal update (dual averaging prox)
                old_block = x[sl].clone()
                x_tmp = x_init[sl] - z_dual[sl]

                if use_weighted_prox:
                    new_block = g.prox_block_torch(x_tmp, block_type=btype, tau=A_accum, weights_block=normalizers_recip[sl])
                else:
                    new_block = g.prox_block_torch(x_tmp, block_type=btype, tau=A_accum, weights_block=None)

                x[sl] = new_block
                r2 = oracle.update_state_after_block_update_(x, Au, ATv, r2, sl, old_block, new_block)

            # end epoch
            F_store = oracle.func_map_with_state(x, Au, ATv, r2)
            iteration += m

            if (iteration // m) % int(exitcriterion.loggingfreq) == 0:
                opt_measure = compute_opt_measure(
                    opt_kind,
                    x=x,
                    F_x=F_store,
                    g=g,
                    oracle=oracle,
                    Au=Au,
                    r2=r2,
                )
                logresult(results, iteration, time.time() - start, opt_measure, L=L0)
                logging.info(f"CODER iter {iteration}: opt_measure={opt_measure:.3e}")

        # final log
        opt_measure = compute_opt_measure(
            opt_kind,
            x=x,
            F_x=F_store,
            g=g,
            oracle=oracle,
            Au=Au,
            r2=r2,
        )
        logresult(results, iteration, time.time() - start, opt_measure, L=L0)
        return results, x.detach().cpu().numpy()

    # ------------------------------------------------------------
    # CODER with epoch-wise linesearch
    # ------------------------------------------------------------
    L_min = float(parameters.get("L_min", 1e-12))
    L_max = float(parameters.get("L_max", 1e12))
    L_cur = float(parameters.get("L_init", L0))
    L_cur = max(L_min, min(L_cur, L_max))

    min_step = float(parameters.get("min_step", 0.0))
    max_backtracks_param = parameters.get("max_backtracks")
    max_backtracks = float("inf") if max_backtracks_param is None else int(max_backtracks_param)
    logging.info(f"max_backtracks = {max_backtracks}")

    a_prev = 0.0
    A_accum = 0.0

    # We keep the same reference x_init for the entire run.
    while not CheckExitCondition(exitcriterion, iteration, time.time() - start, opt_measure):
        x_prev_epoch = x.clone()
        Au_prev_epoch = Au.clone()
        ATv_prev_epoch = ATv.clone()
        r2_prev_epoch = r2.clone()

        z_dual_prev.copy_(z_dual)
        F_prev_epoch.copy_(F_store)
        F_tilde_prev_epoch = F_tilde.clone()  # delayed operator from previous epoch

        accepted = False
        backtrack = 0

        while not accepted:
            if backtrack > max_backtracks:
                raise RuntimeError("CODER_linesearch: too many backtracks")

            a_cur = 1.0 / (2.0 * L_cur)
            if (min_step > 0.0) and (a_cur <= min_step):
                # Step is already tiny; accept to avoid stalling.
                accepted = True

            # Trial copies
            x_trial = x_prev_epoch.clone()
            z_dual_trial = z_dual_prev.clone()
            F_tilde_trial = F_tilde_prev_epoch.clone()

            Au_trial = Au_prev_epoch.clone()
            ATv_trial = ATv_prev_epoch.clone()
            r2_trial = r2_prev_epoch.clone()

            p = torch.zeros(n, dtype=dtype, device=device)

            A_trial = float(A_accum + a_cur)

            # One epoch
            for sl, btype in zip(blocks, block_types):
                F_block = oracle.func_map_slice_with_state(x_trial, Au_trial, ATv_trial, r2_trial, sl)
                p[sl] = F_block

                if a_prev == 0.0:
                    F_bar = F_block
                else:
                    F_bar = F_block + (a_prev / a_cur) * (F_prev_epoch[sl] - F_tilde_prev_epoch[sl])

                if normalized:
                    z_dual_trial[sl] = z_dual_prev[sl] + a_cur * (normalizers[sl] * F_bar)
                else:
                    z_dual_trial[sl] = z_dual_prev[sl] + a_cur * F_bar

                old_block = x_trial[sl].clone()
                x_tmp = x_init[sl] - z_dual_trial[sl]

                if use_weighted_prox:
                    new_block = g.prox_block_torch(x_tmp, block_type=btype, tau=A_trial, weights_block=normalizers_recip[sl])
                else:
                    new_block = g.prox_block_torch(x_tmp, block_type=btype, tau=A_trial, weights_block=None)

                x_trial[sl] = new_block
                r2_trial = oracle.update_state_after_block_update_(x_trial, Au_trial, ATv_trial, r2_trial, sl, old_block, new_block)

                # delayed operator for next epoch if accepted
                F_tilde_trial[sl] = F_block

            # End epoch: compute full operator at trial point
            F_trial = oracle.func_map_with_state(x_trial, Au_trial, ATv_trial, r2_trial)

            # Acceptance test: ||F_trial - p|| <= L_cur ||x_trial - x_prev_epoch||.
            if not accepted:
                diff_F = F_trial - p
                diff_x = x_trial - x_prev_epoch

                if normalized:
                    norm_dF = _weighted_norm(diff_F, normalizers)
                    norm_dx = _weighted_norm(diff_x, normalizers_recip)
                else:
                    norm_dF = torch.linalg.vector_norm(diff_F)
                    norm_dx = torch.linalg.vector_norm(diff_x)

                if float(norm_dx.item()) <= 1e-18:
                    accepted = True
                else:
                    accepted = float(norm_dF.item()) <= float(L_cur) * float(norm_dx.item())

            if accepted:
                # Commit
                x = x_trial
                z_dual = z_dual_trial
                F_store = F_trial
                F_tilde = F_tilde_trial

                Au, ATv, r2 = Au_trial, ATv_trial, r2_trial

                A_accum = A_trial
                a_prev = float(a_cur)
            else:
                L_cur = min(L_max, 2.0 * L_cur)
                backtrack += 1
                iteration += m # count backtrack as work

        iteration += m

        if (iteration // m) % int(exitcriterion.loggingfreq) == 0:
            opt_measure = compute_opt_measure(
                opt_kind,
                x=x,
                F_x=F_store,
                g=g,
                oracle=oracle,
                Au=Au,
                r2=r2,
            )
            logresult(results, iteration, time.time() - start, opt_measure, L=L_cur)
            logging.info(f"CODER_linesearch iter {iteration}: L={L_cur:.3e}, opt_measure={opt_measure:.3e}")

    # final log
    opt_measure = compute_opt_measure(
        opt_kind,
        x=x,
        F_x=F_store,
        g=g,
        oracle=oracle,
        Au=Au,
        r2=r2,
    )
    logresult(results, iteration, time.time() - start, opt_measure, L=L_cur)
    return results, x.detach().cpu().numpy()
