"""Distributed (torch) implementation of ADUCA for the Nash-Cournot GMVI.

This implementation is purpose-built for the Nash_Equilibrium problem in this repo.
It exploits ADUCA's *one-cycle delayed operator* to update *all blocks in parallel*
(each epoch updates every block once, without a Python loop over blocks).

Key performance features:
- Fully vectorized GPU kernels for the primal update (no per-block Python loop).
- Efficient computation of the delayed operator \tilde{F}_{k+1} using prefix sums of
  block-wise quantity updates (also vectorized on GPU).
- Distributed partitioning by *contiguous block ranges* to keep communication minimal
  (only a few scalar collectives per epoch).

The public entrypoint is `aduca_distributed(problem, exit_criterion, parameters, u_0=None)`,
which matches the call pattern used by Nash_Equilibrium/main.py and run_algos.py.
"""

from __future__ import annotations

import math
import os
import time
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from src.algorithms.utils.exitcriterion import CheckExitCondition, ExitCriterion
from src.algorithms.utils.results import Results, logresult
from src.problems.GMVI_func import GMVIProblem


# -----------------------------
# Helpers
# -----------------------------


def _torch_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "float32").lower()
    if s in {"float32", "fp32", "f32"}:
        return torch.float32
    if s in {"float64", "fp64", "f64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype '{dtype_str}'. Use 'float32' or 'float64'.")


def _maybe_init_dist(dist_backend: str) -> Tuple[bool, int, int, int]:
    """Initialize torch.distributed if not initialized.

    Returns:
        (dist_on, rank, world_size, local_rank)
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return True, rank, world_size, local_rank

    # If launched via torchrun, these are set.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size == 1:
        # Single-process fallback: do NOT force init unless needed.
        return False, 0, 1, 0

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend=dist_backend, rank=rank, world_size=world_size)
    return True, rank, world_size, local_rank


def _select_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n <= 0:
            return torch.device("cpu")
        dev_id = local_rank
        if dev_id >= n:
            # Allow oversubscription (multiple ranks per GPU) instead of crashing.
            dev_id = dev_id % n
            logging.warning(
                f"LOCAL_RANK={local_rank} but only {n} visible CUDA devices; "
                f"mapping to cuda:{dev_id}."
            )
        torch.cuda.set_device(dev_id)
        return torch.device(f"cuda:{dev_id}")
    return torch.device("cpu")


def _contiguous_block_range(m: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Split m blocks into contiguous ranges across ranks."""
    start = (rank * m) // world_size
    end = ((rank + 1) * m) // world_size
    return start, end


@torch.no_grad()
def _demand_p_dp(Q: torch.Tensor, p_const: float, p_power: float, min_Q: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute p(Q) and p'(Q) for Nash inverse demand.

    p(Q) = p_const * Q^p_power, where p_power = -1/gamma_op
    dp(Q) = p'(Q) = p_power * p(Q) / Q
    """
    Qs = torch.clamp(Q, min=min_Q)
    # Qs = Q
    p = (p_const * torch.pow(Qs, p_power)).to(Qs.dtype)
    dp = (p_power * p / Qs).to(Qs.dtype)
    return p, dp


@torch.no_grad()
def _df_cost(q: torch.Tensor, c: torch.Tensor, L_pow: torch.Tensor, inv_beta_op: torch.Tensor) -> torch.Tensor:
    """Compute f'(q) = c + L^{1/beta} * q^{1/beta} elementwise (q >= 0)."""
    q_pos = torch.clamp(q, min=0.0)
    # torch.pow supports elementwise exponent tensor
    return c + L_pow * torch.pow(q_pos, inv_beta_op)


@torch.no_grad()
def _allreduce_sums(vals: torch.Tensor, dist_on: bool) -> torch.Tensor:
    """All-reduce SUM on a 1D tensor of scalars, returns reduced tensor."""
    if dist_on:
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    return vals


@torch.no_grad()
def _allgather_scalar(val: torch.Tensor, world_size: int, dist_on: bool) -> torch.Tensor:
    """All-gather a scalar tensor into a 1D tensor of shape (world_size,)."""
    if not dist_on or world_size == 1:
        return val.reshape(1)
    out = torch.empty(world_size, device=val.device, dtype=val.dtype)
    dist.all_gather_into_tensor(out, val.reshape(1))
    return out


@torch.no_grad()
def _gather_full_vector(u_local: torch.Tensor, n_local: int, dist_on: bool, rank: int, world_size: int) -> Optional[np.ndarray]:
    """Gather the full decision vector to rank 0 and return it as a numpy array.

    For maximum backend compatibility (including NCCL), this uses ALL_GATHER with padding,
    and then only rank 0 concatenates the pieces. This is called only at logging frequency
    and at the end of the run, so the extra replication is acceptable.

    IMPORTANT: This is a collective when dist_on==True and world_size>1, so ALL ranks
    must call it whenever it is invoked.
    """
    if not dist_on or world_size == 1:
        return u_local.detach().cpu().numpy()

    # Gather sizes (tiny) so we can pad to a common length.
    local_n_t = torch.tensor([n_local], device=u_local.device, dtype=torch.long)
    sizes = torch.empty(world_size, device=u_local.device, dtype=torch.long)
    dist.all_gather_into_tensor(sizes, local_n_t)

    max_n = int(sizes.max().item())
    u_pad = torch.zeros(max_n, device=u_local.device, dtype=u_local.dtype)
    u_pad[:n_local].copy_(u_local)

    gather_list = [torch.empty(max_n, device=u_local.device, dtype=u_local.dtype) for _ in range(world_size)]
    dist.all_gather(gather_list, u_pad)

    if rank != 0:
        return None

    # Concatenate in rank order (contiguous partition preserves global ordering).
    parts = []
    for r in range(world_size):
        parts.append(gather_list[r][: int(sizes[r].item())].detach().cpu())
    return torch.cat(parts, dim=0).numpy()


# -----------------------------
# Main solver
# -----------------------------


@torch.no_grad()
def aduca_distributed(
    problem: GMVIProblem,
    exit_criterion: ExitCriterion,
    parameters: Dict[str, Any],
    u_0: Optional[np.ndarray] = None,
):
    """Run distributed ADUCA on the Nash-Cournot GMVI.

    parameters keys used:
        beta (float): ADUCA extrapolation parameter in ( (sqrt(5)-1)/2, 1 )
        gamma (float): ADUCA gamma parameter in (0, 1 - 1/(beta(1+beta)))
        rho (float): ADUCA rho parameter (>1), step growth factor upper bound
        mu (float): strong convexity parameter (>=0)
        strong_convexity (bool): enable omega_k update
        block_size (int): block size (>=1)
        dist_backend (str): "nccl" or "gloo"
        dtype (str): "float32" or "float64"
        reduce_dtype (str|None): dtype for reductions (default: float64)
        sync_step (bool): if True, broadcast (a_k, omega_k) from rank 0 each epoch
        q_sync_every (int): kept for CLI compatibility; unused in this fully-parallel implementation
    """
    # -------------------------
    # Dist / device setup
    # -------------------------
    dist_backend = parameters.get("dist_backend", "nccl")
    dist_on, rank, world_size, local_rank = _maybe_init_dist(dist_backend)
    logging.info(f"Distributed: {dist_on}, rank: {rank}/{world_size}, local_rank: {local_rank}")
    device = _select_device(local_rank)

    dtype = _torch_dtype(parameters.get("dtype", "float32"))
    reduce_dtype = _torch_dtype(parameters.get("reduce_dtype", "float64"))

    sync_step = bool(parameters.get("sync_step", False))

    # -------------------------
    # Numerical stability / endgame heuristics (Fixes A, D, E)
    # -------------------------
    # Fix A: Stabilize local Lipschitz estimates when ||u_k - u_{k-1}||_Lambda is tiny.
    # This prevents (L_k, L_hat_k) from spuriously exploding near convergence and collapsing a_k.
    if reduce_dtype == torch.float32:
        denom_eps = float(parameters.get("denom_eps", 1e-20))
    else:
        denom_eps = float(parameters.get("denom_eps", 1e-30))
    L_cap = float(parameters.get("L_cap", 0.0))         # 0.0 => disabled
    L_hat_cap = float(parameters.get("L_hat_cap", 0.0)) # 0.0 => disabled

    # Fix D: Which iterate to use for reporting/stopping (opt_measure) and for the returned solution.
    # - opt_measure_point: 'u' (default) or 'v'
    # - return_point:      'v' (default, often smoother near a solution) or 'u'
    opt_measure_point = str(parameters.get("opt_measure_point", "u")).lower()
    if opt_measure_point not in {"u", "v"}:
        raise ValueError("opt_measure_point must be 'u' or 'v'")
    return_point = str(parameters.get("return_point", "v")).lower()
    if return_point not in {"u", "v"}:
        raise ValueError("return_point must be 'u' or 'v'")

    # Fix E: Final-phase step-size decay to damp oscillations when close to a solution.
    final_phase_enabled = bool(parameters.get("final_phase_enabled", True))
    final_phase_trigger = float(parameters.get("final_phase_trigger", 1e-3))
    final_phase_decay = float(parameters.get("final_phase_decay", 0.995))
    final_phase_a_max = float(parameters.get("final_phase_a_max", float("inf")))

    if denom_eps < 0.0:
        raise ValueError("denom_eps must be >= 0")
    if final_phase_trigger < 0.0:
        raise ValueError("final_phase_trigger must be >= 0")
    if final_phase_decay <= 0.0:
        raise ValueError("final_phase_decay must be > 0")
    if final_phase_decay > 1.0:
        # Treat values > 1 as 'disabled' (no decay).
        final_phase_decay = 1.0

    # -------------------------
    # Algorithm parameters
    # -------------------------
    beta_alg = float(parameters["beta"])
    gamma_alg = float(parameters["gamma"])
    rho = float(parameters["rho"])
    mu = float(parameters.get("mu", 0.0))
    strong_convexity = bool(parameters.get("strong_convexity", False))

    block_size = int(parameters.get("block_size", 1))
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    # Problem dimension
    n = int(problem.operator_func.n)  # number of firms / variables
    m = int(math.ceil(n / block_size))  # number of blocks

    # -------------------------
    # Operator parameters (Nash-Cournot)
    # -------------------------
    # These are numpy arrays in the problem; move only the local slice to GPU.
    gamma_op = float(problem.operator_func.gamma)
    c_full = np.asarray(problem.operator_func.c)
    L_full = np.asarray(problem.operator_func.L)
    beta_op_full = np.asarray(problem.operator_func.beta)

    # Partition blocks contiguously by rank
    start_b, end_b = _contiguous_block_range(m, rank, world_size)
    if end_b <= start_b:
        raise RuntimeError(
            f"Rank {rank} received an empty block range with m={m}, world_size={world_size}. "
            "Use fewer processes or smaller world_size."
        )

    start_idx = start_b * block_size
    end_idx = min(end_b * block_size, n)
    n_local = int(end_idx - start_idx)
    num_blocks_local = int(end_b - start_b)

    # Local slices of operator params
    c = torch.as_tensor(c_full[start_idx:end_idx], device=device, dtype=dtype)
    L = torch.as_tensor(L_full[start_idx:end_idx], device=device, dtype=dtype)
    beta_op = torch.as_tensor(beta_op_full[start_idx:end_idx], device=device, dtype=dtype)

    inv_beta_op = (1.0 / beta_op).to(dtype)
    L_pow = torch.pow(L, inv_beta_op)  # L^{1/beta_i}

    # Preconditioner (diagonal): Λ^{-1} corresponds to `normalizer`
    normalizer = (1.0 / L_pow) # L^{-1/beta_i} / beta_alg
    normalizer_recip = 1.0 / normalizer

    # normalizer = torch.ones_like(L_pow)
    # normalizer_recip = normalizer

    # Demand constants
    p_const = float(5000.0 ** (1.0 / gamma_op))
    p_power = float(-1.0 / gamma_op)
    min_Q = 1e-12

    # -------------------------
    # Step-size constants (Lemma 3.1 / Eq. (3.9))
    # -------------------------
    # rho_0 := min{rho, beta(1+beta)(1-gamma)}
    rho_0 = min(rho, beta_alg * (1.0 + beta_alg) * (1.0 - gamma_alg))
    # eta := sqrt( gamma(1+beta) / (1+beta^2) )
    eta = math.sqrt(gamma_alg * (1.0 + beta_alg) / (1.0 + beta_alg * beta_alg))
    # tau := 3 rho_0^2 (1+rho beta) / ( 2 (rho beta)^2 + 3 rho_0^2 (1+rho beta) )
    rb = rho * beta_alg
    tau = (3.0 * rho_0 * rho_0 * (1.0 + rb)) / (2.0 * (rb * rb) + 3.0 * rho_0 * rho_0 * (1.0 + rb))

    # C and C_hat coefficients in (3.9) for 1/L_k and 1/\hat{L}_k
    C = (eta / (2.0 * math.sqrt(beta_alg))) * (math.sqrt(tau) * rb) / (math.sqrt(3.0) * math.sqrt(1.0 + rb))
    C_hat = (eta / (2.0 * math.sqrt(beta_alg))) * (math.sqrt(1.0 - tau) * rb) / math.sqrt(2.0)

    # -------------------------
    # Initialization u0, F0, line search (Algorithm 5.1)
    # -------------------------
    if u_0 is None:
        # Avoid allocating a full n-vector on every rank.
        u0 = torch.ones(n_local, device=device, dtype=dtype)
    else:
        u0_np = np.asarray(u_0, dtype=np.float64)
        if u0_np.shape != (n,):
            raise ValueError(f"u_0 must have shape ({n},), got {u0_np.shape}")
        u0 = torch.as_tensor(u0_np[start_idx:end_idx], device=device, dtype=dtype)
    v = u0.clone()  # v0 = u0
    v_is_updated = False  # Fix D: v is not yet updated to v_k until the first main-loop step

    # Compute Q0 (global sum) once (all_reduce)
    local_Q0 = u0.sum(dtype=reduce_dtype)
    Q0 = local_Q0.clone()
    if dist_on:
        dist.all_reduce(Q0, op=dist.ReduceOp.SUM)

    df0 = _df_cost(u0, c, L_pow, inv_beta_op)
    p0, dp0 = _demand_p_dp(Q0, p_const, p_power, min_Q)
    F0 = df0 - p0 - u0 * dp0  # full operator at u0

    # We will need these buffers throughout
    u_prev = u0.clone()
    u = u0.clone()
    df = df0.clone()
    F_prev = F0.clone()
    F = F0.clone()

    # Delayed operators
    F_tilde_prev = F0.clone()  # \tilde{F}_0 = F(u0)
    F_tilde = F0.clone()       # placeholder, will be overwritten by \tilde{F}_1

    # Results / timing (rank 0 only logs)
    results = Results()
    start_time = time.time()

    # Fix E state: once we enter the 'final phase' we keep applying a gentle step-size decay.
    final_phase_on = False

    # Initial logging at k=0
    k = 0
    if k % (m * exit_criterion.loggingfreq) == 0:
        elapsed_time = time.time() - start_time

        # Keep for reference (do NOT delete):
        # u_full = _gather_full_vector(u0, n_local, dist_on, rank, world_size)  # numpy on rank0
        # opt_measure = float(problem.residual(u_full))

        # Keep for reference (do NOT delete): for k=0, averaged iterate coincides with u0
        # u_hat_full = _gather_full_vector(u0, n_local, dist_on, rank, world_size)  # numpy on rank0
        # opt_measure = float(problem.residual(u_hat_full))

        # Fix D (reporting): evaluate opt_measure on either u or v (configured by opt_measure_point)
        if opt_measure_point == "v":
            v_full = _gather_full_vector(v, n_local, dist_on, rank, world_size)  # numpy on rank0

            if rank == 0:
                opt_measure = float(problem.residual(v_full))
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
                logresult(results, k, elapsed_time, opt_measure, L=0.0, L_hat=0.0)
                exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)

                if final_phase_enabled and (not final_phase_on) and (opt_measure <= final_phase_trigger):
                    final_phase_on = True
            else:
                exit_flag = False
        else:
            u_full = _gather_full_vector(u0, n_local, dist_on, rank, world_size)  # numpy on rank0

            if rank == 0:
                opt_measure = float(problem.residual(u_full))
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
                logresult(results, k, elapsed_time, opt_measure, L=0.0, L_hat=0.0)
                exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)

                if final_phase_enabled and (not final_phase_on) and (opt_measure <= final_phase_trigger):
                    final_phase_on = True
            else:
                exit_flag = False
    else:
        exit_flag = False

    # Broadcast flags so all ranks stay consistent
    if dist_on and world_size > 1:
        # Keep for reference (do NOT delete):
        # flag_t = torch.tensor([1 if exit_flag else 0], device=device, dtype=torch.int64)
        # dist.broadcast(flag_t, src=0)
        # exit_flag = bool(flag_t.item())

        flags_t = torch.tensor(
            [1 if exit_flag else 0, 1 if final_phase_on else 0],
            device=device,
            dtype=torch.int64,
        )
        dist.broadcast(flags_t, src=0)
        exit_flag = bool(flags_t[0].item())
        final_phase_on = bool(flags_t[1].item())

    if exit_flag:
        # IMPORTANT: _gather_full_vector is collective under dist_on, so call on all ranks.
        # Keep for reference (do NOT delete):
        # output_x = _gather_full_vector(u0, n_local, dist_on, rank, world_size)

        output_local = v if return_point == "v" else u0
        output_x = _gather_full_vector(output_local, n_local, dist_on, rank, world_size)
        return results, output_x

    # Helper for block-wise ops
    last_block_len_global = n - (m - 1) * block_size  # in [1, block_size]
    has_partial_last_local = (end_b == m) and (last_block_len_global != block_size)
    num_full_blocks_local = num_blocks_local - (1 if has_partial_last_local else 0)
    full_elems_local = num_full_blocks_local * block_size

    def block_sum_1d(x: torch.Tensor) -> torch.Tensor:
        """Sum a local 1D tensor by blocks, returning (num_blocks_local,) tensor."""
        if num_blocks_local == 1 and has_partial_last_local:
            return torch.stack([x.sum(dtype=reduce_dtype)], dim=0).to(dtype)
        sums = []
        if num_full_blocks_local > 0:
            sums_full = x[:full_elems_local].view(num_full_blocks_local, block_size).sum(dim=1, dtype=reduce_dtype)
            sums.append(sums_full)
        if has_partial_last_local:
            sums.append(x[full_elems_local:].sum(dtype=reduce_dtype).view(1))
        out = torch.cat(sums, dim=0)
        return out.to(dtype)

    def compute_F_tilde_next(
        u_new: torch.Tensor,
        u_old: torch.Tensor,
        df_old: torch.Tensor,
        Q_old: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute \tilde{F}_{next} for local slice, plus Q_new.

        Uses:
            \tilde{F}_{next}^block(i) = F^block(i)(u_new^{<i}, u_old^{>=i})
        For Nash, this depends only on the total quantity Q at that mixed point, which can be
        obtained from block-wise prefix sums of (u_new - u_old).
        """
        delta = u_new - u_old  # local
        # block delta sums (local)
        delta_b = block_sum_1d(delta)  # (num_blocks_local,)

        # per-rank total delta (scalar)
        delta_total_local = delta_b.sum(dtype=reduce_dtype)
        deltas_all = _allgather_scalar(delta_total_local, world_size, dist_on)  # (world_size,)

        # prefix offset for this rank (exclusive)
        if dist_on and world_size > 1:
            prefix_offset = deltas_all[:rank].sum(dtype=reduce_dtype)
            global_delta = deltas_all.sum(dtype=reduce_dtype)
        else:
            prefix_offset = torch.zeros((), device=device, dtype=reduce_dtype)
            global_delta = delta_total_local

        Q_new = (Q_old + global_delta).to(reduce_dtype)

        # exclusive prefix within this rank's blocks
        if num_blocks_local == 1:
            prefix_local = torch.zeros(1, device=device, dtype=reduce_dtype)
        else:
            prefix_local = torch.zeros(num_blocks_local, device=device, dtype=reduce_dtype)
            prefix_local[1:] = torch.cumsum(delta_b[:-1].to(reduce_dtype), dim=0)

        # global prefix before each local block
        prefix_global = prefix_local + prefix_offset

        # Q at the mixed point for each local block
        Q_blocks = (Q_old.to(reduce_dtype) + prefix_global).to(dtype)

        # p, dp per block
        p_blocks, dp_blocks = _demand_p_dp(Q_blocks, p_const, p_power, min_Q)

        # Build \tilde{F} (vectorized by blocks)
        Ftilde_next = torch.empty_like(u_old)

        if num_full_blocks_local > 0:
            q_old_2d = u_old[:full_elems_local].view(num_full_blocks_local, block_size)
            df_old_2d = df_old[:full_elems_local].view(num_full_blocks_local, block_size)
            p2d = p_blocks[:num_full_blocks_local].view(num_full_blocks_local, 1)
            dp2d = dp_blocks[:num_full_blocks_local].view(num_full_blocks_local, 1)
            Ftilde_next[:full_elems_local].copy_((df_old_2d - p2d - q_old_2d * dp2d).view(-1))

        if has_partial_last_local:
            p_last = p_blocks[-1]
            dp_last = dp_blocks[-1]
            q_tail = u_old[full_elems_local:]
            df_tail = df_old[full_elems_local:]
            Ftilde_next[full_elems_local:].copy_(df_tail - p_last - q_tail * dp_last)

        return Ftilde_next, Q_new

    def compute_L_Lhat(
        u_k: torch.Tensor,
        u_km1: torch.Tensor,
        F_k: torch.Tensor,
        F_km1: torch.Tensor,
        Ft_k: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute L_k and \hat{L}_k as in (3.5)."""
        du = u_k - u_km1
        # Denominator: ||du||_Lambda^2 = sum_i Lambda_i * du_i^2, with Lambda = normalizer_recip
        u_norm_sq_local = torch.sum((du * du) * normalizer_recip, dtype=reduce_dtype)

        dF = F_k - F_km1
        dFt = F_k - Ft_k
        F_norm_sq_local = torch.sum((dF * dF) * normalizer, dtype=reduce_dtype)
        Ft_norm_sq_local = torch.sum((dFt * dFt) * normalizer, dtype=reduce_dtype)

        pack = torch.stack([u_norm_sq_local, F_norm_sq_local, Ft_norm_sq_local])
        pack = _allreduce_sums(pack, dist_on)

        u_norm_sq = float(pack[0].item())

        # Fix A: stabilize L_k and L_hat_k when ||u_k - u_{k-1}||_Lambda is tiny.
        # Keep the old behavior for reference (do NOT delete):
        # if u_norm_sq <= 0.0:
        #     return 0.0, 0.0
        # L_val = math.sqrt(float(pack[1].item()) / u_norm_sq)
        # L_hat = math.sqrt(float(pack[2].item()) / u_norm_sq)
        # return L_val, L_hat

        denom = max(u_norm_sq, denom_eps)
        if denom <= 0.0:
            return 0.0, 0.0

        L_val = math.sqrt(float(pack[1].item()) / denom)
        L_hat = math.sqrt(float(pack[2].item()) / denom)

        # Optional caps (set L_cap / L_hat_cap > 0.0 to enable)
        if L_cap > 0.0:
            L_val = min(L_val, L_cap)
        if L_hat_cap > 0.0:
            L_hat = min(L_hat, L_hat_cap)

        return L_val, L_hat

    # Line-search initialization (Algorithm 5.1)
    alpha = math.sqrt(2.0)
    i_ls = 0

    # These will be set by the last line-search iteration
    a0 = 1.0
    u1 = None
    df1 = None
    F1 = None
    Ft1 = None
    Q1 = None
    L1 = None
    Lhat1 = None

    # repeat loop (shrink a0)
    while True:
        a0 = alpha ** (-i_ls)

        # u1 = prox(u0 - a0 * normalizer * F0)
        u1_tmp = torch.clamp(u0 - (a0 * normalizer) * F0, min=0.0)

        # df(u1), F(u1) uses Q1
        df1_tmp = _df_cost(u1_tmp, c, L_pow, inv_beta_op)

        # Compute Ft1 and Q1 via prefix sums (also gives Q1)
        Ft1_tmp, Q1_tmp = compute_F_tilde_next(u1_tmp, u0, df0, Q0)

        p1, dp1 = _demand_p_dp(Q1_tmp.to(dtype), p_const, p_power, min_Q)
        F1_tmp = df1_tmp - p1 - u1_tmp * dp1

        # L1 and Lhat1 based on u1-u0
        L1_tmp, Lhat1_tmp = compute_L_Lhat(u1_tmp, u0, F1_tmp, F0, Ft1_tmp)

        # Stop when conditions satisfied
        if (a0 <= (C_hat / Lhat1_tmp)) and (a0 <= (min(C, 1.0 / alpha) / L1_tmp)):
            u1, df1, F1, Ft1, Q1, L1, Lhat1 = u1_tmp, df1_tmp, F1_tmp, Ft1_tmp, Q1_tmp, L1_tmp, Lhat1_tmp
            break

        i_ls += 1

    # while loop (try to increase a0 a bit, still safe)
    while True:
        if not (
            (a0 < (C_hat / (alpha * Lhat1)))
            and (a0 < (min(C, 1.0 / alpha) / (alpha * L1)))
        ):
            break

        a0_candidate = a0 * alpha

        u1_tmp = torch.clamp(u0 - (a0_candidate * normalizer) * F0, min=0.0)
        df1_tmp = _df_cost(u1_tmp, c, L_pow, inv_beta_op)
        Ft1_tmp, Q1_tmp = compute_F_tilde_next(u1_tmp, u0, df0, Q0)
        p1, dp1 = _demand_p_dp(Q1_tmp.to(dtype), p_const, p_power, min_Q)
        F1_tmp = df1_tmp - p1 - u1_tmp * dp1
        L1_tmp, Lhat1_tmp = compute_L_Lhat(u1_tmp, u0, F1_tmp, F0, Ft1_tmp)

        # accept increase
        a0 = a0_candidate
        u1, df1, F1, Ft1, Q1, L1, Lhat1 = u1_tmp, df1_tmp, F1_tmp, Ft1_tmp, Q1_tmp, L1_tmp, Lhat1_tmp

    # Set initial states after line search
    # We treat u1 as the end of the first "epoch" (k = m).
    u_prev = u0
    u = u1
    v = u0.clone()  # v0
    v_is_updated = False  # Fix D: v is not yet updated to v_k until the first main-loop step
    df = df1
    F_prev = F0
    F = F1
    F_tilde_prev = F0
    F_tilde = Ft1
    Q = Q1.to(reduce_dtype)

    # Step sizes: a_{-1} = a0, a_0 = a0
    a_km2 = a0
    a_km1 = a0

    # omega_0 (stored as omega_prev for ratio_bar in first epoch after init)
    if strong_convexity:
        omega_prev = (1.0 + rho * beta_alg * mu * a0) / (1.0 + mu * a0)
    else:
        omega_prev = 1.0

    # -------------------------
    # Weighted averaged iterate u_hat (kept for reference, but not used for opt_measure)
    # -------------------------
    # Implements the commented averaging scheme:
    #   A += a
    #   u_hat = ((A - a) * u_hat / A) + (a*u / A)
    # A = float(a0)
    # u_hat = u.clone()

    # Iteration counter measured in "block updates" for compatibility with the existing codebase.
    k = m  # after initialization we conceptually completed one full block sweep

    # Initial logging after initialization (k=m)
    if k % (m * exit_criterion.loggingfreq) == 0:
        elapsed_time = time.time() - start_time

        # Keep for reference (do NOT delete):
        # u_hat_full = _gather_full_vector(u_hat, n_local, dist_on, rank, world_size)  # numpy on rank0
        # opt_measure = float(problem.residual(u_hat_full))

        if opt_measure_point == "v":
            v_full = _gather_full_vector(v, n_local, dist_on, rank, world_size)  # numpy on rank0

            if rank == 0:
                opt_measure = float(problem.residual(v_full))
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
                logresult(results, k, elapsed_time, opt_measure, L=L1, L_hat=Lhat1)
                exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)

                if final_phase_enabled and (not final_phase_on) and (opt_measure <= final_phase_trigger):
                    final_phase_on = True
            else:
                exit_flag = False
        else:
            u_full = _gather_full_vector(u, n_local, dist_on, rank, world_size)  # numpy on rank0

            if rank == 0:
                opt_measure = float(problem.residual(u_full))
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
                logresult(results, k, elapsed_time, opt_measure, L=L1, L_hat=Lhat1)
                exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)

                if final_phase_enabled and (not final_phase_on) and (opt_measure <= final_phase_trigger):
                    final_phase_on = True
            else:
                exit_flag = False
    else:
        exit_flag = False

    # Broadcast flags
    if dist_on and world_size > 1:
        # Keep for reference (do NOT delete):
        # flag_t = torch.tensor([1 if exit_flag else 0], device=device, dtype=torch.int64)
        # dist.broadcast(flag_t, src=0)
        # exit_flag = bool(flag_t.item())

        flags_t = torch.tensor(
            [1 if exit_flag else 0, 1 if final_phase_on else 0],
            device=device,
            dtype=torch.int64,
        )
        dist.broadcast(flags_t, src=0)
        exit_flag = bool(flags_t[0].item())
        final_phase_on = bool(flags_t[1].item())

    if exit_flag:
        # IMPORTANT: collective under dist_on, so call on all ranks.
        # Keep for reference (do NOT delete):
        # output_x = _gather_full_vector(u, n_local, dist_on, rank, world_size)

        output_local = v if (return_point == "v" and v_is_updated) else u
        output_x = _gather_full_vector(output_local, n_local, dist_on, rank, world_size)
        return results, output_x

    # Buffer for F_bar
    F_bar = torch.empty_like(u)

    # -------------------------
    # Main loop (each epoch updates all blocks in parallel)
    # -------------------------
    while True:
        # Safety stop on time
        elapsed_time = time.time() - start_time
        if elapsed_time >= exit_criterion.maxtime:
            break
        if k >= exit_criterion.maxiter:
            break

        # Compute L_k and \hat{L}_k at current state (u_k=u, u_{k-1}=u_prev)
        L_val, L_hat = compute_L_Lhat(u, u_prev, F, F_prev, F_tilde)

        # Step-size update (Eq. (3.10) style; uses ratio sqrt(a_{k-1}/a_{k-2}))
        ratio = math.sqrt(a_km1 / a_km2) if a_km2 > 0 else 1.0
        cand1 = rho_0 * a_km1
        cand2 = (C / L_val) * ratio if L_val > 0 else float("inf")
        cand3 = (C_hat / L_hat) * ratio if L_hat > 0 else float("inf")
        a_k = min(cand1, cand2, cand3)

        # Fix E: final-phase step-size decay to damp oscillations near a solution.
        # Activated when (opt_measure <= final_phase_trigger) at a logging step.
        if final_phase_enabled and final_phase_on and (final_phase_decay < 1.0):
            if math.isfinite(final_phase_a_max):
                a_k = min(a_k, final_phase_a_max)
            a_k *= final_phase_decay

        # omega_k for next epoch
        if strong_convexity:
            omega_k = (1.0 + rho * beta_alg * mu * a_k) / (1.0 + mu * a_k)
        else:
            omega_k = 1.0

        # Optional synchronization to ensure bitwise-identical scalars across ranks
        if dist_on and world_size > 1 and sync_step:
            scal = torch.tensor([a_k, omega_k], device=device, dtype=torch.float64)
            dist.broadcast(scal, src=0)
            a_k = float(scal[0].item())
            omega_k = float(scal[1].item())

        # ratio_bar = a_{k-1} * omega_{k-1} / a_k (or a_{k-1}/a_k if not strong convex)
        if strong_convexity:
            ratio_bar = (a_km1 * omega_prev) / a_k
        else:
            ratio_bar = a_km1 / a_k

        # Build F_bar = F_tilde + ratio_bar * (F_prev - F_tilde_prev)
        torch.sub(F_prev, F_tilde_prev, out=F_bar)  # F_bar = F_prev - F_tilde_prev
        F_bar.mul_(ratio_bar).add_(F_tilde)         # F_bar = F_tilde + ratio_bar*(...)

        # v_k = (1-beta) u_k + beta v_{k-1}
        v.mul_(beta_alg).add_(u, alpha=(1.0 - beta_alg))
        v_is_updated = True

        # u_{k+1} = prox( v_k - a_k * normalizer * F_bar )
        u_next = v - (a_k * normalizer) * F_bar
        u_next.clamp_(min=0.0)

        # Compute \tilde{F}_{k+1} and Q_{k+1} (also performs the only per-epoch all_gather)
        # Note: uses df at u (old), and Q at u (old)
        F_tilde_next, Q_next = compute_F_tilde_next(u_next, u, df, Q)

        # Compute F(u_{k+1}) (full operator uses Q_{k+1})
        df_next = _df_cost(u_next, c, L_pow, inv_beta_op)
        p_next, dp_next = _demand_p_dp(Q_next.to(dtype), p_const, p_power, min_Q)
        F_next = df_next - p_next - u_next * dp_next

        # Shift states
        u_prev, u = u, u_next
        df = df_next
        F_prev, F = F, F_next
        F_tilde_prev, F_tilde = F_tilde, F_tilde_next
        Q = Q_next

        # Shift step-size scalars
        a_km2, a_km1 = a_km1, a_k
        omega_prev = omega_k

        # Advance iteration counter by one full sweep (m block updates)
        k += m

        # -------------------------
        # Averaging (u_hat) is kept for reference, but disabled for speed.
        # -------------------------
        # A += float(a_k)
        # w = float(a_k) / A
        # u_hat.mul_(1.0 - w).add_(u, alpha=w)

        # Logging / exit check (requested snippet)
        if k % (m * exit_criterion.loggingfreq) == 0:
            # Compute averaged variables
            # step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_, F, F_, F_tilde)
            # a_ = a
            # a = step
            # A += a
            # u_hat = ((A - a) * u_hat / A) + (a*u / A)
            elapsed_time = time.time() - start_time

            # Keep for reference (do NOT delete):
            # u_full = _gather_full_vector(u, n_local, dist_on, rank, world_size)  # numpy on rank0
            # opt_measure = float(problem.residual(u_full))

            if opt_measure_point == "v":
                # Keep u_full definition for reference (do NOT delete):
                # u_full = _gather_full_vector(u, n_local, dist_on, rank, world_size)  # numpy on rank0

                v_full = _gather_full_vector(v, n_local, dist_on, rank, world_size)  # numpy on rank0

                if rank == 0:
                    opt_measure = float(problem.residual(v_full))
                    logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
                    logresult(results, k, elapsed_time, opt_measure, L=L_val, L_hat=L_hat)
                    exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)

                    if final_phase_enabled and (not final_phase_on) and (opt_measure <= final_phase_trigger):
                        final_phase_on = True
                else:
                    exit_flag = False
            else:
                u_full = _gather_full_vector(u, n_local, dist_on, rank, world_size)  # numpy on rank0

                if rank == 0:
                    opt_measure = float(problem.residual(u_full))
                    logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
                    logresult(results, k, elapsed_time, opt_measure, L=L_val, L_hat=L_hat)
                    exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)

                    if final_phase_enabled and (not final_phase_on) and (opt_measure <= final_phase_trigger):
                        final_phase_on = True
                else:
                    exit_flag = False

            if dist_on and world_size > 1:
                # Keep for reference (do NOT delete):
                # flag_t = torch.tensor([1 if exit_flag else 0], device=device, dtype=torch.int64)
                # dist.broadcast(flag_t, src=0)
                # exit_flag = bool(flag_t.item())

                flags_t = torch.tensor(
                    [1 if exit_flag else 0, 1 if final_phase_on else 0],
                    device=device,
                    dtype=torch.int64,
                )
                dist.broadcast(flags_t, src=0)
                exit_flag = bool(flags_t[0].item())
                final_phase_on = bool(flags_t[1].item())
            if exit_flag:
                break

    # End loop: gather final solution on rank 0
    # IMPORTANT: collective under dist_on, so call on all ranks.
    # Keep for reference (do NOT delete):
    # output_x = _gather_full_vector(u, n_local, dist_on, rank, world_size)

    output_local = v if (return_point == "v" and v_is_updated) else u
    output_x = _gather_full_vector(output_local, n_local, dist_on, rank, world_size)
    return results, output_x
