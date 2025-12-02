import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import vstack, hstack, diags, eye, block_diag
import logging
import copy
import random
import time
import sys
from math import inf

from src.algorithms.utils.exitcriterion import ExitCriterion, check_exit_condition
from src.algorithms.utils.results import Results
from src.algorithms.utils.helper import compute_fvaluegap_metricLP, compute_nzrows_for_blocks
from src.problems.standardLP import StandardLinearProgram

# Placeholder implementations for helper functions and classes.
# These should be replaced with actual implementations.


def clvr_lazy_restart_x_y(
    problem:StandardLinearProgram,
    exitcriterion:ExitCriterion,
    gamma=1.0,
    sigma=0.0,
    R=10,
    blocksize=10,
    restartfreq=inf,
    io=None
):
    """
    Coordinate Linear Variance Reduction (Lazy Update Version) with Restart.

    Args:
        problem (StandardLinearProgram): The standard linear program.
        exitcriterion (ExitCriterion): The exit criteria.
        gamma (float, optional): Step size parameter. Defaults to 1.0.
        sigma (float, optional): Additional parameter. Defaults to 0.0.
        R (int, optional): Radius parameter for variance reduction. Defaults to 10.
        blocksize (int, optional): Size of blocks for coordinate updates. Defaults to 10.
        restartfreq (float, optional): Frequency for restarting the algorithm. Defaults to inf.
        io (IO stream, optional): Optional I/O stream for flushing outputs. Defaults to None.

    Returns:
        Results: An instance containing logged results.
    """
    # Set up logging
    logger = logging.getLogger('clvr_lazy_restart_x_y')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout) if io is None else logging.StreamHandler(io)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.info("Running clvr_lazy_restart_x_y with")
    logger.info(f"blocksize = {blocksize}")
    logger.info(f"γ = {gamma}")
    logger.info(f"σ = {sigma}")
    logger.info(f"R = {R}")
    logger.info(f"restartfreq = {restartfreq}")
    if io is not None:
        io.flush()
    
    # Algorithm 2 from the paper

    A_T = problem.A_T.tocsc()
    b = problem.b.flatten()
    c = problem.c  # Assuming c is a numpy array

    d, n = A_T.shape
    x0 = np.zeros(d)
    y0 = np.zeros(n)

    time_start_initialization = time.time()

    # Precomputing blocks, nzrows, sliced_A_T
    blocks, C = compute_nzrows_for_blocks(A_T, blocksize)
    sliced_A_Ts = []
    for j in range(len(C)):
        # Extract rows corresponding to C[j] and columns corresponding to blocks[j]
        sliced = A_T[C[j], blocks[j]]
        sliced_A_Ts.append(sliced)

    time_end_initialization = time.time()
    logger.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

    ##### Start of clvr_lazy_restart_x_y

    m = len(blocks)

    # Log initial measure
    starttime = time.time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    Results.log_result(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 1
    exitflag = False

    while not exitflag:
        # Init of CLVR Lazy
        a = 1.0 / (R * m)
        pre_a = a
        idx_seq = list(range(m))

        x = copy.deepcopy(x0)
        y = copy.deepcopy(y0)
        x_tilde = np.zeros_like(x0)
        y_tilde = np.zeros_like(y0)
        z = A_T.dot(y) + c  # z = A_T * y + c
        q = a * np.copy(z)
        theta_x = np.ones(len(x0), dtype=int)
        theta_y = np.ones(len(y0), dtype=int)

        k = 1
        restartflag = False
        while not exitflag and not restartflag:
            # Line 4: Select a random block
            j = random.choice(idx_seq)

            # Slice of variables based on nzrowsC[j]
            z_sliced = z[C[j]]
            q_sliced = q[C[j]]
            Adelta_sliced = a * ((k-1) - theta_x[C[j]])

            # Line 5: Update q_hat
            q_hat = q_sliced + Adelta_sliced * z_sliced

            # Line 6: Update x_hat using proximal operator
            x_hat = problem.prox(x0[C[j]] - 1 / gamma * q_hat , (1.0 / gamma) * a * k)

            # Line 7 & 12: Update y
            sliced_A_T = sliced_A_Ts[j]
            
            # print(f"!!! x_hat's shape: {x_hat.shape}")
            # print(f"!!! sliced_A_T's shape: {sliced_A_T.shape}")
            Delta_y = np.asarray(gamma * m * a * (sliced_A_T @ x_hat - b[blocks[j]])).reshape(-1)
            # print(f"!!! Delta_y.shape: {Delta_y.shape}")


            y_tilde[blocks[j]] += a * (k - theta_y[blocks[j]]) * y[blocks[j]] + (m - 1) * a * Delta_y
            y[blocks[j]] += Delta_y

            # Line 10: Update q
            Delta_Delta_y = np.asarray((Delta_y * sliced_A_T)).reshape(-1)
            q[C[j]] = q_sliced + a * (k + 1 - theta_x[C[j]]) * z_sliced + (m + 1) * a * Delta_Delta_y

            # Line 11: Update x_tilde
            x_tilde[C[j]] += a * (k - theta_x[C[j]]) * x[C[j]]

            # Line 9: Update z
            z[C[j]] += Delta_Delta_y

            # Line 13: Update x using proximal operator
            x[C[j]] = problem.prox(x0[C[j]] - (1.0 / gamma) * q[C[j]], (1.0 / gamma) * (k + 1) * a)

            # Line 14: Update theta_x and theta_y
            theta_x[C[j]] = k
            theta_y[blocks[j]] = k

            # Logging and checking exit condition
            if outer_k % (exitcriterion.loggingfreq * m) == 0:
                x_tilde_tmp = x_tilde + a * (k + 1 - theta_x) * x
                y_tilde_tmp = y_tilde + a * (k + 1 - theta_y) * y
                x_out = x_tilde_tmp / (a * k)
                y_out = y_tilde_tmp / (a * k)

                # Progress measures
                fvaluegap, metricLP = compute_fvaluegap_metricLP(x_out, y_out, problem)

                elapsedtime = time.time() - starttime
                logger.info(f"elapsedtime: {elapsedtime}")
                logger.info(f"outer_k: {outer_k}, fvaluegap: {fvaluegap}, metricLP: {metricLP}")
                Results.log_result(results, outer_k, elapsedtime, fvaluegap, metricLP)

                if io is not None:
                    io.flush()

                # Check exit conditions
                exitflag = check_exit_condition(exitcriterion, outer_k, elapsedtime, metricLP)
                if exitflag:
                    break

                # Check restart conditions
                if (k >= restartfreq * m) or (restartfreq == inf and metricLP <= 0.5 * init_metricLP):
                    logger.info("<===== RESTARTING")
                    logger.info(f"k ÷ m: {k / m}")
                    logger.info(f"elapsedtime: {elapsedtime}")
                    logger.info(f"outer_k: {outer_k}, fvaluegap: {fvaluegap}, metricLP: {metricLP}")
                    if io is not None:
                        io.flush()

                    x0 = copy.deepcopy(x_out)
                    y0 = copy.deepcopy(y_out)
                    init_fvaluegap = fvaluegap
                    init_metricLP = metricLP
                    restartflag = True
                    break

            k += 1
            outer_k += 1

    return results