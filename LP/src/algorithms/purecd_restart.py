import numpy as np
from scipy.sparse import csc_matrix
import logging
import copy
import random
import time
from math import inf
import sys

from src.algorithms.utils.exitcriterion import ExitCriterion, check_exit_condition
from src.algorithms.utils.results import Results
from src.algorithms.utils.helper import compute_fvaluegap_metricLP, compute_nzrows_for_blocks
from src.problems.standardLP import StandardLinearProgram

# Assuming the following classes are defined as per previous translations:
# - StandardLinearProgram
# - ExitCriterion

def purecd_restart_x_y(
    problem,
    exitcriterion,
    gamma=1.0,
    sigma=0.0,
    R=10,
    blocksize=10,
    restartfreq=inf,
    io=None
):
    """
    Pure Coordinate Descent with Restart.

    Args:
        problem (StandardLinearProgram): The standard linear program.
        exitcriterion (ExitCriterion): The exit criteria.
        gamma (float, optional): Parameter γ. Defaults to 1.0.
        sigma (float, optional): Parameter σ. Defaults to 0.0.
        R (int, optional): Radius parameter. Defaults to 10.
        blocksize (int, optional): Size of blocks for coordinate updates. Defaults to 10.
        restartfreq (float, optional): Frequency for restarting the algorithm. Defaults to inf.
        io (IO stream, optional): Optional I/O stream for flushing outputs. Defaults to None.

    Returns:
        Results: An instance containing logged results.
    """
    # Set up logging
    logger = logging.getLogger('purecd_restart_x_y')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout) if io is None else logging.StreamHandler(io)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Log initial parameters
    logger.info("Running pure_cd_restart_x_y...")
    logger.info(f"blocksize = {blocksize}")
    logger.info(f"γ = {gamma}")
    logger.info(f"σ = {sigma}")
    logger.info(f"R = {R}")
    logger.info(f"restartfreq = {restartfreq}")
    if io is not None:
        io.flush()

    # Extract problem data
    A_T = problem.A_T.tocsc()  # Ensure A_T is in CSC format
    # print(f"!!! b's type: {type(problem.b)}")
    b = problem.b.flatten()  # Convert b to a 1D numpy array
    c = problem.c  # Assuming c is a numpy array
    prox = problem.prox

    # Get dimensions
    d, n = A_T.shape

    # Initialize x0, y0 as zeros
    x0 = np.zeros(d)
    y0 = np.zeros(n)

    # Precomputing blocks, nzrows, sliced_A_T
    start_time_init = time.time()
    blocks, C = compute_nzrows_for_blocks(A_T, blocksize)
    num_nnz = A_T.getnnz(axis=1)  # Number of non-zeros per row
    # print(f"!!! A_T.shape: {A_T.shape}")
    # sliced_A_Ts = []
    # for C_j, blocks_j in zip(C, blocks):
    #     print(f"!!! C_j: {C_j}")
    #     print(f"!!! blocks_j: {blocks_j}")
    #     sliced_A_Ts.append(A_T[C_j, blocks_j])
    # print(f"!!! Type of A_T: {type(A_T)}")
    sliced_A_Ts = [A_T[C_j, blocks_j] for C_j, blocks_j in zip(C, blocks)]
    end_time_init = time.time()
    logger.info(f"Initialization time = {end_time_init - start_time_init:.4f} seconds")

    # Start of PURE_CD
    m = len(blocks)

    # Log initial measure
    starttime = time.time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    logger.info(f"init_metricLP: {init_metricLP}")
    Results.log_result(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 1
    exitflag = False

    while not exitflag:
        # Initialize PURE_CD parameters
        tau = 0.99 / (gamma * R)
        sigma_param = 0.99 * gamma / R  # Avoid naming conflict with function argument 'sigma'
        idx_seq = list(range(m))

        # Initialize variables
        x = copy.deepcopy(x0)
        y = copy.deepcopy(y0)
        x_bar = copy.deepcopy(x0)
        x_tilde = np.zeros_like(x0)
        y_tilde = np.zeros_like(y0)

        z = A_T.dot(y)  # z = A_T * y
        theta_x = np.ones(d, dtype=int)  # Tracking updates for x
        theta_y = np.ones(n, dtype=int)  # Tracking updates for y

        k = 1
        restartflag = False

        while not exitflag and not restartflag:
            # Select a random block j
            j = random.choice(idx_seq)

            # Update x_tilde and y_tilde for the selected block
            # print(f"!!! C[j]: {C[j]}")

            x_tilde[C[j]] += (k - theta_x[C[j]]) * x[C[j]]
            # x_tilde[C[j]] += (k - theta_x[C[j]]) * x[C[j]]
            y_tilde[blocks[j]] += (k - theta_y[blocks[j]]) * y[blocks[j]]

            # Update x_bar for the selected block
            tau_div_num_nnz = tau / num_nnz[C[j]]
            z_plus_c = z[C[j]] + c[C[j]]
            x_bar[C[j]] = x[C[j]] - tau_div_num_nnz * z_plus_c

            # Apply proximal operator
            x_bar[C[j]] = prox(x_bar[C[j]], tau)

            # Compute Delta_y
            sliced_A_T = sliced_A_Ts[j]
            x_bar_sliced = x_bar[C[j]]
            # print(f"!!! x_bar_sliced.shape: {x_bar_sliced.shape}")
            # print(f"!!! sliced_A_T's shape: {sliced_A_T.shape}")
            # print(f"!!! b[blocks[j]].shape: {b[blocks[j]].shape}")
            Delta_y = np.asarray( sigma_param * ((np.dot(sliced_A_T, x_bar_sliced)) - b[blocks[j]]) ).reshape(-1)

            # Update y for the selected block
            # print(f"!!! Delta_y'shape: {Delta_y.shape}")
            # print(f"!!! Delta_y'type: {type(Delta_y)}")
            y[blocks[j]] += Delta_y

            # Compute temporary variable
            tmp = Delta_y @ sliced_A_T  # Result is a sparse matrix
            tmp_dense = np.asarray(tmp).reshape(-1)

            # Update z for the selected block
            z[C[j]] += tmp_dense

            # Update x for the selected block
            x[C[j]] = x_bar[C[j]] - tau * tmp_dense

            # Update theta_x and theta_y
            theta_x[C[j]] = k
            theta_y[blocks[j]] = k

            # Logging and checking exit condition
            if outer_k % (exitcriterion.loggingfreq * m) == 0:
                # Compute averaged variables
                x_tilde_tmp = x_tilde + ((k + 1) - theta_x) * x_bar
                y_tilde_tmp = y_tilde + ((k + 1) - theta_y) * y
                x_out = x_tilde_tmp / k
                y_out = y_tilde_tmp / k

                # Compute progress measures
                fvaluegap, metricLP = compute_fvaluegap_metricLP(x_out, y_out, problem)

                # Compute elapsed time
                elapsedtime = time.time() - starttime
                logger.info(f"elapsedtime: {elapsedtime}")
                logger.info(f"outer_k: {outer_k}, fvaluegap: {fvaluegap}, metricLP: {metricLP}")

                # Log the results
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

                    # Update x0 and y0 for restart
                    x0 = copy.deepcopy(x_out)
                    y0 = copy.deepcopy(y_out)
                    init_fvaluegap = fvaluegap
                    init_metricLP = metricLP
                    restartflag = True
                    break

            # Increment iteration counters
            k += 1
            outer_k += 1

    return results