import numpy as np
from scipy.sparse import csc_matrix
import logging
import copy
import time
from math import inf
import sys
from src.algorithms.utils.exitcriterion import ExitCriterion, check_exit_condition
from src.algorithms.utils.results import Results
from src.algorithms.utils.helper import compute_fvaluegap_metricLP, compute_nzrows_for_blocks
from src.problems.standardLP import StandardLinearProgram

def pdhg_restart_x_y(
    problem:StandardLinearProgram,
    exitcriterion:ExitCriterion,
    gamma=1.0,
    L=10,
    restartfreq=inf,
    io=None
):
    """
    Primal-Dual Hybrid Gradient (PDHG) Algorithm with Restart.

    Args:
        problem (StandardLinearProgram): The standard linear program.
        exitcriterion (ExitCriterion): The exit criteria.
        gamma (float, optional): Step size parameter γ. Defaults to 1.0.
        L (float, optional): Lipschitz constant or related parameter. Defaults to 10.
        restartfreq (float, optional): Frequency for restarting the algorithm. Defaults to inf.
        io (IO stream, optional): Optional I/O stream for flushing outputs. Defaults to None.

    Returns:
        Results: An instance containing logged results.
    """
    # Set up logging
    logger = logging.getLogger('pdhg_restart_x_y')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout) if io is None else logging.StreamHandler(io)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info("Running pdhg_restart_x_y...")
    logger.info(f"γ = {gamma}")
    logger.info(f"L = {L}")
    logger.info(f"restartfreq = {restartfreq}")
    if io is not None:
        io.flush()

    # Extract problem data
    A_T = problem.A_T.tocsc()  # Ensure A_T is in CSC format
    b = problem.b.flatten()  # Convert b to a 1D numpy array
    c = problem.c  # Assuming c is a numpy array

    d, n = A_T.shape
    x0 = np.zeros(d)
    y0 = np.zeros(n)
    m = 1  # Assuming m is set to 1 as in Julia code

    ##### Start of pdhg_restart_x_y

    # Log initial measure
    starttime = time.time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    Results.log_result(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 1
    exitflag = False

    while not exitflag:
        # Initialize PDHG parameters
        tau = 1.0 / (gamma * L)
        sigma = 1.0 * gamma / L
        x_bar = copy.deepcopy(x0)
        x = copy.deepcopy(x0)
        y = copy.deepcopy(y0)
        x_tilde = np.zeros_like(x0)
        y_tilde = np.zeros_like(y0)

        k = 1
        restartflag = False

        while not exitflag and not restartflag:
            # Update y
            # print(f"!!! A_T.shape: {A_T.shape}")
            # print(f"!!! x_bar.shape: {x_bar.shape}")
            y = y + sigma * (x_bar @ A_T - b)

            # Store previous x
            x_pre = copy.deepcopy(x)

            # Update x
            x = x - tau * (A_T.dot(y) + c)
            x = problem.prox(x, tau)

            # Update x_bar
            x_bar = 2 * x - x_pre  # Correct term from original paper

            # Update tilde variables
            x_tilde = x_tilde + x
            y_tilde = y_tilde + y

            # Increment counters
            if outer_k % exitcriterion.loggingfreq == 0:
                # Compute averaged variables
                x_out = x_tilde / k
                y_out = y_tilde / k

                # Compute progress measures
                fvaluegap, metricLP = compute_fvaluegap_metricLP(x_out, y_out, problem)

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
                if k >= restartfreq * m or (restartfreq == inf and metricLP <= 0.5 * init_metricLP):
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

            # Update iteration counters
            k += 1
            outer_k += 1

    return results