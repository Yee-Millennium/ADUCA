import numpy as np
import time
import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult

class CODERVRParams:
    def __init__(self, L, M, gamma, K):
        self.L = L
        self.M = M
        self.gamma = gamma
        self.K = K

def codervr(problem, exitcriterion, parameters, x0=None):
    # Initialize parameters and variables
    L, M, gamma, K = parameters.L, parameters.M, parameters.gamma, parameters.K
    a, A = 0.0, 0.0
    beta = 2 * M / np.sqrt(K)
    x0 = np.zeros(problem.d) if x0 is None else x0

    # Initialize variables
    x_hat, x_hat_prev = x0.copy(), x0.copy()
    x0_0, x0_prev = x0.copy(), x0.copy()
    x_out = x0.copy()
    y0_0, y0_prev = x0.copy(), x0.copy()
    z0_0, z0_prev, q0_0 = np.zeros(problem.d), np.zeros(problem.d), np.zeros(problem.d)
    x_hat_sum, x_tilde_sum, x_out_sum = np.zeros(problem.d), np.zeros(problem.d), np.zeros(problem.d)
    m = problem.d
    n = range(problem.operator_func.n)

    # Initialize logging
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.func_value(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        x_hat_prev[:] = x_hat
        x_hat_sum.fill(0)
        x_tilde_sum.fill(0)

        # Step 4: Calculate full gradient
        mu = problem.operator_func.func_map(x_hat_prev)

        # Step 5: Update steps a and A
        a_prev = a
        A_prev = A
        if a_prev == 0.0:
            a = min(np.sqrt(K) / (8 * M), K / (8 * L))
        else:
            a = min((1 + gamma / beta) * a_prev, (1 + A_prev * gamma) * min(np.sqrt(K) / (8 * M), K / (8 * L)))
        A = A_prev + a

        # Loop over K iterations
        for k in range(K):
            y0_prev[:] = x0_prev
            x0_prev[:] = x0_0
            z0_prev[:] = z0_0
            y0_0[:] = x0_prev

            for j in range(m):
                # Step 8: Updating y0 based on x values
                if j >= 1:
                    y0_0[j - 1] = x0_0[j - 1]
                    y0_prev[j - 1] = x0_prev[j - 1]

                # Step 9: Sample random index t
                t = np.random.choice(n)

                # Step 10: Compute q
                F = problem.operator_func.func_map_block_sample
                a0_prev = a_prev if j == 0 else a
                q0_0[j] = (F(j + 1, t+1, y0_0) - F(j + 1, t+1, x_hat_prev) + mu[j] +
                           (a0_prev / a) * (F(j + 1, t+1, x0_prev) - F(j + 1, t+1, y0_prev)) +
                           beta * (x0_prev[j] - x_hat_prev[j]))

                # Step 11
                z0_0[j] = z0_prev[j] + a * q0_0[j]

                # Step 12: Proximal operator update
                x0_0[j] = problem.g_func.prox_opr_block(j + 1, x0[j] - z0_0[j] / K, A_prev + a * (k + 1) / K)

            # Accumulate sums for averages
            x_hat_sum += beta / (beta + gamma) * x0_prev + gamma / (beta + gamma) * x0_0
            x_tilde_sum += x0_0

        # Step 16: Compute averages for final updates
        x_hat = x_hat_sum / K
        x_tilde = x_tilde_sum / K
        x_out_sum += a * x_tilde
        iteration += m * K

        # Logging and exit condition check
        if iteration % (m * problem.operator_func.n * exitcriterion.loggingfreq) == 0:
            x_out = x_out_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x_out)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)
    # x_out
    return results, x0_0