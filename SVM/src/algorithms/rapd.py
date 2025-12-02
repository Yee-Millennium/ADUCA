import numpy as np
import time
import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult


def elastic_net_function(x, alpha, beta):
    return 0.5 * alpha * np.sum(x * x) + beta * np.sum(np.abs(x))

def _prox_func(_x0, p1, p2):
    if _x0 > p1:
        return p2 * (_x0 - p1)
    elif _x0 < -p1:
        return p2 * (_x0 + p1)
    return 0.0

def elastic_net_prox(x, alpha, beta, weight):
    p1 = weight * beta
    p2 = 1.0 / (1.0 + weight * alpha)
    return _prox_func(x, p1, p2)

def rapd(B, x0, y0, L, epochs, lam1, lam2, exitcriterion):
    # Initialize parameters
    x = x0.copy()
    y = y0.copy()
    theta = 1
    tau = 1.0 / L
    sigma = 1.0 / L
    n, d = B.shape
    dBx = B @ x / n
    s = dBx
    idx_seq = np.arange(d)
    K = epochs * d

    # Initialize logging
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    res = 1 - (B @ x)
    init_optmeasure = np.sum(np.maximum(res, 0.0)) / n + elastic_net_function(x, lam2, lam1)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        # Dual update
        y = np.clip(y + sigma * (s - 1 / n), -1.0, 0.0)

        # Randomly select an index for primal update
        i = np.random.choice(idx_seq)
        pre_xi = x[i]

        # Primal update with elastic net proximal operator
        x[i] = elastic_net_prox(x[i] - tau * (B[:, i].T @ y), lam2, lam1, tau)

        # Update dBx and s
        tmp = B[:, i] * (x[i] - pre_xi)
        dBx += tmp / n
        s = dBx + theta * d * tmp

        iteration += 1

        # Logging and exit condition
        if iteration % (exitcriterion.loggingfreq * d) == 0:
            res = 1 - (B @ x)
            opt_measure = np.sum(np.maximum(res, 0.0)) / n + elastic_net_function(x, lam2, lam1)
            elapsed_time = time.time() - starttime
            logresult(results, iteration, elapsed_time, opt_measure)
            print(f"Epoch: {iteration // d}, Loss: {opt_measure}")
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    return results, x