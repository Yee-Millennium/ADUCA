import numpy as np


class Results:
    def __init__(self):
        self.iterations = []
        self.times = []
        self.optmeasures = []
        self.L = []
        self.L_hat = []


def logresult(results: Results, current_iter, elapsed_time, opt_measure, L_hat=None, L=None):
    """Append execution measures to Results."""
    results.iterations.append(current_iter)
    results.times.append(elapsed_time)
    results.optmeasures.append(opt_measure)
    if L is not None:
        results.L.append(L)
    if L_hat is not None:
        results.L_hat.append(L_hat)
    return
