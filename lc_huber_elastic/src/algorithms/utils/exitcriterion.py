import math


class ExitCriterion:
    """Lightweight stopping criteria container.

    This mirrors the small utility used in the other benchmark folders in the
    repository (SVM, Nash_Equilibrium, mm_cournot).
    """

    def __init__(self, maxiter, maxtime, targetaccuracy, loggingfreq):
        self.maxiter = maxiter            # Max #iterations allowed
        self.maxtime = maxtime            # Max execution time allowed
        self.targetaccuracy = targetaccuracy  # Target accuracy to halt algorithm
        self.loggingfreq = loggingfreq    # #datapass between logging


def CheckExitCondition(exitcriterion, currentiter, elapsedtime, measure):
    """Return True if any stopping criterion is met."""
    if currentiter >= exitcriterion.maxiter:
        return True
    elif elapsedtime >= exitcriterion.maxtime:
        return True
    elif measure <= exitcriterion.targetaccuracy:
        return True
    elif math.isnan(measure):
        return True

    return False
