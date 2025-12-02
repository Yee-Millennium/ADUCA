from dataclasses import dataclass
from typing import Union

@dataclass
class ExitCriterion:
    """
    Defines the exit criterion for each algorithm run.

    Attributes:
        maxiter (int): Maximum number of iterations allowed.
        maxtime (float): Maximum execution time allowed (in seconds).
        targetaccuracy (float): Target accuracy to halt the algorithm.
        loggingfreq (int): Number of data passes between logging events.
    """
    maxiter: int
    maxtime: float
    targetaccuracy: float
    loggingfreq: int

def check_exit_condition(
    exit_criterion: ExitCriterion,
    current_iter: int,
    elapsed_time: float,
    measure: float
) -> bool:
    """
    Check if the given exit criterion has been satisfied.

    Args:
        exit_criterion (ExitCriterion): An instance containing the exit conditions.
        current_iter (int): The current iteration count.
        elapsed_time (float): The elapsed execution time in seconds.
        measure (float): The current accuracy measure.

    Returns:
        bool: True if any of the exit conditions are met; False otherwise.
    """
    if current_iter >= exit_criterion.maxiter:
        return True
    elif elapsed_time >= exit_criterion.maxtime:
        return True
    elif measure <= exit_criterion.targetaccuracy:
        return True
    return False