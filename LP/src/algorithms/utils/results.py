from dataclasses import dataclass, field
from typing import List


@dataclass
class Results:
    """
    Defines the progress of execution at each logging step.

    Attributes:
        iterations (List[float]): Number of iterations elapsed.
        times (List[float]): Elapsed times since start of execution (in seconds).
        fvaluegaps (List[float]): Primal and dual objective value gaps.
        metricLPs (List[float]): The computed values of the LP metric.
    """
    iterations: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    fvaluegaps: List[float] = field(default_factory=list)
    metricLPs: List[float] = field(default_factory=list)

    def log_result(self, current_iter: float, elapsed_time: float, fvalue_gap: float, metric_lp: float) -> None:
        """
        Append execution measures to Results.

        Args:
            current_iter (float): Current iteration number.
            elapsed_time (float): Elapsed time since start of execution (in seconds).
            fvalue_gap (float): Primal and dual objective value gaps.
            metric_lp (float): The computed value of the LP metric.
        """
        self.iterations.append(current_iter)
        self.times.append(elapsed_time)
        self.fvaluegaps.append(fvalue_gap)
        self.metricLPs.append(metric_lp)