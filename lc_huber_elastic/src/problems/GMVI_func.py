import numpy as np


class GMVIProblem:
    """Generic Minty VI (GMVI) / composite VI wrapper.

    The problem is specified by:
      - a monotone operator F provided by operator_func.func_map
      - a convex, prox-friendly function g provided by g_func.prox_opr

    This mirrors the minimal interface used in the repository's existing
    benchmarks.
    """

    def __init__(self, operator_func, g_func):
        self.n = operator_func.n
        self.operator_func = operator_func
        self.g_func = g_func

    def func_value(self, x: np.ndarray) -> float:
        """Return the composite objective value (if available).

        For some benchmarks (e.g., SVM) this is the primary progress metric.
        For LC-Huber, operator_func.func_value returns h_δ(u) (ignoring the
        constraint), while the experiment scripts typically monitor a projected
        primal gap computed by the torch oracle.
        """

        val = 0.0
        if hasattr(self.operator_func, "func_value"):
            val += float(self.operator_func.func_value(x))
        if hasattr(self.g_func, "func_value"):
            val += float(self.g_func.func_value(x))
        return float(val)

    def residual(self, q: np.ndarray) -> float:
        """Standard proximal residual || q - prox_g(q - F(q)) ||."""

        Fq = self.operator_func.func_map(q)
        z = q - Fq

        # In this repo, g.prox_opr typically expects a stepsize `tau`.
        # We use the standard choice tau=1.0 for a scale-free residual.
        try:
            prox = self.g_func.prox_opr(z, tau=1.0, weights=None)
        except TypeError:
            # Backward compatibility for g implementations that do not accept `weights`.
            prox = self.g_func.prox_opr(z, tau=1.0)

        return float(np.linalg.norm(q - prox))
