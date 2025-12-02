"""
Data structure for formalizing a standard-form linear program.
"""

from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc


@dataclass
class StandardLinearProgram:
    """
    A standard linear programming problem.

    Attributes:
        A_T (csc_matrix): The transposed constraint matrix in CSC format.
        b (np.ndarray): The right-hand side vector.
        c (np.ndarray): The objective function coefficients.
        prox (Callable[[np.ndarray, float], np.ndarray]): Proximal operator function.
    """
    A_T: csc_matrix
    b: np.ndarray
    c: np.ndarray
    prox: Callable[[np.ndarray, float], np.ndarray] = field(init=False)

    def __post_init__(self):
        # Ensure A_T is a CSC (Compressed Sparse Column) matrix
        if not isspmatrix_csc(self.A_T):
            self.A_T = csc_matrix(self.A_T)

        # Ensure b and c are NumPy arrays of type float
        # print(f"self.b's type: {type(self.b)}")
        self.b = self.b.toarray().reshape(-1)
        # print(f"b's shape: {self.b.shape}")
        self.b = np.array(self.b, dtype=float)
        self.c = np.array(self.c, dtype=float)

        # Define the proximal operator
        self.prox = self._prox_operator

    @staticmethod
    def _prox_operator(x: np.ndarray, tau: float) -> np.ndarray:
        """
        Proximal operator function.

        Args:
            x (np.ndarray): Input array.
            tau (float): Parameter (unused in this prox).

        Returns:
            np.ndarray: Result after applying the proximal operator.
        """
        return np.maximum(x, 0.0)
    
    