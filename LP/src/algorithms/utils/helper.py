"""
File containing helper functions for algorithms.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, find
from dataclasses import dataclass, field
from src.algorithms.utils.results import Results
from src.problems.standardLP import StandardLinearProgram

def compute_nzrows_for_blocks(A_T: csc_matrix, blocksize: int) -> Tuple[List[range], List[np.ndarray]]:
    """
    Compute the nonzero rows in A_T for each block.

    Args:
        A_T (csc_matrix): The transposed sparse matrix in CSC format.
        blocksize (int): The size of each block.

    Returns:
        Tuple[List[range], List[np.ndarray]]: A tuple containing a list of block ranges and a list of arrays with non-zero row indices for each block.
    """
    n_cols = A_T.shape[1]
    blocks = []
    C = []

    # Iterate over column indices in steps of blocksize
    for start in range(0, n_cols, blocksize):
        end = min(start + blocksize, n_cols)  # Ensure the end doesn't exceed total columns
        block_range = range(start, end)
        blocks.append(block_range)
        
        row_set = set()
        
        # Iterate through each column in the current block
        for j in block_range:
            start_ptr = A_T.indptr[j]
            end_ptr = A_T.indptr[j + 1]
            row_indices = A_T.indices[start_ptr:end_ptr]
            row_set.update(row_indices)
        
        # Convert the set of row indices to a sorted NumPy array
        row_vec = np.array(sorted(row_set))
        C.append(row_vec)
    
    return blocks, C

    # len_b = n_cols // blocksize
    # # print(f"!!! len_b: {len_b}")
    # for i in range(0, len_b):
    #     if i == len_b - 1:
    #         block_range = range((len_b - 1) * blocksize, n_cols)
    #     else:
    #         block_range = range((i - 1) * blocksize, i * blocksize)
    #     blocks.append(block_range)

    #     row_set = set()
    #     # print(f"!!! A_T.indptr.shape: {A_T.indptr.shape}")
    #     for j in block_range:
    #         # In Python, column indices are 0-based
    #         start_ptr = A_T.indptr[j]
    #         end_ptr = A_T.indptr[j + 1]
    #         row_indices = A_T.indices[start_ptr:end_ptr]
    #         # print(f"row_indices: {row_indices}")
    #         row_set.update(row_indices) 
    #     row_vec = np.array(sorted(row_set))
    #     C.append(row_vec)
    # print(f"!!! C[0]: {C[0]}")

    # return blocks, C


def export_results_to_csv(results: Results, outputfile: str) -> None:
    """
    Export results into a CSV formatted file.

    Args:
        results (Results): The Results object containing execution data.
        outputfile (str): The path to the output CSV file.
    """
    df = pd.DataFrame({
        'iterations': results.iterations,
        'times': results.times,
        'fvaluegaps': results.fvaluegaps,
        'metricLPs': results.metricLPs
    })
    df.to_csv(outputfile, index=False)

def compute_fvaluegap_metricLP(x_out: np.ndarray, y_out: np.ndarray, problem: StandardLinearProgram) -> Tuple[float, float]:
    """
    Compute a common metric for LP. See Eqn (20) in Applegate et al 2020.

    Args:
        x_out (np.ndarray): Solution vector x.
        y_out (np.ndarray): Solution vector y.
        problem (StandardLinearProgram): The LP problem instance.

    Returns:
        Tuple[float, float]: The value of norm5 and the combined metric.
    """
    A_T = problem.A_T
    b = problem.b
    c = problem.c

    # Compute (x_out' * A_T)' which is equivalent to A * x_out in Julia
    A_x = A_T.transpose().dot(x_out)  # Shape should match b

    norm1 = np.linalg.norm(np.maximum(-x_out, 0))
    norm2 = np.linalg.norm(np.maximum(A_x - b, 0))
    norm3 = np.linalg.norm(np.maximum(-A_x + b, 0))
    norm4 = np.linalg.norm(np.maximum(-A_T.dot(y_out) - c, 0))
    # print(f"c's shape: {c.shape}")
    # print(f"b's shape: {b.shape}")
    norm5 = np.linalg.norm(c.dot(x_out) + b.dot(y_out))

    combined_metric = np.sqrt(norm1**2 + norm2**2 + norm3**2 + norm4**2 + norm5**2)
    return norm5, combined_metric


def nnz_per_row(A_T: csc_matrix) -> np.ndarray:
    """
    Compute number of nonzero elements of each row in a sparse column matrix.

    Args:
        A_T (csc_matrix): The transposed sparse matrix in CSC format.

    Returns:
        np.ndarray: An array where each element represents the number of non-zero elements in the corresponding row.
    """
    # Convert to CSR format for efficient row operations
    A_csr = A_T.transpose().tocsr()
    return A_csr.getnnz(axis=1)