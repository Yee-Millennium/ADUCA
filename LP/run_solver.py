#!/usr/bin/env python3
"""
Script for executing Gurobi on selected datasets for the DRO problem with Wasserstein metric-based ambiguity sets.

Command line usage:
    python run_solver.py <dataset> <method>

Example:
    python run_solver.py a1a 0
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import numpy as np
from scipy.sparse import csc_matrix
from gurobipy import Model, GRB, GurobiError

# Import translated modules and functions
from src.problems.standardLP import StandardLinearProgram
from src.problems.dro.utils.libsvm_parser import read_libsvm_into_yXT_sparse
from src.problems.dro.wasserstein import droreformuation_wmetric_hinge_standardformnormalized

# Define dataset information
DATASET_INFO = {
    "a1a": (123, 1605),
    "a9a": (123, 32561),
    "gisette": (5000, 6000),
    "news20": (1355191, 19996),
    "rcv1": (47236, 20242),
}

def setup_logging(outputdir: str, dataset: str, method: int) -> None:
    """
    Set up logging to a timestamped file and the console.

    Args:
        outputdir (str): Directory where log files will be stored.
        dataset (str): Name of the dataset.
        method (int): Gurobi method identifier.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # Trim microseconds to milliseconds
    logging_filename = os.path.join(outputdir, f"{dataset}-method={method}-solver_log-{timestamp}.txt")

    # Create output directory if it doesn't exist
    os.makedirs(outputdir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logging_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"Logging initialized. Log file: {logging_filename}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Execute Gurobi on selected datasets for the DRO problem with Wasserstein metric-based ambiguity sets.")
    parser.add_argument('dataset', type=str, help='Name of the dataset (e.g., a1a, a9a, gisette, news20, rcv1)')
    parser.add_argument('method', type=int, choices=[-1, 0, 1, 2, 3, 4, 5],
                        help='Gurobi method: -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.')

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    dataset = args.dataset
    gurobimethod = args.method

    # Set up logging
    outputdir = "./solver_results/"
    setup_logging(outputdir, dataset, gurobimethod)

    # Validate dataset name
    if dataset not in DATASET_INFO:
        logging.error(f"Invalid dataset name '{dataset}'. Available datasets: {list(DATASET_INFO.keys())}")
        sys.exit(1)

    dim_dataset, num_dataset = DATASET_INFO[dataset]
    filepath = os.path.join("data", f"{dataset}.txt")

    # Check if dataset file exists
    if not os.path.isfile(filepath):
        logging.error(f"Dataset file '{filepath}' not found.")
        sys.exit(1)

    # Problem instance parameters
    kappa = 0.1
    rho = 10.0

    # Problem instance instantiation
    try:
        logging.info("Reading dataset...")
        yX_T = read_libsvm_into_yXT_sparse(filepath, dim_dataset, num_dataset)
        logging.info("Reformulating DRO problem...")
        A_T, b, c = droreformuation_wmetric_hinge_standardformnormalized(yX_T, kappa, rho)
        problem = StandardLinearProgram(A_T, b, c)
    except Exception as e:
        logging.error(f"Error during problem instantiation: {e}")
        sys.exit(1)

    # Setting up the Gurobi model
    try:
        logging.info("Setting up LP...")
        starttime_setup = datetime.now()

        # Initialize Gurobi model
        vector_model = Model("DRO_Wasserstein_LP")

        # Set Gurobi parameters
        vector_model.setParam("Threads", 1)
        vector_model.setParam("Method", gurobimethod)

        # Add variables
        num_vars = A_T.shape[0]
        x = vector_model.addVars(num_vars, lb=0.0, name="x")

        # Add constraints: A_T' * x == b
        # Since A_T is a sparse matrix in CSC format, A_T' is CSR format (row-wise)
        logging.info("Adding constraints...")
        for i in range(A_T.shape[1]):
            # Extract the i-th row of A_T (which is the i-th column of A)
            start_ptr = A_T.indptr[i]
            end_ptr = A_T.indptr[i + 1]
            indices = A_T.indices[start_ptr:end_ptr]
            data = A_T.data[start_ptr:end_ptr]
            # Construct the constraint: sum(A_T[i, j] * x[j] for j in non-zero A_T[i, j]) == b[i]
            expr = sum(data[j] * x[indices[j]] for j in range(len(indices)))
            vector_model.addConstr(expr == b[i], name=f"c_{i}")

        # Set objective: Minimize c' * x
        logging.info("Setting objective...")
        objective = sum(c[j] * x[j] for j in range(num_vars))
        vector_model.setObjective(objective, GRB.MINIMIZE)

        # Set up complete
        endtime_setup = datetime.now()
        setup_duration = (endtime_setup - starttime_setup).total_seconds()
        logging.info(f"=====> Setting up: {setup_duration:.4f} seconds")
    except GurobiError as e:
        logging.error(f"Gurobi Error during model setup: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during model setup: {e}")
        sys.exit(1)

    # Solving the LP
    try:
        logging.info("Solving LP...")
        starttime_solve = datetime.now()
        vector_model.optimize()
        endtime_solve = datetime.now()
        solve_duration = (endtime_solve - starttime_solve).total_seconds()
        logging.info(f"=====> Solve time: {solve_duration:.4f} seconds")
    except GurobiError as e:
        logging.error(f"Gurobi Error during optimization: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during optimization: {e}")
        sys.exit(1)

    # Retrieve and print results
    try:
        if vector_model.status == GRB.OPTIMAL:
            obj_val = vector_model.objVal
            logging.info(f"Objective Value: {obj_val}")

            # Gurobi does not have a direct 'primal_status', but we can interpret the status
            termination_status = vector_model.status
            if termination_status == GRB.OPTIMAL:
                primal_status = "Optimal"
            elif termination_status == GRB.INFEASIBLE:
                primal_status = "Infeasible"
            elif termination_status == GRB.UNBOUNDED:
                primal_status = "Unbounded"
            else:
                primal_status = "Other"

            logging.info(f"Termination Status: {termination_status}")
            logging.info(f"Primal Status: {primal_status}")

            # Optionally, you can retrieve the solution vector
            # x_values = vector_model.getAttr('X', x)
            # logging.info(f"Solution Vector x: {x_values}")
        else:
            logging.warning(f"Optimization was stopped with status {vector_model.status}")
    except GurobiError as e:
        logging.error(f"Gurobi Error when retrieving results: {e}")
    except Exception as e:
        logging.error(f"Unexpected error when retrieving results: {e}")

if __name__ == "__main__":
    main()