"""
Script for executing algorithms on selected datasets for the DRO problem with Wasserstein metric-based ambiguity sets.

Command line usage:
    python run_algo.py <dataset> <gamma> <algo1> <algo2> ... <algoN>

Example:
    python run_algo.py a1a 0.5 0 1 2
"""

import sys
import os
import numpy as np
import logging
from datetime import datetime
from math import sqrt, inf

from scipy.sparse.linalg import svds

# Import translated modules and functions
from src.problems.standardLP import StandardLinearProgram
from src.algorithms.utils.exitcriterion import ExitCriterion
from src.algorithms.utils.results import Results
from src.algorithms.utils.helper import export_results_to_csv
from src.problems.dro.utils.libsvm_parser import read_libsvm_into_yXT_sparse
from src.problems.dro.wasserstein import droreformuation_wmetric_hinge_standardformnormalized

from src.algorithms.clvr_lazy_restart import clvr_lazy_restart_x_y
from src.algorithms.pdhg_restart import pdhg_restart_x_y
from src.algorithms.spdhg_restart import spdhg_restart_x_y
from src.algorithms.purecd_restart import purecd_restart_x_y
from src.algorithms.aduca import aduca, aduca_pdhg

# Set BLAS threads to 1 (similar to Julia's BLAS.set_num_threads(1))
os.environ['OMP_NUM_THREADS'] = '1'

## (Dimension, Sample size)
DATASET_INFO = { 
    "sonar_scale": (60, 208),
    "a1a": (123, 1605),
    "a9a": (123, 32561),
    "gisette_scale": (5000, 6000),
    "news20": (1355191, 19996),
    "rcv1": (47236, 20242),
    "phishing": (68, 11055),
    "colon-cancer": (2000, 62),
    "madelon": (500, 2000),
    "mushrooms": (112, 8124),
    "skin_nonskin": (3, 245057),
    "SUSY": (18,5000000),
    "epsilon_normalized": (2000,400000),
    "HIGGS": (28, 11000000),
    "ijcnn1": (22, 49990),
    "w1a": (300, 2477),
    "w7a":(300, 24692),
    "w8a": (300, 49749),
    "covtype":(54, 581012)
}

def setup_logging(outputdir: str, dataset: str, algo_indices: list) -> str:
    """
    Set up logging to a timestamped file.

    Args:
        outputdir (str): Directory where log files will be stored.
        dataset (str): Name of the dataset.
        algo_indices (list): List of algorithm indices as strings.

    Returns:
        str: Path to the logging file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # Trim microseconds to milliseconds
    algo_names = "_".join(algo_indices)
    logging_filename = os.path.join(outputdir, f"{dataset}-{algo_names}-execution_log-{timestamp}.txt")

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
    return logging_filename

def execute_algorithm(algo: str, params: dict, outputdir: str, dataset: str):
    """
    Executes the specified algorithm with given parameters and exports the results.

    Args:
        algo (str): Algorithm index as a string.
        params (dict): Dictionary containing algorithm-specific parameters.
        outputdir (str): Directory to store output results.
        dataset (str): Name of the dataset.
    """
    if algo == "0":
        logging.info("========================================")
        logging.info("Running clvr_lazy_restart_x_y with blocksize=1.")

        logging.info("clvr_blocksize = 1")
        logging.info("clvr_R = 1.0")

        r_clvr_lazy_restart = clvr_lazy_restart_x_y(
            problem=params['problem'],
            exitcriterion=params['exitcriterion'],
            blocksize=1,
            R=1.0,
            gamma=params['gamma'],
            restartfreq=params['restartfreq'],
        )
        export_filename = os.path.join(
            outputdir,
            f"{dataset}-clvr_lazy_restart_x_y-blocksize=1-R=1-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv"
        )
        export_results_to_csv(r_clvr_lazy_restart, export_filename)
        logging.info(f"Results exported to '{export_filename}'.")
        logging.info("========================================")

    elif algo == "1":
        logging.info("========================================")
        logging.info("Running clvr_lazy_restart_x_y.")

        clvr_R_multiplier = 1.0
        logging.info(f"clvr_R_multiplier = {clvr_R_multiplier}")

        r_clvr_lazy_restart = clvr_lazy_restart_x_y(
            problem=params['problem'],
            exitcriterion=params['exitcriterion'],
            blocksize=params['blocksize'],
            R=params['R'] * clvr_R_multiplier,
            gamma=params['gamma'],
            restartfreq=params['restartfreq'],
        )
        export_filename = os.path.join(
            outputdir,
            f"{dataset}-clvr_lazy_restart_x_y-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv"
        )
        export_results_to_csv(r_clvr_lazy_restart, export_filename)
        logging.info(f"Results exported to '{export_filename}'.")
        logging.info("========================================")

    elif algo == "2":
        logging.info("========================================")
        logging.info("Running pdhg_restart_x_y.")

        pdhg_L_multiplier = 1.0
        logging.info(f"pdhg_L_multiplier = {pdhg_L_multiplier}")

        r_pdhg_restart = pdhg_restart_x_y(
            problem=params['problem'],
            exitcriterion=params['exitcriterion'],
            L=params['L'] * pdhg_L_multiplier,
            gamma=params['gamma'],
            restartfreq=params['restartfreq'],
        )
        export_filename = os.path.join(
            outputdir,
            f"{dataset}-pdhg_restart_x_y-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv"
        )
        export_results_to_csv(r_pdhg_restart, export_filename)
        logging.info(f"Results exported to '{export_filename}'.")
        logging.info("========================================")

    elif algo == "3":
        logging.info("========================================")
        logging.info("Running spdhg_restart_x_y.")

        spdhg_R_multiplier = 1.0
        logging.info(f"spdhg_R_multiplier = {spdhg_R_multiplier}")

        r_spdhg_restart = spdhg_restart_x_y(
            problem=params['problem'],
            exitcriterion=params['exitcriterion'],
            blocksize=params['blocksize'],
            R=params['R'] * spdhg_R_multiplier,
            gamma=params['gamma'],
            restartfreq=params['restartfreq'],
        )
        export_filename = os.path.join(
            outputdir,
            f"{dataset}-spdhg_restart_x_y-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv"
        )
        export_results_to_csv(r_spdhg_restart, export_filename)
        logging.info(f"Results exported to '{export_filename}'.")
        logging.info("========================================")

    elif algo == "4":
        logging.info("========================================")
        logging.info("Running purecd_restart_x_y.")

        logging.info("purecd_blocksize = 1")
        logging.info("purecd_R = 1.0")

        r_purecd_restart = purecd_restart_x_y(
            problem=params['problem'],
            exitcriterion=params['exitcriterion'],
            blocksize=1,
            R=1.0,
            gamma=params['gamma'],
            restartfreq=params['restartfreq'],
        )
        export_filename = os.path.join(
            outputdir,
            f"{dataset}-purecd_restart_x_y-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv"
        )
        export_results_to_csv(r_purecd_restart, export_filename)
        logging.info(f"Results exported to '{export_filename}'.")
        logging.info("========================================")

    elif algo == "5":
        logging.info("========================================")
        logging.info("Running ADUCA_restart.")

        logging.info("aduca_blocksize = 1")
        result_aduca = aduca(
            problem=params['problem'],
            exitcriterion=params['exitcriterion'],
            blocksize=1,
            gamma=params['gamma'],
            restartfreq=params['restartfreq'],
            beta=0.85,
            xi=0.34
        )
        export_filename = os.path.join(
            outputdir,
            f"{dataset}-aduca_restart-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv"
        )
        export_results_to_csv(result_aduca, export_filename)
        logging.info(f"Results exported to '{export_filename}'.")
        logging.info("========================================")
    elif algo == "6":
        logging.info("========================================")
        logging.info("Running ADUCA_pdhg_restart.")

        logging.info("aduca_blocksize = 1")
        result_aduca_pdhg = aduca_pdhg(
            problem=params['problem'],
            exitcriterion=params['exitcriterion'],
            blocksize=1,
            gamma=params['gamma'],
            restartfreq=params['restartfreq'],
            beta=0.85,
            xi=0.34
        )
        export_filename = os.path.join(
            outputdir,
            f"{dataset}-aduca_pdhg_restart-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.csv"
        )
        export_results_to_csv(result_aduca_pdhg, export_filename)
        logging.info(f"Results exported to '{export_filename}'.")
        logging.info("========================================")

    else:
        logging.warning(f"Unknown algorithm index '{algo}'. Skipping.")

def main():
    # Check for minimum required arguments
    if len(sys.argv) < 4:
        print("Usage: python run_algo.py <dataset> <gamma> <algo1> <algo2> ... <algoN>")
        print("Example: python run_algo.py a1a 0.5 0 1 2")
        sys.exit(1)

    # Parse command-line arguments
    dataset = sys.argv[1]
    try:
        gamma = float(sys.argv[2])
    except ValueError:
        print("Error: <gamma> must be a floating-point number.")
        sys.exit(1)

    algo_indices = sys.argv[3:]

    # Validate dataset name
    if dataset not in DATASET_INFO:
        print(f"Error: Invalid dataset name '{dataset}'. Available datasets: {list(DATASET_INFO.keys())}")
        sys.exit(1)

    dim_dataset, num_dataset = DATASET_INFO[dataset]
    filepath = os.path.join("data", f"{dataset}")

    # Check if dataset file exists
    if not os.path.isfile(filepath):
        print(f"Error: Dataset file '{filepath}' not found.")
        sys.exit(1)

    # Problem instance parameters
    kappa = 0.1
    rho = 10.0

    # Set up logging
    outputdir = "./output/"
    logging_filename = setup_logging(outputdir, dataset, algo_indices)

    logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("Completed initialization.")

    # Log problem and algorithm parameters
    logging.info(f"Running on '{dataset}' dataset.")
    logging.info("--------------------------------------------------")
    logging.info(f"kappa = {kappa}")
    logging.info(f"rho = {rho}")
    logging.info("--------------------------------------------------")

    # Read dataset
    logging.info("Reading dataset...")
    yX_T = read_libsvm_into_yXT_sparse(filepath, dim_dataset, num_dataset)
    
    # Reformulate DRO problem
    logging.info("Reformulating DRO problem...")
    A_T, b, c = droreformuation_wmetric_hinge_standardformnormalized(yX_T, kappa, rho)
    problem = StandardLinearProgram(A_T, b, c)

    # Compute largest singular value (L)
    logging.info("Computing largest singular value (L)...")
    u, s, vt = svds(A_T, k=1, which='LM')
    L = s[0]
    logging.info(f"L = {L}")

    # Log matrix details
    logging.info("--------------------------------------------------")
    logging.info(f"A_T has size: {A_T.shape}")
    nnz_A_T = A_T.nnz
    logging.info(f"A_T has nnz: {nnz_A_T}")
    nnz_ratio = nnz_A_T / (A_T.shape[0] * A_T.shape[1])
    logging.info(f"nnz ratio: {nnz_ratio}")
    logging.info("--------------------------------------------------")

    # Exit criterion
    maxiter = 10000000
    maxtime = 3600 * 12  # 12 hours in seconds
    targetaccuracy = 1e-7
    loggingfreq = 30
    exitcriterion = ExitCriterion(maxiter, maxtime, targetaccuracy, loggingfreq)

    # Common algorithm parameters
    blocksize = 50
    R = sqrt(blocksize)
    restartfreq = inf  # For restart when metric halves, set restartfreq=inf

    logging.info(f"maxiter = {maxiter}")
    logging.info(f"maxtime = {maxtime} seconds")
    logging.info(f"targetaccuracy = {targetaccuracy}")
    logging.info(f"loggingfreq = {loggingfreq}")
    logging.info("--------------------------------------------------")
    logging.info(f"blocksize = {blocksize}")
    logging.info(f"R = {R}")
    logging.info(f"gamma = {gamma}")
    logging.info(f"restartfreq = {restartfreq}")
    logging.info("--------------------------------------------------")

    # Prepare parameters for algorithm execution
    params = {
        'problem': problem,
        'exitcriterion': exitcriterion,
        'blocksize': blocksize,
        'R': R,
        'L': L,
        'gamma': gamma,
        'restartfreq': restartfreq
    }

    # Execute selected algorithms
    for algo in algo_indices:
        execute_algorithm(algo, params, outputdir, dataset)

if __name__ == "__main__":
    main()
    