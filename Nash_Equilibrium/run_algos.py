"""Nash Equilibrium (Nash-Cournot) experiment runner.

This script is invoked by `main.py` and is used for multiple algorithms.

Important: when launched via `torchrun` (multi-process), the script is executed by
*every rank*. For distributed algorithms we therefore only write the JSON output on
rank 0 to avoid file clobbering and huge redundant output.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os

import numpy as np

from src.algorithms.aduca import aduca
from src.algorithms.aduca_torch_dist import aduca_distributed
from src.algorithms.coder import (
    coder,
    coder_linesearch,
    coder_linesearch_normalized,
    coder_normalized,
)
from src.algorithms.gr import gr, gr_normalized
from src.algorithms.pccm import pccm, pccm_normalized
from src.algorithms.utils.exitcriterion import ExitCriterion
from src.problems.GMVI_func import GMVIProblem
from src.problems.nash_g_func import NASHGFunc
from src.problems.nash_opr_func import NASHOprFunc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimization algorithms.")

    # Generic experiment controls
    parser.add_argument("--outputdir", required=True, help="Output json path")
    parser.add_argument("--maxiter", required=True, type=int, help="Max iterations")
    parser.add_argument("--maxtime", required=True, type=int, help="Max execution time in seconds")
    parser.add_argument("--targetaccuracy", required=True, type=float, help="Target accuracy")
    parser.add_argument("--optval", type=float, default=0.0, help="Known optimal value")
    parser.add_argument("--loggingfreq", type=int, default=100, help="Logging frequency")

    # Problem config
    parser.add_argument("--scenario", required=True, help="Scenario id")
    parser.add_argument("--lossfn", default="Nash", help="Loss function (kept for compatibility)")

    # Algorithm selection
    parser.add_argument("--algo", required=True, help="Algorithm to run")

    # Common algorithm params
    parser.add_argument("--lipschitz", required=True, type=float, help="Lipschitz constant")
    parser.add_argument("--mu", type=float, default=0.0, help="Mu")
    parser.add_argument("--beta", type=float, help="Algorithm parameter beta")
    parser.add_argument("--gamma", type=float, help="Algorithm parameter gamma")
    parser.add_argument("--rho", type=float, default=0.0, help="Algorithm parameter rho")
    parser.add_argument("--block_size", type=int, default=1, help="Block size (>=1)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: OS entropy)")

    # ADUCA_torch_dist specific
    parser.add_argument(
        "--strong-convexity",
        "--strong_convexity",
        dest="strong_convexity",
        action="store_true",
        help="Enable strong convexity ratio_bar update (ADUCA_torch_dist)",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="torch.distributed backend (ADUCA_torch_dist)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Torch dtype (ADUCA_torch_dist)",
    )
    parser.add_argument(
        "--reduce_dtype",
        type=str,
        default=None,
        choices=["float32", "float64"],
        help="Reduction dtype (ADUCA_torch_dist)",
    )
    parser.add_argument(
        "--sync_step",
        action="store_true",
        help="Broadcast step size each epoch (ADUCA_torch_dist)",
    )
    parser.add_argument(
        "--q_sync_every",
        type=int,
        default=1,
        help=(
            "(ADUCA_torch_dist) Synchronize Q/p/dp every N distributed steps. "
            "N=1 means sync every step; larger values reduce communication (more delay)."
        ),
    )

    return parser.parse_args()


def build_problem(scenario: int, n: int, rng: np.random.Generator) -> GMVIProblem:
    """Construct the Nash-Cournot GMVI instance used by the experiments."""

    if scenario not in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
        raise ValueError("Invalid scenario selected.")

    c = rng.uniform(1, 100, n)

    if scenario == 1:
        gamma = 1.1
        beta = rng.uniform(0.5, 2, n)
        L = rng.uniform(0.5, 5, n)
    elif scenario == 2:
        gamma = 1.1
        beta = rng.uniform(0.5, 2, n)
        L = rng.uniform(0.5, 20, n)
    elif scenario == 3:
        gamma = 1.1
        beta = rng.uniform(0.5, 2, n)
        L = rng.uniform(0.5, 50, n)
    elif scenario == 4:
        gamma = 1.5
        beta = rng.uniform(0.3, 4, n)
        L = rng.uniform(0.5, 5, n)
    elif scenario == 5:
        gamma = 1.5
        beta = rng.uniform(0.3, 10, n)
        L = rng.uniform(0.5, 5, n)
    elif scenario == 6:
        gamma = 1.5
        beta = rng.uniform(0.3, 4, n)
        L = rng.uniform(0.5, 20, n)
    elif scenario == 7:
        gamma = 1.3
        beta = rng.uniform(0.3, 4, n)
        L = rng.uniform(0.5, 5, n)
    elif scenario == 8:
        gamma = 1.3
        beta = rng.uniform(0.3, 10, n)
        L = rng.uniform(0.5, 5, n)
    else:  # scenario == 9
        gamma = 1.3
        beta = rng.uniform(0.3, 4, n)
        L = rng.uniform(0.5, 20, n)

    F = NASHOprFunc(n, gamma, beta, c, L)
    g = NASHGFunc(n)
    return GMVIProblem(F, g)


def main() -> None:
    args = parse_commandline()

    # torchrun executes this script on every rank.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    outputdir = args.outputdir
    algorithm = args.algo

    scenario = int(args.scenario)
    n = 1_000_000
    # n=1000
    logging.info(f"scenario: {scenario}, n: {n}")
    logging.info("--------------------------------------------------")

    # Exit criterion
    exitcriterion = ExitCriterion(
        args.maxiter,
        args.maxtime,
        args.targetaccuracy + args.optval,
        args.loggingfreq,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    logging.info(f"timestamp = {timestamp}")
    logging.info("Completed initialization")
    logging.info("--------------------------------------------------")

    # Problem instance
    # rng = np.random.default_rng(args.seed)
    rng = np.random.default_rng(32)
    if args.seed is not None:
        logging.info(f"Random seed: {args.seed}")
    else:
        logging.info("Random seed: OS entropy")

    problem = build_problem(scenario=scenario, n=n, rng=rng)

    # Dispatch algorithm
    if algorithm == "CODER":
        logging.info("Running CODER...")
        param = {"L": args.lipschitz, "mu": args.mu, "block_size": args.block_size}
        output, output_x = coder(problem, exitcriterion, param)

    elif algorithm == "CODER_normalized":
        logging.info("Running CODER_normalized...")
        param = {
            "L": args.lipschitz,
            "mu": args.mu,
            "block_size": args.block_size,
            "beta": args.beta,
        }
        output, output_x = coder_normalized(problem, exitcriterion, param)

    elif algorithm == "CODER_linesearch":
        logging.info("Running CODER_linesearch...")
        param = {"mu": args.mu, "block_size": args.block_size}
        output, output_x = coder_linesearch(problem, exitcriterion, param)

    elif algorithm == "CODER_linesearch_normalized":
        logging.info("Running CODER_linesearch_normalized...")
        param = {"mu": args.mu, "block_size": args.block_size, "beta": args.beta}
        output, output_x = coder_linesearch_normalized(problem, exitcriterion, param)

    elif algorithm == "PCCM":
        logging.info("Running PCCM...")
        param = {"L": args.lipschitz, "mu": args.mu, "block_size": args.block_size}
        output, output_x = pccm(problem, exitcriterion, param)

    elif algorithm == "PCCM_normalized":
        logging.info("Running PCCM_normalized...")
        param = {
            "L": args.lipschitz,
            "mu": args.mu,
            "block_size": args.block_size,
            "beta": args.beta,
        }
        output, output_x = pccm_normalized(problem, exitcriterion, param)

    elif algorithm == "GR":
        logging.info("Running Golden Ratio...")
        param = {"beta": args.beta, "block_size": args.block_size}
        output, output_x = gr(problem, exitcriterion, param)

    elif algorithm == "GR_normalized":
        logging.info("Running Golden Ratio (normalized)...")
        param = {"beta": args.beta, "block_size": args.block_size}
        output, output_x = gr_normalized(problem, exitcriterion, param)

    elif algorithm == "ADUCA":
        logging.info("Running ADUCA...")
        param = {
            "beta": args.beta,
            "gamma": args.gamma,
            "rho": args.rho,
            "mu": args.mu,
            "block_size": args.block_size,
        }
        output, output_x = aduca(problem, exitcriterion, param)

    elif algorithm == "ADUCA_torch_dist":
        logging.info("Running ADUCA (torch distributed)...")
        param = {
            "beta": args.beta,
            "gamma": args.gamma,
            "rho": args.rho,
            "mu": args.mu,
            "strong_convexity": args.strong_convexity,
            "block_size": args.block_size,
            "backend": "torch_dist",
            "dist_backend": args.dist_backend,
            "dtype": args.dtype,
            "q_sync_every": args.q_sync_every,
        }
        if args.reduce_dtype is not None:
            param["reduce_dtype"] = args.reduce_dtype
        if args.sync_step:
            param["sync_step"] = True

        output, output_x = aduca_distributed(problem, exitcriterion, param)

    else:
        raise ValueError("Wrong algorithm name supplied")

    # Only rank 0 writes JSON output when running under torchrun.
    if algorithm == "ADUCA_torch_dist" and world_size > 1 and rank != 0:
        logging.info("Non-zero rank finished; skipping JSON output write.")
        return

    if output_x is None:
        raise RuntimeError(
            "output_x is None on rank 0; this indicates a bug in the distributed algorithm wrapper."
        )

    with open(outputdir, "w") as outfile:
        json.dump(
            {
                "args": vars(args),
                "output_x": output_x.tolist(),
                "iterations": output.iterations,
                "times": output.times,
                "optmeasures": output.optmeasures,
                "L": output.L,
                "L_hat": output.L_hat,
            },
            outfile,
        )

    logging.info(f"output saved to {outputdir}")


if __name__ == "__main__":
    main()
