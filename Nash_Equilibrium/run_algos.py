### SVM

import argparse
import datetime
import numpy as np
import numpy as np
import logging
import sys
import json
import os

from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.problems.svmelastic_opr_func import SVMElasticOprFunc
from src.problems.svmelastic_g_func import SVMElasticGFunc
from src.algorithms.utils.results import Results, logresult
from src.algorithms.coder import coder, coder_linesearch
from src.algorithms.pccm import pccm
from src.algorithms.gr import gr
from src.algorithms.aduca import aduca_restart_scale, aduca_scale

import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_commandline():
    parser = argparse.ArgumentParser(description='Run optimization algorithms.')
    parser.add_argument('--outputdir', required=True, help='Output directory')
    parser.add_argument('--maxiter', required=True, type=int, help='Max iterations')
    parser.add_argument('--maxtime', required=True, type=int, help='Max execution time in seconds')
    parser.add_argument('--targetaccuracy', required=True, type=float, help='Target accuracy')
    parser.add_argument('--optval', type=float, default=0.0, help='Known optimal value')
    parser.add_argument('--loggingfreq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--scenario', required=True, help='Choice of dataset')
    parser.add_argument('--lossfn', default='SVM', help='Choice of loss function')
    parser.add_argument('--algo', required=True, help='Algorithm to run')
    parser.add_argument('--lipschitz', required=True, type=float, help='Lipschitz constant')
    parser.add_argument('--beta', type = float, help='aduca constant parameter')
    parser.add_argument('--xi', type = float, help='aduca constant parameter')
    parser.add_argument('--restartfreq', type = int, default=float('inf'), help='aduca_restart constant parameter')
    parser.add_argument('--block_size', type = int, default=1, help='block_size parameter >= 1, <= n')

    return parser.parse_args()

def main():
    # Run setup
    args = parse_commandline()
    outputdir = args.outputdir
    algorithm = args.algo
    # Problem Setup
    scenario = int(args.scenario)


    if scenario not in {1,2,3,4,5,6,7,8}:
        raise ValueError("Invalid scenario selected.")

    n=1000
    logging.info(f"scenario: {scenario}, n: {1000}")
    logging.info("--------------------------------------------------")
    
    # Exit criterion
    maxiter = args.maxiter
    maxtime = args.maxtime
    targetaccuracy = args.targetaccuracy + args.optval
    loggingfreq = args.loggingfreq
    exitcriterion = ExitCriterion(maxiter, maxtime, targetaccuracy, loggingfreq)

    # Runing
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    logging.info(f"timestamp = {timestamp}")
    logging.info("Completed initialization")
    logging.info("--------------------------------------------------")

    # Problem instance instantiation
    np.random.seed(40)
    c = np.random.uniform(1,100,n)
    # L = np.random.uniform(0.5,30,n)
    if scenario == 1:
         gamma = 1.1
         beta = np.random.uniform(0.5, 2,n)
         L = np.random.uniform(0.5,5,n)

    if scenario == 2:
         gamma = 1.1
         beta = np.random.uniform(0.5, 2,n)
         L = np.random.uniform(0.5,20,n)

    if scenario == 3:
        gamma = 1.4
        beta = np.random.uniform(0.3, 4,n)
        L = np.random.uniform(0.5,5,n)
         
    if scenario == 4:
        gamma = 1.4
        beta = np.random.uniform(0.3, 4,n)
        L = np.random.uniform(0.5,20,n)

    if scenario == 5:
        gamma = 0.8
        beta = np.random.uniform(0.3, 4, n)
        L = np.random.uniform(0.5,5,n)

    if scenario == 6:
        gamma = 0.8
        beta = np.random.uniform(0.3, 4, n)
        L = np.random.uniform(0.5,20,n)
    
    if scenario == 7:
        gamma = 1.3
        beta = np.random.uniform(0.3, 3, n)
        L = np.random.uniform(0.5,5,n)

    if scenario == 8:
        gamma = 1.3
        beta = np.random.uniform(0.3, 3, n)
        L = np.random.uniform(0.5,20,n)

    F = SVMElasticOprFunc(n, gamma, beta, c, L)
    g = SVMElasticGFunc(n)
    problem = GMVIProblem(F, g)

    # # elif algorithm == "CODER_linesearch":
    # logging.info("Running CODER_linesearch...")
    # L = args.lipschitz
    # block_size = args.block_size
    # coder_params = {"L": L, "block_size": block_size}
    # output, output_x = coder_linesearch(problem, exitcriterion, coder_params)
    # outputfilename = f"{outputdir}/{scenario}-CODER_linesearch-{args.lipschitz}-output-{timestamp}.json"
    # logging.info(f"outputfilename = {outputfilename}")
    # with open(outputfilename, 'w') as outfile:
    #     json.dump({"args": vars(args), 
    #             "output_x": output_x.tolist(),
    #             "iterations": output.iterations, 
    #             "times": output.times,
    #             "optmeasures": output.optmeasures,
    #             "L": output.L,
    #             "L_hat": output.L_hat}, 
    #             outfile)
    #     logging.info(f"output saved to {outputfilename}")

    # # if algorithm == "CODER":
    # logging.info("Running CODER...")
    # L = args.lipschitz
    # block_size = args.block_size
    # coder_params = {"L": L, "block_size": block_size}
    # output, output_x = coder(problem, exitcriterion, coder_params)
    # outputfilename = f"{outputdir}/{scenario}-CODER-{args.lipschitz}-output-{timestamp}.json"
    # logging.info(f"outputfilename = {outputfilename}")
    # with open(outputfilename, 'w') as outfile:
    #     json.dump({"args": vars(args), 
    #             "output_x": output_x.tolist(),
    #             "iterations": output.iterations, 
    #             "times": output.times,
    #             "optmeasures": output.optmeasures,
    #             "L": output.L,
    #             "L_hat": output.L_hat}, 
    #             outfile)
    #     logging.info(f"output saved to {outputfilename}")

    # # elif algorithm == "PCCM":
    # logging.info("Running PCCM...")
    # L = args.lipschitz
    # block_size = args.block_size
    # pccm_params = {"L": L, "block_size": block_size}
    # output, output_x = pccm(problem, exitcriterion, pccm_params)
    # outputfilename = f"{outputdir}/{scenario}-PCCM-{args.lipschitz}-output-{timestamp}.json"
    # logging.info(f"outputfilename = {outputfilename}")
    # with open(outputfilename, 'w') as outfile:
    #     json.dump({"args": vars(args), 
    #             "output_x": output_x.tolist(),
    #             "iterations": output.iterations, 
    #             "times": output.times,
    #             "optmeasures": output.optmeasures,
    #             "L": output.L,
    #             "L_hat": output.L_hat}, 
    #             outfile)
    #     logging.info(f"output saved to {outputfilename}")

    # elif algorithm == "GR":
    # beta = args.beta
    # block_size = args.block_size
    # logging.info("Running Golden Ratio...")
    # param = {"beta": beta, "block_size": block_size}
    # output, output_x = gr(problem, exitcriterion, param)
    # outputfilename = f"{outputdir}/{scenario}-GR-{args.lipschitz}-output-{timestamp}.json"
    # logging.info(f"outputfilename = {outputfilename}")
    # with open(outputfilename, 'w') as outfile:
    #     json.dump({"args": vars(args), 
    #             "output_x": output_x.tolist(),
    #             "iterations": output.iterations, 
    #             "times": output.times,
    #             "optmeasures": output.optmeasures,
    #             "L": output.L,
    #             "L_hat": output.L_hat}, 
    #             outfile)
    #     logging.info(f"output saved to {outputfilename}")

    # # elif algorithm == "ADUCA_scale":
    beta = args.beta
    xi = args.xi
    block_size = args.block_size
    L = args.lipschitz
    logging.info("Running ADUCA_scale...")
    param = {"beta": beta, "xi": xi, "block_size": block_size, "L": L}
    output, output_x = aduca_scale(problem, exitcriterion, param)
    outputfilename = f"{outputdir}/{scenario}-ADUCA_scale-{args.lipschitz}-output-{timestamp}.json"
    logging.info(f"outputfilename = {outputfilename}")
    with open(outputfilename, 'w') as outfile:
        json.dump({"args": vars(args), 
                "output_x": output_x.tolist(),
                "iterations": output.iterations, 
                "times": output.times,
                "optmeasures": output.optmeasures,
                "L": output.L,
                "L_hat": output.L_hat}, 
                outfile)
        logging.info(f"output saved to {outputfilename}")

    # # elif algorithm == "ADUCA_restart_scale":
    # beta = args.beta
    # xi = args.xi
    # restartfreq = args.restartfreq
    # block_size = args.block_size
    # L = args.lipschitz
    # logging.info("Running ADUCA_restart_scale...")
    # param = {"beta": beta, "xi": xi, "restartfreq": restartfreq, "block_size": block_size, "L": L}
    # output, output_x = aduca_restart_scale(problem, exitcriterion, param)
    # outputfilename = f"{outputdir}/{scenario}-ADUCA_restart_scale-{args.lipschitz}-output-{timestamp}.json"
    # logging.info(f"outputfilename = {outputfilename}")
    # with open(outputfilename, 'w') as outfile:
    #     json.dump({"args": vars(args), 
    #             "output_x": output_x.tolist(),
    #             "iterations": output.iterations, 
    #             "times": output.times,
    #             "optmeasures": output.optmeasures,
    #             "L": output.L,
    #             "L_hat": output.L_hat}, 
    #             outfile)
    #     logging.info(f"output saved to {outputfilename}")

    # else:
    #     print("Wrong algorithm name supplied")
    

if __name__ == "__main__":
    main()