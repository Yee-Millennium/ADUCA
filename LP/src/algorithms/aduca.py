import numpy as np
from scipy.sparse import csc_matrix
import logging
import copy
import time
from math import inf
import sys

from src.algorithms.utils.exitcriterion import ExitCriterion, check_exit_condition
from src.algorithms.utils.results import Results
from src.algorithms.utils.helper import compute_fvaluegap_metricLP, compute_nzrows_for_blocks
from src.problems.standardLP import StandardLinearProgram


def prox_block_u(z, block:range, d):
    if block.start >= d:  
        return z
    else:
        prox = np.maximum(0, z[: d-block.start])
        # print(np.any(prox < 0))
        z[:d-block.start] = prox
        return z

def prox_u(u, d):
    prox = np.maximum(0, u[: d])
    u[:d] = prox
    return u

'''
Given d, A_T, F, we want to update the the operator F corresponding to the udpated block coordinate uslice and its block location.
'''
def F_block_update(d, A_T, F, block:range, uslice, uslice_):
    ### Only update x
    if block.stop <= d:
        F_new_slice_diff = -(uslice - uslice_) @ A_T[block]
        F[d:] += F_new_slice_diff

    ### Only update y
    elif block.start >= d:
        F_new_slice_diff = A_T[:, :block.stop-block.start] @ (uslice - uslice_)
        F[:d] += F_new_slice_diff 
    else:
        uslicex = uslice[:(d- block.start)]
        uslicex_ = uslice_[:(d- block.start)]
        uslicey = uslice[(d- block.start):]
        uslicey_ = uslice_[(d- block.start):]
        F_x_new_slice_diff = -(uslicex - uslicex_) @ A_T[block.start: d]
        F_y_new_slice_diff = A_T[:, : block.stop] @ (uslicey - uslicey_)
        F[d:] = F_x_new_slice_diff + F[d:]
        F[0:d] = F_y_new_slice_diff + F[0:d]
    return F
    
    

def aduca(
    problem,
    exitcriterion,
    gamma=1.0,
    blocksize=10,
    restartfreq=inf,
    io=None,
    beta=0.85,
    xi=0.34
):
    """
    Adaptive Delayed-Update Cyclic Algorithm with Restart.

    Args:
        problem (StandardLinearProgram): The standard linear program.
        exitcriterion (ExitCriterion): The exit criteria.
        gamma (float, optional): Parameter γ. Defaults to 1.0.
        sigma (float, optional): Parameter σ. Defaults to 0.0.
        R (int, optional): Radius parameter. Defaults to 10.
        blocksize (int, optional): Size of blocks for coordinate updates. Defaults to 10.
        restartfreq (float, optional): Frequency for restarting the algorithm. Defaults to inf.
        io (IO stream, optional): Optional I/O stream for flushing outputs. Defaults to None.

    Returns:
        Results: An instance containing logged results.
    """
    # Set up logging
    logger = logging.getLogger('aduca_restart')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout) if io is None else logging.StreamHandler(io)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    
    if io is not None:
        io.flush()

    # Extract problem data
    A_T = problem.A_T.tocsc()  # Ensure A_T is in CSC format
    # print(f"!!! b's type: {type(problem.b)}")
    b = problem.b.flatten()  # Convert b to a 1D numpy array
    c = problem.c  # Assuming c is a numpy array

    # Get dimensions
    d, n = A_T.shape

    # Constants
    phi_1 = 2 * xi * beta * (1+beta)
    phi_2 = (((1-2*xi) * (1 + beta))  /  7 * beta)**0.5 * 0.5
    phi_3 = 1 / (7 * beta)
    alpha = min(phi_1, 4 / (7 * beta * (1+beta) * (1-2 * xi) ))

    # Stepsize selection function
    def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
        step_1 = alpha * a 

        L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
        if L_hat_k == 0:
            step_2 = 100
        else:    
            step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    

        L_k = np.linalg.norm(F - F_) / (np.linalg.norm(u - u_)) 
        if L_k == 0:
            step_3 = 100
        else:
            step_3 = (phi_3 ** 2) / (a_ * L_k**2)  
        
        step = min(step_1, step_2, step_3)
        return step


    # Initialize x0, y0 as zeros
    x0 = np.full(fill_value=0.001, shape = d)
    y0 = np.full(fill_value=0.001, shape = n)

    # Precomputing blocks, nzrows, sliced_A_T
    start_time_init = time.time()
    blocks, C = compute_nzrows_for_blocks(A_T, blocksize)
    num_nnz = A_T.getnnz(axis=1)  # Number of non-zeros per row
    sliced_A_Ts = [A_T[C_j, blocks_j] for C_j, blocks_j in zip(C, blocks)]
    end_time_init = time.time()
    logger.info(f"Initialization time = {end_time_init - start_time_init:.4f} seconds")

    # Start of ADUCA
    m = len(blocks)

    # Log initial measure
    starttime = time.time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    logger.info(f"init_metricLP: {init_metricLP}")
    Results.log_result(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 0
    exitflag = False

    while not exitflag:
        # print(f"!!! outer_k: {outer_k}")
        # Initialize ADUCA parameters
        # idx_seq = list(range(m))

        # Initialize variables
        x = copy.deepcopy(x0)
        y = copy.deepcopy(y0)
        u = np.concatenate((x,y))
        x_ = copy.deepcopy(x0)
        y_ = np.zeros_like(x0)
        u_ = np.concatenate((x_,y_))

        u_0 = np.copy(u)
        u_1 = np.copy(u)
        
        F_0 = np.concatenate((c + A_T @ y, -x @ A_T + b))
        F_tilde_0 = np.copy(F_0)
        F_tilde_1 = np.copy(F_0)
        F_store = np.copy(F_0)
        a_0 = phi_1 * 2

        # linesearch
        while(True):
            a_0 = a_0 / 2
            u_1 = prox_u(u_0 - a_0 * F_0, d)
            F_1 = np.concatenate((c + A_T @ u_1[d:], -u_1[:d] @ A_T + b))
            for block in blocks:
                if block.start == 0:
                    F_store = F_block_update(d, A_T, F_store, block, u_1[block], u_0[block])
                else:
                    F_tilde_1[block] = F_store[block]
                    F_store = F_block_update(d, A_T, F_store, block, u_1[block], u_0[block])
            norm_F = np.linalg.norm(F_1 - F_0)
            norm_F_tilde = np.linalg.norm(F_1 - F_tilde_1)
            norm_u = np.linalg.norm(u_1 - u_0)
            if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
                
                break
        
        ### Initialization after linesearch
        a_ = a_0
        a = a_0
        A = 0

        u = np.copy(u_1)
        u_ = np.copy(u_0)
        v = np.copy(u_)
        v_ = np.copy(u_)
        u_hat = A * u_

        F = np.copy(F_1)
        F_ = np.copy(F_0)
        F_tilde = np.copy(F_tilde_1)
        F_tilde_ = np.copy(F_tilde_0)
        F_bar = np.zeros(d + n)

        k = 0
        restartflag = False

        while not exitflag and not restartflag:
            # Step 7
            # t_start = time.time()
            step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
            a_ = a
            a = step
            A += a
            # t_end = time.time()
            # print(f"!!! Time used in stepsize selection: {t_end - t_start}")

            # Step 8
            for block in blocks:
                # Step 9
                F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])

                # Step 10
                v[block] = (1-beta) * u[block] + beta * v_[block]

                # Step 11
                u_[block] = u[block]
                # t_start = time.time()
                u[block] = prox_block_u(v[block] - a * F_bar[block], block, d)

                # Step 12
                F_tilde_[block] = F_tilde[block]
                F_tilde[block] = F_store[block]
                # t_start = time.time()
                F_store = F_block_update(d, A_T, F_store, block, u[block], u_[block])
                # t_end = time.time()
                # print(f"!!! Time used in F_block update: {t_end - t_start}")
            
            np.copyto(F_, F)
            F = np.copy(F_store)
            np.copyto(v_, v)

            u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

            # Increment iteration counters
            k += 1
            outer_k += 1

            # Logging and checking exit condition
            if outer_k % (exitcriterion.loggingfreq * m) == 0:
                # Compute averaged variables
                # step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
                # a_ = a
                # a = step
                # A += a      
                # u_hat = ((A - a) * u_hat / A) + (a*u / A)
                 
                x_out = u_hat[:d]
                y_out = u_hat[d:]

                # Compute progress measures
                fvaluegap, metricLP = compute_fvaluegap_metricLP(x_out, y_out, problem)

                # Compute elapsed time
                elapsedtime = time.time() - starttime
                logger.info(f"elapsedtime: {elapsedtime}")
                logger.info(f"outer_k: {outer_k}, fvaluegap: {fvaluegap}, metricLP: {metricLP}")

                # Log the results
                Results.log_result(results, outer_k, elapsedtime, fvaluegap, metricLP)
                if io is not None:
                    io.flush()

                # Check exit conditions
                exitflag = check_exit_condition(exitcriterion, outer_k, elapsedtime, metricLP)
                if exitflag:
                    break

                # Check restart conditions
                if (k >= restartfreq * m) or (restartfreq == inf and metricLP <= 0.5 * init_metricLP):
                    logger.info("<===== RESTARTING")
                    logger.info(f"k ÷ m: {k / m}")
                    logger.info(f"elapsedtime: {elapsedtime}")
                    logger.info(f"outer_k: {outer_k}, fvaluegap: {fvaluegap}, metricLP: {metricLP}")
                    if io is not None:
                        io.flush()

                    # Compute averaged variables
                    step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
                    a_ = a
                    a = step
                    A += a      
                    u_hat = ((A - a) * u_hat / A) + (a*u / A)
                    x_out = u_hat[:d]
                    y_out = u_hat[d:]
                    # Update x0 and y0 for restart
                    x0 = copy.deepcopy(x_out)
                    y0 = copy.deepcopy(y_out)
                    init_fvaluegap = fvaluegap
                    init_metricLP = metricLP
                    restartflag = True
                    break

    return results





def aduca_pdhg(
    problem,
    exitcriterion,
    gamma=1.0,
    blocksize=10,
    restartfreq=inf,
    io=None,
    beta=0.85,
    xi=0.34
):
    """
    Adaptive Delayed-Update Cyclic Algorithm with Restart.

    Args:
        problem (StandardLinearProgram): The standard linear program.
        exitcriterion (ExitCriterion): The exit criteria.
        gamma (float, optional): Parameter γ. Defaults to 1.0.
        sigma (float, optional): Parameter σ. Defaults to 0.0.
        R (int, optional): Radius parameter. Defaults to 10.
        blocksize (int, optional): Size of blocks for coordinate updates. Defaults to 10.
        restartfreq (float, optional): Frequency for restarting the algorithm. Defaults to inf.
        io (IO stream, optional): Optional I/O stream for flushing outputs. Defaults to None.

    Returns:
        Results: An instance containing logged results.
    """
    # Set up logging
    logger = logging.getLogger('aduca_restart')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout) if io is None else logging.StreamHandler(io)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


    if io is not None:
        io.flush()

    # Extract problem data
    A_T = problem.A_T.tocsc()  # Ensure A_T is in CSC format
    # print(f"!!! b's type: {type(problem.b)}")
    b = problem.b.flatten()  # Convert b to a 1D numpy array
    c = problem.c  # Assuming c is a numpy array

    # Get dimensions
    d, n = A_T.shape

    # Constants
    phi_1 = 2 * xi * beta * (1+beta)
    phi_2 = (((1-2*xi) * (1 + beta))  /  7 * beta)**0.5 * 0.5
    phi_3 = 1 / (7 * beta)
    alpha = min(phi_1, 4 / (7 * beta * (1+beta) * (1-2 * xi) ))

    # Stepsize selection function
    def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
        step_1 = alpha * a 

        L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
        if L_hat_k == 0:
            step_2 = 100
        else: 
            step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    

        L_k = np.linalg.norm(F - F_) / (np.linalg.norm(u - u_)) 
        if L_k == 0:
            step_3 = 100
        else:
            step_3 = (phi_3 ** 2) / (a_ * L_k**2)  
        
        step = min(step_1, step_2, step_3)
        return step


    # Initialize x0, y0 as zeros
    x0 = np.zeros(shape=d)
    y0 = np.zeros(shape=n)
    # x0 = np.full(fill_value=1, shape = d)
    # y0 = np.full(fill_value=1, shape = n)
    m = 1

    # Precomputing blocks, nzrows, sliced_A_T
    start_time_init = time.time()
    end_time_init = time.time()
    logger.info(f"Initialization time = {end_time_init - start_time_init:.4f} seconds")

    # Start of ADUCA

    # Log initial measure
    starttime = time.time()
    results = Results()
    init_fvaluegap, init_metricLP = compute_fvaluegap_metricLP(x0, y0, problem)
    logger.info(f"init_metricLP: {init_metricLP}")
    Results.log_result(results, 1, 0.0, init_fvaluegap, init_metricLP)

    outer_k = 0
    exitflag = False
    
    while not exitflag:
        # Initialize variables
        x = copy.deepcopy(x0)
        y = copy.deepcopy(y0)
        u = np.concatenate((x,y))
        x_ = copy.deepcopy(x0)
        y_ = np.zeros_like(x0)
        u_ = np.concatenate((x_,y_))

        u_0 = np.copy(u)
        u_1 = np.copy(u)
        
        F_0 = np.concatenate((c + A_T @ y, -x @ A_T + b))
        F_tilde_0 = np.copy(F_0)
        F_tilde_1 = np.copy(F_0)
        F_store = np.copy(F_0)
        a_0 = phi_1 * 2
        # linesearch
        while(True):
            a_0 = a_0 / 2
            u_1 = prox_u(u_0 - a_0 * F_0, d)
            F_1 = np.concatenate((c + A_T @ u_1[d:], -u_1[:d] @ A_T + b))

            F_tilde_1[:d] = F_store[:d]
            F_store = F_block_update(d, A_T, F_store, range(0,d), u_1[:d], u_0[:d])

            F_tilde_1[d:] = F_store[d:]
            F_store = F_block_update(d, A_T, F_store, range(d,d+n), u_1[d:], u_0[d:])
            
            norm_F = np.linalg.norm(F_1 - F_0)
            norm_F_tilde = np.linalg.norm(F_1 - F_tilde_1)
            norm_u = np.linalg.norm(u_1 - u_0)
            # print(f"!!! norm_F: {norm_F}")
            # print(f"!!! norm_F_tilde: {norm_F_tilde}")
            # print(f"!!! norm_u: {norm_u}")
            if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
                break
        
        u_1 = prox_u(u_0 - a_0 * F_0, d)
        F_1 = np.concatenate((c + A_T @ u_1[d:], -u_1[:d] @ A_T + b))
        F_tilde_1[:d] = F_store[:d]
        F_store = F_block_update(d, A_T, F_store, range(0,d), u_1[:d], u_0[:d])
        F_tilde_1[d:] = F_store[d:]
        F_store = F_block_update(d, A_T, F_store, range(d,d+n), u_1[d:], u_0[d:])

        ### Initialization after linesearch
        a_ = a_0
        a = a_0
        A = 0

        u = np.copy(u_1)
        u_ = np.copy(u_0)
        v = np.copy(u_)
        v_ = np.copy(u_)
        u_hat = A * u_

        F = np.copy(F_1)
        F_ = np.copy(F_0)
        F_tilde = np.copy(F_tilde_1)
        F_tilde_ = np.copy(F_tilde_0)
        F_bar = np.zeros(d + n)

        k = 0
        restartflag = False

        while not exitflag and not restartflag:
            # Step 7
            # t_start = time.time()
            # print(f"!!! a_: {a_}")
            step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
            a_ = a
            a = step
            A += a
            # t_end = time.time()
            # print(f"!!! Time used in stepsize selection: {t_end - t_start}")

            ### Update x
            # Step 9
            F_bar[:d] = F_tilde[:d] + (a_ / a) * (F_[:d] - F_tilde_[:d])
            # Step 10
            v[:d] = (1-beta) * u[:d] + beta * v_[:d]
            # Step 11
            u_[:d] = u[:d]
            # t_start = time.time()
            u[:d] = prox_block_u(v[:d] - a * F_bar[:d], range(0,d), d)
            # Step 12
            F_tilde_[:d] = F_tilde[:d]
            F_tilde[:d] = F_store[:d]
            # t_start = time.time()
            F_store = F_block_update(d, A_T, F_store, range(0,d), u[:d], u_[:d])
            # t_end = time.time()
            # print(f"!!! Time used in F_block update: {t_end - t_start}")

            ### Update y
            F_bar[d:] = F_tilde[d:] + (a_ / a) * (F_[d:] - F_tilde_[d:])
            v[d:] = (1-beta) * u[d:] + beta * v_[d:]
            u_[d:] = u[d:]
            u[d:] = prox_block_u(v[d:] - a * F_bar[d:], range(d,d+n), d)
            F_tilde_[d:] = F_tilde[d:]
            F_tilde[d:] = F_store[d:]
            F_store = F_block_update(d, A_T, F_store, range(d,d+n), u[d:], u_[d:])

            np.copyto(F_, F)
            F = np.copy(F_store)
            np.copyto(v_, v)

            u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

            # Increment iteration counters
            k += 1
            outer_k += 1

            # Logging and checking exit condition
            if outer_k % (exitcriterion.loggingfreq * m) == 0:
                # Compute averaged variables
                # step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
                # a_ = a
                # a = step
                # A += a      
                # u_hat = ((A - a) * u_hat / A) + (a*u / A)
                 
                x_out = u_hat[:d]
                y_out = u_hat[d:]

                # Compute progress measures
                fvaluegap, metricLP = compute_fvaluegap_metricLP(x_out, y_out, problem)

                # Compute elapsed time
                elapsedtime = time.time() - starttime
                logger.info(f"elapsedtime: {elapsedtime}")
                logger.info(f"outer_k: {outer_k}, fvaluegap: {fvaluegap}, metricLP: {metricLP}")

                # Log the results
                Results.log_result(results, outer_k, elapsedtime, fvaluegap, metricLP)
                if io is not None:
                    io.flush()

                # Check exit conditions
                exitflag = check_exit_condition(exitcriterion, outer_k, elapsedtime, metricLP)
                if exitflag:
                    break

                # Check restart conditions
                if (k >= restartfreq * m) or (restartfreq == inf and metricLP <= 0.5 * init_metricLP):
                    logger.info("<===== RESTARTING")
                    logger.info(f"k ÷ m: {k / m}")
                    logger.info(f"elapsedtime: {elapsedtime}")
                    logger.info(f"outer_k: {outer_k}, fvaluegap: {fvaluegap}, metricLP: {metricLP}",)
                    if io is not None:
                        io.flush()

                    # Compute averaged variables
                    step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
                    a_ = a
                    a = step
                    A += a      
                    u_hat = ((A - a) * u_hat / A) + (a*u / A)
                    x_out = u_hat[:d]
                    y_out = u_hat[d:]
                    # Update x0 and y0 for restart
                    x0 = copy.deepcopy(x_out)
                    y0 = copy.deepcopy(y_out)
                    init_fvaluegap = fvaluegap
                    init_metricLP = metricLP
                    restartflag = True
                    break

    return results