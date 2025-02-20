import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range

def aduca_scale(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of adapCODER
    n = problem.operator_func.n
    param_L = problem.operator_func.L
    param_gamma = problem.operator_func.gamma
    beta = parameters["beta"]
    xi = parameters["xi"]

    block_size = parameters['block_size']
    blocks= construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    # phi_1 = 1/beta
    phi_2 = xi * beta * (1+beta)
    phi_3 = 4 / (7 * beta * (1+beta) * (1-xi) )
    phi_4 = (((1-xi) * (1 + beta))  /  7 * beta)**0.5 * 0.5
    phi_5 = 1 / (7 * beta)
    alpha = min(phi_2, phi_3)

    # Stepsize selection function
    def aduca_stepsize(normalizer, normalizer_recip, u, u_, a, a_, F, F_, F_tilde):
        step_1 = alpha * a 

        u_diff = np.copy(u - u_)

        ### we can heuristically scale the step
        F_tilde_diff = np.copy(F-F_tilde)
        L_hat_k = np.sqrt(np.inner(F_tilde_diff, (normalizer * F_tilde_diff)) / np.inner(u_diff, (normalizer_recip * u_diff))) 
        if L_hat_k == 0:
            step_2 = 100000
        else:    
            step_2 = (phi_4 / L_hat_k) * (a / a_)**0.5    

        F_diff = np.copy(F-F_)
        L_k = np.sqrt(np.inner(F_diff, (normalizer * F_diff)) / np.inner(u_diff, (normalizer_recip * u_diff)))
        # print(f"!!! L_k: {L_k}")
        if L_k == 0:
            step_3 = 100000
        else:
            step_3 = (phi_5 ** 2) / (a_ * L_k**2)  
            # print(f"!!! step_3: {step_3}")
        
        step = min(step_1, step_2, step_3)
        # print(f" !!! Stepsize: {step}")
        return step, L_k , L_hat_k

    ### normalizers
    time_start_initialization = time.time()


    # normalizers = np.power(np.copy(1/param_L), np.copy(1/problem.operator_func.beta))
    # normalizers_recip = np.copy(1/normalizers)
    normalizers = np.power(np.copy(1/param_L), np.copy(1/problem.operator_func.beta)) * param_gamma / 5000**(1/param_gamma)
    normalizers_recip = np.copy(1/normalizers)
    # normalizers = np.ones(n)
    # normalizers_recip = np.ones(n)

    time_end_initialization = time.time()
    logging.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

    a = 0
    a_ = 0
    A = 0

    if u_0 is None:
        # u_0 = np.full(shape=n, fill_value=-0.0001)
        u_0 = np.ones(n)
    u_ = np.copy(u_0)
    u_hat = np.zeros(n)
    v = np.zeros(n)
    v_ = np.zeros(n)

    F = np.zeros(n)
    F_ = np.zeros(n)
    F_tilde = np.zeros(n)
    F_tilde_ = np.zeros(n)
    F_bar = np.copy(F_tilde)


    k = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.residual(u_)
    logresult(results, 1, 0.0, init_opt_measure)

    F_0 = problem.operator_func.func_map(u_0)
    F_tilde_0 = np.copy(F_0)
    F_tilde_1 = np.copy(F_tilde_0)

    ## line-search for the first step
    a_0 = 4*alpha
    while True:
        u = np.copy(u_0)
        u_ = np.copy(u_0)
        F_store = np.copy(F_0)
        Q = np.sum(u)
        p = problem.operator_func.p(Q)
        p_ = p
        dp = problem.operator_func.dp(Q)
        dp_ = dp

        a_0 = a_0 / 2
        u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0)
        for block in blocks:
                u[block] = u_1[block]
                F_tilde_1[block] = F_store[block]

                Q += np.sum(u[block] - u_0[block])
                p_ = p
                p = problem.operator_func.p(Q)
                dp_ = dp
                dp = problem.operator_func.dp(Q)
                problem.operator_func.func_map_block_update(F_store, u, p, p_, dp, dp_, block)
                # print(f"Caught an exception")

        F_1 = np.copy(F_store)
        norm_F = np.linalg.norm((F_1 - F_0))
        norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
        norm_u = np.linalg.norm((u_1 - u_0))

        # print(f"phi_2: {phi_2}")
        # print(f"a_0: {a_0}")
        if ((a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u)) or a_0 < 0.00001:
            break
        
    a_ = a_0
    a = a_0
    A = 0

    u = np.copy(u_1)
    u_ = np.copy(u_0)
    v_ = np.copy(u_)
    u_hat = A * u_

    Q = np.sum(u)
    p = problem.operator_func.p(Q)
    p_ = p
    dp = problem.operator_func.dp(Q)
    dp_ = dp

    F = np.copy(F_1)
    F_ = np.copy(F_0)
    F_tilde = np.copy(F_tilde_1)
    F_tilde_ = np.copy(F_tilde_0)
    F_bar = np.zeros(n)

    while not exit_flag:
        # Step 6
        step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_, F, F_, F_tilde)
        a_ = a
        a = step
        A += a

        for block in blocks:
            # Step 8
            F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
            
            # Step 9
            v[block] = (1-beta) * u[block] + beta * v_[block]

            # Step 10
            u_[block] = u[block]
            u[block] = problem.g_func.prox_opr_block(v[block] - a * normalizers[block] * F_bar[block])

            # Step 11
            F_tilde_[block] = F_tilde[block]
            F_tilde[block] = F_store[block]
            Q += np.sum(u[block] - u_[block])
            p_ = p
            p = problem.operator_func.p(Q)
            dp_ = dp
            dp = problem.operator_func.dp(Q)
            problem.operator_func.func_map_block_update(F_store, u, p, p_, dp, dp_, block)
            

        np.copyto(F_, F)
        F = np.copy(F_store)
        # print(f"If F equal to F_tilde")
        np.copyto(v_, v)

        # print(f"!!! (a / A): {(a / A)}")
        u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

        # Increment iteration counters
        k += m
        
        if k % (m *  exit_criterion.loggingfreq) == 0:
            # Compute averaged variables
            # step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
            # a_ = a
            # a = step
            # A += a
            # u_hat = ((A - a) * u_hat / A) + (a*u / A)
            elapsed_time = time.time() - start_time
            opt_measure = problem.residual(u)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {k}, opt_measure: {opt_measure}")
            logresult(results, k, elapsed_time, opt_measure, L=L, L_hat=L_hat)
            exit_flag = CheckExitCondition(exit_criterion, k, elapsed_time, opt_measure)
            if exit_flag:
                break
            
    return results, u



def aduca_restart_scale(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of adapCODER
    n = problem.operator_func.n
    param_L = problem.operator_func.L
    beta = parameters["beta"]
    xi = parameters["xi"]
    restartfreq = parameters["restartfreq"]

    block_size = parameters['block_size']
    blocks= construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    phi_2 = xi * beta * (1+beta)
    phi_3 = 4 / (7 * beta * (1+beta) * (1-xi) )
    phi_4 = (((1-xi) * (1 + beta))  /  7 * beta)**0.5 * 0.5
    phi_5 = 1 / (7 * beta)
    alpha = min(phi_2, phi_3)

    # Stepsize selection function
    def aduca_stepsize(normalizer, normalizer_recip, u, u_, a, a_, F, F_, F_tilde):
        step_1 = alpha * a 

        u_diff = np.copy(u - u_)

        ### we can heuristically scale the step
        F_tilde_diff = np.copy(F-F_tilde)
        L_hat_k = np.sqrt(np.inner(F_tilde_diff, (normalizer * F_tilde_diff)) / np.inner(u_diff, (normalizer_recip * u_diff))) 
        if L_hat_k == 0:
            step_2 = 100000
        else:    
            step_2 = (phi_4 / L_hat_k) * (a / a_)**0.5    

        F_diff = np.copy(F-F_)
        L_k = np.sqrt(np.inner(F_diff, (normalizer * F_diff)) / np.inner(u_diff, (normalizer_recip * u_diff)))
        # print(f"!!! L_k: {L_k}")
        if L_k == 0:
            step_3 = 100000
        else:
            step_3 = (phi_5 ** 2) / (a_ * L_k**2)  
            # print(f"!!! step_3: {step_3}")
        
        step = min(step_1, step_2, step_3)
        # print(f" !!! Stepsize: {step}")
        return step, L_k , L_hat_k

    ### normalizers
    time_start_initialization = time.time()

    normalizers = np.power(np.copy(1/param_L), np.copy(1/problem.operator_func.beta))
    normalizers_recip = np.copy(1/normalizers)
    # normalizers = np.ones(n)
    # normalizers_recip = np.ones(n)

    time_end_initialization = time.time()
    logging.info(f"Initialization time = {time_end_initialization - time_start_initialization:.4f} seconds")

    a = 0
    a_ = 0
    A = 0

    if u_0 is None:
        # u_0 = np.full(shape=n, fill_value=-0.0001)
        u_0 = np.ones(n)
    u_ = np.copy(u_0)
    u_hat = np.zeros(n)
    v = np.zeros(n)
    v_ = np.zeros(n)

    F = np.zeros(n)
    F_ = np.zeros(n)
    F_tilde = np.zeros(n)
    F_tilde_ = np.zeros(n)
    F_bar = np.copy(F_tilde)


    outer_k = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.residual(u_)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        u = np.copy(u_0)
        u_ = np.copy(u_0)
    
        F_0 = problem.operator_func.func_map(u_0)
        F_tilde_0 = np.copy(F_0)
        F_tilde_1 = np.copy(F_tilde_0)

        ## line-search for the first step
        a_0 = 4 * alpha
        while True:
            u = np.copy(u_0)
            u_ = np.copy(u_0)
            F_store = np.copy(F_0)
            Q = np.sum(u)
            p = problem.operator_func.p(Q)
            p_ = p
            dp = problem.operator_func.dp(Q)
            dp_ = dp
            a_0 = a_0 / 2
            u_1 = problem.g_func.prox_opr(u_0 - a_0 * normalizers * F_0)

            for block in blocks:
                u[block] = u_1[block]
                F_tilde_1[block] = F_store[block]

                Q += np.sum(u[block] - u_0[block])
                p_ = p
                p = problem.operator_func.p(Q)
                dp_ = dp
                dp = problem.operator_func.dp(Q)
                problem.operator_func.func_map_block_update(F_store, u, p, p_, dp, dp_, block)
                
            F_1 = np.copy(F_store)
            norm_F = np.linalg.norm((F_1 - F_0))
            norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
            norm_u = np.linalg.norm((u_1 - u_0))

            # print(f"phi_2: {phi_2}")
            # print(f"a_0: {a_0}")
            if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
                break

        a_ = a_0
        a = a_0
        A = 0

        u = np.copy(u_1)
        u_ = np.copy(u_0)
        v_ = np.copy(u_)
        u_hat = A * u_

        F = np.copy(F_1)
        F_ = np.copy(F_0)
        F_tilde = np.copy(F_tilde_1)
        F_tilde_ = np.copy(F_tilde_0)
        F_bar = np.zeros(n)
        

        k = 0
        restart_flag = False
        
        while not exit_flag and not restart_flag:

            step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip, u, u_, a, a_, F, F_, F_tilde)
            a_ = a
            a = step
            A += a

            for block in blocks:
                # Step 8
                F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
            
                # Step 9
                v[block] = (1-beta) * u[block] + beta * v_[block]

                # Step 10
                u_[block] = u[block]
                u[block] = problem.g_func.prox_opr_block(v[block] - a * normalizers[block] * F_bar[block])

                # Step 11
                Q += np.sum(u[block] - u_[block])
                F_tilde_[block] = F_tilde[block]
                F_tilde[block] = F_store[block]
                p_ = p
                p = problem.operator_func.p(Q)
                dp_ = dp
                dp = problem.operator_func.dp(Q)
                problem.operator_func.func_map_block_update(F_store, u, p, p_, dp, dp_, block)


            np.copyto(F_, F)
            F = np.copy(F_store)
            # print(f"If F equal to F_tilde")
            np.copyto(v_, v)

            # print(f"!!! (a / A): {(a / A)}")
            u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

            # Increment iteration counters
            outer_k += m
            k += m
            
            if outer_k % (m *  exit_criterion.loggingfreq) == 0:
                # Compute averaged variables
                # step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
                # a_ = a
                # a = step
                # A += a      
                # u_hat = ((A - a) * u_hat / A) + (a*u / A)
                elapsed_time = time.time() - start_time
                opt_measure = problem.residual(v)
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {outer_k}, opt_measure: {opt_measure}")
                logresult(results, outer_k, elapsed_time, opt_measure, L=L, L_hat=L_hat)
                exit_flag = CheckExitCondition(exit_criterion, outer_k, elapsed_time, opt_measure)
                if exit_flag:
                    break
            
            if (k >= restartfreq):
                elapsed_time = time.time() - start_time
                opt_measure = problem.residual(v)
                logging.info("<===== RESTARTING")
                logging.info(f"k: {k}")
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {outer_k}, opt_measure: {opt_measure}")
                # Compute averaged variables
                step, L, L_hat = aduca_stepsize(normalizers, normalizers_recip,u,u_,a,a_,F,F_,F_tilde)
                a_ = a
                a = step
                A += a      
                u_hat = ((A - a) * u_hat / A) + (a*u / A)
                u_0 = np.copy(u_hat)
                # Update x0 and y0 for restart
                init_opt_measure = opt_measure
                restart_flag = True
                break
            
    return results, u









