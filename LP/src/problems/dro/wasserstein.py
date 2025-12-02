import numpy as np
from scipy.sparse import csc_matrix, identity, diags, bmat, vstack

def droreformuation_wmetric_hinge_standardformnormalized(yX_T: csc_matrix, kappa: float, rho: float):
    """
    Reformulate a R-DRO problem with Wasserstein-metric based ambiguity set into a standard form LP.
    See Appendix C in the CLVR paper.

    Args:
        yX_T (csc_matrix): Transposed data matrix in CSC format (dimensions: dim_dataset x num_dataset).
        kappa (float): Parameter κ.
        rho (float): Parameter ρ.

    Returns:
        A_T (csc_matrix): Constraint matrix in standard LP form.
        b (csc_matrix): RHS vector in standard LP form.
        c (numpy.ndarray): Objective function coefficients in standard LP form.
    """
    # Get dimensions
    dim_dataset, num_dataset = yX_T.shape

    # Define vectors (as sparse row matrices)
    v1_n_T = csc_matrix(np.ones((1, num_dataset)))    # Shape: 1 x num_dataset
    v1_d_T = csc_matrix(np.ones((1, dim_dataset)))    # Shape: 1 x dim_dataset
    v0_n_T = csc_matrix(np.zeros((1, num_dataset)))   # Shape: 1 x num_dataset

    # Define identity matrices
    I_nn = identity(num_dataset, format='csc')        # Shape: num_dataset x num_dataset
    I_dd = identity(dim_dataset, format='csc')        # Shape: dim_dataset x dim_dataset

    # Define zero matrices
    O_nn = csc_matrix((num_dataset, num_dataset))      # Zero matrix: num_dataset x num_dataset
    O_dn = csc_matrix((dim_dataset, num_dataset))      # Zero matrix: dim_dataset x num_dataset
    O_nd = csc_matrix((num_dataset, dim_dataset))      # Zero matrix: num_dataset x dim_dataset
    O_dd = csc_matrix((dim_dataset, dim_dataset))      # Zero matrix: dim_dataset x dim_dataset

    # Build A_T as a block matrix with 10 block rows and 5 block columns
    A_T_blocks = [
        [-I_nn,      I_nn,     I_nn,     O_nd,      O_nd],    # Block row 0
        [I_nn,       O_nn,     I_nn,     O_nd,      O_nd],    # Block row 1
        [O_nn,      -I_nn,    -I_nn,     O_nd,      O_nd],    # Block row 2
        [O_nn,       O_nn,    -I_nn,     O_nd,      O_nd],    # Block row 3
        [O_dn,       yX_T,     O_dn,      I_dd,      I_dd],    # Block row 4
        [O_dn,      -yX_T,     O_dn,     -I_dd,     -I_dd],    # Block row 5
        [O_dn,       O_dn,     O_dn,     -I_dd,      O_dd],    # Block row 6
        [O_dn,       O_dn,     O_dn,      O_dd,      I_dd],    # Block row 7
        [-2*kappa*v1_n_T, v0_n_T, v0_n_T, v1_d_T, -v1_d_T],      # Block row 8
        [2*kappa*v1_n_T, v0_n_T, v0_n_T, -v1_d_T, v1_d_T],       # Block row 9
    ]

    # Assemble the block matrix into a single sparse matrix A_T
    A_T = bmat(A_T_blocks, format='csc')

    # Build the RHS vector b by stacking sparse vectors vertically
    b_parts = [
        csc_matrix((num_dataset, 1)),             # spzeros(num_dataset)
        csc_matrix(np.ones((num_dataset, 1))),    # sparse(ones(num_dataset))
        csc_matrix(2 * np.ones((num_dataset, 1))),# 2 * sparse(ones(num_dataset))
        csc_matrix((dim_dataset, 1)),             # spzeros(dim_dataset)
        csc_matrix((dim_dataset, 1))              # spzeros(dim_dataset)
    ]

    # Stack the parts vertically to form the complete b vector
    b = vstack(b_parts, format='csc')

    # Normalize all column norms of A_T to 1
    # Compute L2 norm of each column
    col_norms = np.sqrt(A_T.power(2).sum(axis=0)).A1  # Shape: (num_variables,)

    # Avoid division by zero by setting inverse norms to 0 where norm is 0
    divrownorm_A = np.where(col_norms != 0, 1.0 / col_norms, 0.0)

    # Create a diagonal matrix with the inverse column norms
    diag_divrownorm_A = diags(divrownorm_A, offsets=0, format='csc')

    # Normalize A_T by multiplying with the diagonal matrix
    A_T = A_T.dot(diag_divrownorm_A)

    # Normalize b by multiplying with the diagonal matrix
    b = diag_divrownorm_A.dot(b)

    # Build the objective vector c by concatenating various parts
    c_parts = [
        (1.0 / num_dataset) * np.ones(num_dataset),  # 1 / num_dataset * ones(num_dataset)
        np.zeros(num_dataset),                        # zeros(num_dataset)
        np.zeros(num_dataset),                        # zeros(num_dataset)
        np.zeros(num_dataset),                        # zeros(num_dataset)
        np.zeros(dim_dataset),                        # zeros(dim_dataset)
        np.zeros(dim_dataset),                        # zeros(dim_dataset)
        np.zeros(dim_dataset),                        # zeros(dim_dataset)
        np.zeros(dim_dataset),                        # zeros(dim_dataset)
        np.array([rho]),                               # rho
        np.array([-rho])                               # -rho
    ]

    # Concatenate all parts to form the final c vector
    c = np.concatenate(c_parts)

    return A_T, b, c