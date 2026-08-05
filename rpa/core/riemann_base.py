import numpy as np  
from scipy.linalg import norm
from rpa.core.spd_matrices import (
    matrix_inv_sqrt,
    matrix_log,
    is_spd,
    matrix_exp,
    matrix_sqrt
)

def  riemannian_distance(A, B):
    """
    Affine-Invariant Riemannian Distance (AIRM)
    """

    if not is_spd(A):
        raise ValueError("A is not SPD.")

    if not is_spd(B):
        raise ValueError("B is not SPD.")

    A_inv_sqrt = matrix_inv_sqrt(A)

    C = A_inv_sqrt @ B @ A_inv_sqrt

    return norm(matrix_log(C), "fro")

def log_map(P, X):
    """ 
    Logarithmic map of X at base point P 

    Parameters:
        P : ndarray
            Base SPD matrix
        X : ndarray
            SPD matrix to project

    Returns:
        ndarray
            Tangent-space matrix at P
    """

    if not is_spd(P):
        raise ValueError("P is not SPD")

    if not is_spd(X):
        raise ValueError("X is not SPD")

    P_sqrt = matrix_sqrt(P)
    P_inv_sqrt = matrix_inv_sqrt(P)

    return P_sqrt @ matrix_log(P_inv_sqrt @ X @ P_inv_sqrt) @ P_sqrt


def exp_map(P, V):
    """
    Exponential map of X at base point P
    
    Parameters:
        P : ndarray
            Base SPD matrix
        V : ndarray
            Tangent-space matrix
    
    Returns:
        ndarray
            SPD matrix
    """
    if not is_spd(P):
        raise ValueError("P is not SPD.")

    P_sqrt = matrix_sqrt(P)
    P_inv_sqrt = matrix_inv_sqrt(P)

    return P_sqrt @ matrix_exp(P_inv_sqrt @ V @ P_inv_sqrt) @ P_sqrt

    

def riemannian_mean(matrices, tol=1e-7, max_iter=500):
    """
    Compute the affine-invariant Riemannian mean of SPD matrices.

    Parameters
    ----------
    matrices : ndarray
        Array with shape
        (n_matrices, n_channels, n_channels)
    
    tol : float, default=1e-9
        Convergence tolerance. The algorithm stops when the average tanget
        update has Frobenius norm below this value

    max_iter : int, default = 100
        Maximum number of iterations
    
    Returns 
    -------
    ndarray
        Riemannian mean with shape
        (n_channels, n_channels)
    """

    matrices = np.asarray(matrices, dtype=float)
    
    n_matrices, n_rows, n_cols = matrices.shape

    if matrices.ndim != 3:
        raise ValueError("matrices must be 3-dimensional")
    
    if n_matrices == 0:
        raise ValueError("At least one matrix is required.")

    if n_rows != n_cols:
        raise ValueError("Each matrix must be square.")

    for index, matrix in enumerate(matrices):
        if not is_spd(matrix):
            raise ValueError(
                f"Matrix at index {index} is not SPD."
            )

    # Use the arithmetic mean as the initial estimate 
    mean = np.mean(matrices, axis=0)

    for _ in range(max_iter):

        tangent_sum = np.zeros_like(mean)

        for matrix in matrices:
            tangent_sum += log_map(mean, matrix)

        tangent_average = tangent_sum / n_matrices

        update_norm = np.linalg.norm(
            tangent_average, 
            ord="fro"
        )

        if update_norm < tol:
            return mean
        
        mean = exp_map(mean, tangent_average)

        # Remove tiny numerical asymmetry that could have happened due to floating point errors 
        # in calculating complex matrix calculations
        mean = 0.5 * (mean + mean.T)

    raise RuntimeError(
        "Riemannian mean did not converge within"
        f"{max_iter} iterations."
    )

    


