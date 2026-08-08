import numpy as np  
from scipy.linalg import norm
from rpa.core.spd_matrices import (
    matrix_inv_sqrt,
    matrix_log,
    is_spd,
    matrix_exp,
    matrix_sqrt,
    nearest_spd,
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

    

def riemannian_mean(matrices, tol=1e-6, max_iter=1000):
    """
    Compute the affine-invariant Riemannian mean of SPD matrices.

    This implementation performs the iterative update in the tangent
    space at the identity:

        mean_next = mean^(1/2) exp(avg_log) mean^(1/2)

    where

        avg_log = mean_i log(mean^(-1/2) C_i mean^(-1/2))

    This is numerically more stable than accumulating full tangent matrices
    at the current mean. 
    """

    matrices = np.asarray(matrices, dtype=float)

    if matrices.ndim != 3:
        raise ValueError(
            "matrices must have shape"
            "(n_matrices, n_channels, n_channels)."
        )
    
    n_matrices, n_rows, n_cols = matrices.shape
    
    if n_matrices == 0:
        raise ValueError("At least one matrix is required.")

    if n_rows != n_cols:
        raise ValueError("Each matrix must be square.")

    repaired_matrices = []

    for index, matrix in enumerate(matrices):
        matrix = nearest_spd(matrix)
        if not is_spd(matrix):
            raise ValueError(
                f"Matrix at index {index} is not SPD."
            )

        repaired_matrices.append(matrix)

    matrices = np.array(repaired_matrices)

    if n_matrices == 1:
        return matrices[0]

    mean = nearest_spd(np.mean(matrices, axis=0))

    final_update_norm = None

    for _ in range(max_iter):
        mean_sqrt = matrix_sqrt(mean)
        mean_inv_sqrt = matrix_inv_sqrt(mean)

        tangent_sum = np.zeros_like(mean)

        for matrix in matrices: 
            normalized = mean_inv_sqrt @ matrix @ mean_inv_sqrt
            normalized = nearest_spd(normalized)

            tangent_sum += matrix_log(normalized)

        tangent_average = tangent_sum / n_matrices 

        update_norm = np.linalg.norm(tangent_average, ord='fro')
        final_update_norm = update_norm

        if update_norm < tol:
            return nearest_spd(mean)

        mean = mean_sqrt @ matrix_exp(tangent_average) @ mean_sqrt
        mean = nearest_spd(mean)

    raise RuntimeError(
        "Riemannian mean did not converge within"
        f"{max_iter} iterations. "
        f"Final update norm: {final_update_norm:.3e}." 
    )
    
    


