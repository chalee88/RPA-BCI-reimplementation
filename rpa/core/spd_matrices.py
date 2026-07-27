import numpy as np 
from scipy.linalg import sqrtm, fractional_matrix_power, logm, expm

def is_symmetric(A, tol=1e-8):
    """ 
    Check whether A is symmetric
    """
    return np.allclose(A, A.T, atol=tol)

def is_positive_definite(A):
    """
    Check whether all eigenvalues are positive
    """
    eigvals = np.linalg.eigvalsh(A)
    return np.all(eigvals > 0)

def is_spd(A):
    """
    True only if A is symmetric positive definite 
    """
    return is_symmetric(A) and is_positive_definite(A)

def matrix_sqrt(A):
    """
    Matrix square root 
    """
    return sqrtm(A)

def matrix_inv_sqrt(A):
    """ 
    Compute A^(-1/2) 
    """
    return fractional_matrix_power(A, -0.5)

def matrix_log(A):
    """
    Matrix logarithm of an SPD matrix
    """
    return logm(A)

def matrix_exp(A):
    """
    Matrix exponential
    """
    return expm(A)

def matrix_power(A, power):
    """
    Compute A^power for an SPD matrix.

    For SPD matrices:
        A^p = exp(p log(A))

    Parameters
    ----------
    A : ndarray
        SPD matrix.

    power : float
        Exponent.

    Returns
    -------
    ndarray
        Matrix power A^power.
    """

    if not is_spd(A):
        raise ValueError("A is not SPD.")

    result = matrix_exp(power * matrix_log(A))

    # Remove tiny numerical asymmetry.
    result = 0.5 * (result + result.T)

    return result