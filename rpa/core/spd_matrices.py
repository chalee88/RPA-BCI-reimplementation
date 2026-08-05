import numpy as np
from scipy.linalg import expm


def is_symmetric(A, tol=1e-8):
    """
    Check whether a matrix is symmetric.
    """

    A = np.asarray(A)

    return np.allclose(A, A.T, atol=tol)


def is_positive_definite(A, tol=1e-10):
    """
    Check whether all eigenvalues of a symmetric matrix are positive.
    """

    A = np.asarray(A)

    eigenvalues = np.linalg.eigvalsh(A)

    return np.all(eigenvalues > tol)


def is_spd(A):
    """
    Check whether a matrix is symmetric positive definite.
    """

    return is_symmetric(A) and is_positive_definite(A)


def nearest_spd(A, min_eigenvalue=1e-8):
    """
    Project a symmetric matrix to a numerically safe SPD matrix.

    This is used to remove tiny negative eigenvalues caused by
    floating-point round-off during SPD transformations.
    """

    A = _symmetrize(A)

    eigenvalues, eigenvectors = np.linalg.eigh(A)

    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

    result = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    return _symmetrize(result)


def _symmetrize(A):
    """
    Remove tiny numerical asymmetry.
    """

    A = np.asarray(A, dtype=float)

    return 0.5 * (A + A.T)


def _eigendecompose_spd(A, min_eigenvalue=1e-12):
    """
    Eigendecompose an SPD matrix and clip tiny eigenvalues.

    This function assumes the input should be SPD. It raises an error
    if the matrix is not symmetric or not positive definite.
    """

    A = np.asarray(A, dtype=float)

    if not is_symmetric(A):
        raise ValueError("Matrix is not symmetric.")

    A = _symmetrize(A)

    eigenvalues, eigenvectors = np.linalg.eigh(A)

    if np.any(eigenvalues <= 0):
        raise ValueError("Matrix is not positive definite.")

    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

    return eigenvalues, eigenvectors


def _apply_eigenvalue_function(A, function):
    """
    Apply a scalar function to the eigenvalues of an SPD matrix.
    """

    eigenvalues, eigenvectors = _eigendecompose_spd(A)

    transformed_eigenvalues = function(eigenvalues)

    result = (
        eigenvectors
        @ np.diag(transformed_eigenvalues)
        @ eigenvectors.T
    )

    return _symmetrize(result)


def matrix_sqrt(A):
    """
    Compute the matrix square root of an SPD matrix.
    """

    return _apply_eigenvalue_function(A, np.sqrt)


def matrix_inv_sqrt(A):
    """
    Compute the inverse square root of an SPD matrix.
    """

    return _apply_eigenvalue_function(
        A,
        lambda eigenvalues: 1.0 / np.sqrt(eigenvalues),
    )


def matrix_log(A):
    """
    Compute the matrix logarithm of an SPD matrix.
    """

    return _apply_eigenvalue_function(A, np.log)


def matrix_exp(A):
    """
    Compute the matrix exponential.

    The input here is usually a symmetric tangent-space matrix.
    """

    A = _symmetrize(A)

    result = expm(A)

    return _symmetrize(np.real_if_close(result, tol=1000))


def matrix_power(A, power):
    """
    Compute A^power for an SPD matrix.
    """

    return _apply_eigenvalue_function(
        A,
        lambda eigenvalues: eigenvalues**power,
    )