import numpy as np
from rpa.core.riemann_base import riemannian_mean
from rpa.core.spd_matrices import is_spd, matrix_sqrt, matrix_inv_sqrt


def center_covariances(covariances, mean=None):
    """
    Center SPD covariance matrices around the identity matrix

    Parameters
    ----------
    covariances : ndarray
        Array of SPD matrices with shape
        (n_matrices, n_channels, n_channels)

    mean : ndarray or None, default=None
        Riemannian mean used for centering
        If None, it is computed from covariance

    Returns
    -------
    centered_covariances : ndarray
        Centered covariance matrices.

    mean : ndarray
        Riemannian mean used for centering
    """

    covariances = np.asarray(covariances, dtype=float)
    if covariances.ndim != 3:
        raise ValueError(
            "covariances must have shape "
            "(n_matrices, n_channels, n_channels)."
        )

    n_matrices, n_rows, n_cols = covariances.shape

    if n_rows != n_cols:
        raise ValueError(
            "Each matrix must be square."
        )
   
    for index, covariance in enumerate(covariances):
        if not is_spd(covariance):
            raise ValueError(
                f"Covariance matrix at index {index} is not SPD."
            )
    
    if mean is None:
        mean = riemannian_mean(covariances) # computes the center of the covariance distribution
    else:
        mean = np.asarray(mean, dtype=float)

        if not is_spd(mean):
            raise ValueError("Provided mean is not SPD.")
    
    mean_inv_sqrt = matrix_inv_sqrt(mean)

    centered_covariances = np.array([
        mean_inv_sqrt @ covariance @ mean_inv_sqrt
        for covariance in covariances
    ])

    # Clean tiny numerical asymmetry
    centered_covariances = 0.5 * (
        centered_covariances 
        + np.transpose(centered_covariances, axes=(0, 2, 1))
    )

    return centered_covariances, mean 
    
    