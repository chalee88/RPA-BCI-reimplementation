import numpy as np

from rpa.core.riemann_base import (
    riemannian_mean,
    riemannian_distance,
)

from rpa.core.spd_matrices import (
    is_spd,
    matrix_sqrt,
    matrix_inv_sqrt,
    matrix_power,
    nearest_spd,
)


def _validate_covariance_batch(covariances, name="covariances"):
    """
    Validate a batch of SPD covariance matrices.

    Parameters
    ----------
    covariances : ndarray
        Expected shape: (n_matrices, n_channels, n_channels)

    name : str
        Name used in error messages.

    Returns
    -------
    ndarray
        Validated covariance batch as float ndarray.
    """

    covariances = np.asarray(covariances, dtype=float)

    if covariances.ndim != 3:
        raise ValueError(
            f"{name} must have shape "
            "(n_matrices, n_channels, n_channels)."
        )

    n_matrices, n_rows, n_cols = covariances.shape

    if n_matrices == 0:
        raise ValueError(f"{name} must contain at least one matrix.")

    if n_rows != n_cols:
        raise ValueError(f"Each matrix in {name} must be square.")

    for index, covariance in enumerate(covariances):
        if not is_spd(covariance):
            raise ValueError(
                f"Matrix at index {index} in {name} is not SPD."
            )

    return covariances


def center_covariances(covariances, mean=None):
    """
    Center SPD covariance matrices around the identity matrix.

    Parameters
    ----------
    covariances : ndarray
        Array of SPD matrices with shape
        (n_matrices, n_channels, n_channels).

    mean : ndarray or None, default=None
        Riemannian mean used for centering.
        If None, it is computed from covariances.

    Returns
    -------
    centered_covariances : ndarray
        Centered covariance matrices.

    mean : ndarray
        Riemannian mean used for centering.
    """

    covariances = _validate_covariance_batch(
        covariances,
        name="covariances",
    )

    if mean is None:
        mean = riemannian_mean(covariances)
    else:
        mean = np.asarray(mean, dtype=float)

        if not is_spd(mean):
            raise ValueError("Provided mean is not SPD.")

        if mean.shape != covariances.shape[1:]:
            raise ValueError(
                "Provided mean must have shape "
                "(n_channels, n_channels)."
            )

    mean_inv_sqrt = matrix_inv_sqrt(mean)

    centered_covariances = np.array([
        mean_inv_sqrt @ covariance @ mean_inv_sqrt
        for covariance in covariances
    ])

    centered_covariances = 0.5 * (
        centered_covariances
        + np.transpose(centered_covariances, axes=(0, 2, 1))
    )

    return centered_covariances, mean


def recolor_covariances(centered_covariances, target_mean):
    """
    Recolor centered SPD covariance matrices using a target mean.

    This is the inverse-like operation of centering:

        C_recolored = G^{1/2} C_centered G^{1/2}

    Parameters
    ----------
    centered_covariances : ndarray
        Centered SPD matrices with shape
        (n_matrices, n_channels, n_channels).

    target_mean : ndarray
        Target SPD mean with shape
        (n_channels, n_channels).

    Returns
    -------
    recolored_covariances : ndarray
        Recolored SPD covariance matrices.
    """

    centered_covariances = _validate_covariance_batch(
        centered_covariances,
        name="centered_covariances",
    )

    target_mean = np.asarray(target_mean, dtype=float)

    if target_mean.shape != centered_covariances.shape[1:]:
        raise ValueError(
            "target_mean must have shape "
            "(n_channels, n_channels)."
        )

    if not is_spd(target_mean):
        raise ValueError("target_mean is not SPD.")

    target_sqrt = matrix_sqrt(target_mean)

    recolored_covariances = np.array([
        nearest_spd(target_sqrt @ covariance @ target_sqrt)
        for covariance in centered_covariances
    ])

    return recolored_covariances


def align_mean_to_reference(covariances, reference_covariances):
    """
    Align the Riemannian mean of one covariance distribution
    to the mean of a reference distribution.

    This is a generic mean-alignment helper, not the full RPA
    algorithm from the paper.

    Parameters
    ----------
    covariances : ndarray
        SPD covariance matrices to transform.

    reference_covariances : ndarray
        Reference SPD covariance matrices.

    Returns
    -------
    aligned_covariances : ndarray
        Covariances whose mean is aligned to the reference mean.

    covariance_mean : ndarray
        Mean of the original covariances.

    reference_mean : ndarray
        Mean of the reference covariances.
    """

    covariances = _validate_covariance_batch(
        covariances,
        name="covariances",
    )

    reference_covariances = _validate_covariance_batch(
        reference_covariances,
        name="reference_covariances",
    )

    if covariances.shape[1:] != reference_covariances.shape[1:]:
        raise ValueError(
            "covariances and reference_covariances must have "
            "the same matrix shape."
        )

    centered_covariances, covariance_mean = center_covariances(
        covariances
    )

    reference_mean = riemannian_mean(reference_covariances)

    aligned_covariances = recolor_covariances(
        centered_covariances,
        reference_mean,
    )

    return aligned_covariances, covariance_mean, reference_mean


def dispersion(covariances, mean=None):
    """
    Compute the dispersion of SPD covariance matrices.

    Dispersion is the sum of squared Riemannian distances from
    a reference mean.

    Parameters
    ----------
    covariances : ndarray
        SPD covariance matrices with shape
        (n_matrices, n_channels, n_channels).

    mean : ndarray or None
        Reference mean. If None, the Riemannian mean is computed.

    Returns
    -------
    float
        Sum of squared Riemannian distances.
    """

    covariances = _validate_covariance_batch(
        covariances,
        name="covariances",
    )

    if mean is None:
        mean = riemannian_mean(covariances)
    else:
        mean = np.asarray(mean, dtype=float)

        if not is_spd(mean):
            raise ValueError("mean is not SPD.")

        if mean.shape != covariances.shape[1:]:
            raise ValueError(
                "mean must have shape "
                "(n_channels, n_channels)."
            )

    total = 0.0

    for covariance in covariances:
        total += riemannian_distance(mean, covariance) ** 2

    return total


def stretch_covariances(centered_covariances, scale):
    """
    Stretch covariance matrices along geodesics from identity.

    In the RPA paper, this is applied to the target dataset
    after recentering.

    Parameters
    ----------
    centered_covariances : ndarray
        SPD matrices centered around identity.

    scale : float
        Stretching exponent.

    Returns
    -------
    ndarray
        Stretched covariance matrices.
    """

    centered_covariances = _validate_covariance_batch(
        centered_covariances,
        name="centered_covariances",
    )

    stretched = np.array([
        matrix_power(covariance, scale)
        for covariance in centered_covariances
    ])

    stretched = 0.5 * (
        stretched + np.transpose(stretched, axes=(0, 2, 1))
    )

    return stretched


def class_means(covariances, labels):
    """
    Compute one Riemannian mean per class.

    Parameters
    ----------
    covariances : ndarray
        SPD covariance matrices with shape
        (n_matrices, n_channels, n_channels).

    labels : ndarray
        Class labels with shape (n_matrices,).

    Returns
    -------
    dict
        Mapping from class label to Riemannian class mean.
    """

    covariances = _validate_covariance_batch(
        covariances,
        name="covariances",
    )

    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")

    if covariances.shape[0] != labels.shape[0]:
        raise ValueError(
            "covariances and labels must contain the same number of samples."
        )

    means = {}

    for label in np.unique(labels):
        means[label] = riemannian_mean(covariances[labels == label])

    return means


def rotate_covariances(covariances, U):
    """
    Rotate SPD covariances using:

        C_rotated = U.T @ C @ U

    Parameters
    ----------
    covariances : ndarray
        SPD covariance matrices with shape
        (n_matrices, n_channels, n_channels).

    U : ndarray
        Orthogonal matrix with shape
        (n_channels, n_channels).

    Returns
    -------
    ndarray
        Rotated covariance matrices.
    """

    covariances = _validate_covariance_batch(
        covariances,
        name="covariances",
    )

    U = np.asarray(U, dtype=float)

    n_channels = covariances.shape[1]

    if U.shape != (n_channels, n_channels):
        raise ValueError("U has incompatible shape.")

    if not np.allclose(U.T @ U, np.eye(n_channels), atol=1e-8):
        raise ValueError("U must be orthogonal.")

    rotated = np.array([
        nearest_spd(U.T @ covariance @ U)
        for covariance in covariances
    ])
    
    return rotated