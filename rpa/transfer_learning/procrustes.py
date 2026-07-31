import warnings

import autograd.numpy as anp
import numpy as np

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import SteepestDescent

from rpa.core.riemann_base import riemannian_distance

from rpa.transfer_learning.alignment import (
    class_means,
    rotate_covariances,
)


def _validate_class_mean_dictionaries(source_means, target_means):
    """
    Validate that source and target class-mean dictionaries are compatible.
    """

    source_classes = set(source_means.keys())
    target_classes = set(target_means.keys())

    if source_classes != target_classes:
        raise ValueError(
            "Source and target must contain the same class labels "
            "for rotation estimation."
        )

    classes = np.array(sorted(source_classes))

    first_label = classes[0]
    matrix_shape = source_means[first_label].shape

    for label in classes:
        if source_means[label].shape != matrix_shape:
            raise ValueError(
                "All source class means must have the same shape."
            )

        if target_means[label].shape != matrix_shape:
            raise ValueError(
                "Source and target class means must have the same shape."
            )

    return classes


def _prepare_weights(classes, weights=None):
    """
    Prepare class weights.
    """

    if weights is None:
        prepared_weights = np.ones(len(classes), dtype=float)

    elif isinstance(weights, dict):
        prepared_weights = np.array(
            [weights[label] for label in classes],
            dtype=float,
        )

    else:
        prepared_weights = np.asarray(weights, dtype=float)

        if prepared_weights.shape != (len(classes),):
            raise ValueError(
                "weights must have shape (n_classes,) if provided as an array."
            )

    if np.any(prepared_weights < 0):
        raise ValueError("weights must be non-negative.")

    weight_sum = np.sum(prepared_weights)

    if np.isclose(weight_sum, 0.0):
        raise ValueError("At least one weight must be positive.")

    return prepared_weights / weight_sum


def _is_isotropic(matrix, tol=1e-8):
    """
    Check whether a matrix is approximately a scalar multiple
    of the identity matrix.

    A matrix of the form aI has no orientation information.
    Rotating it does not change it.
    """

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        return False

    if matrix.shape[0] != matrix.shape[1]:
        return False

    n_channels = matrix.shape[0]

    scalar = np.trace(matrix) / n_channels

    isotropic_matrix = scalar * np.eye(n_channels)

    return np.allclose(
        matrix,
        isotropic_matrix,
        atol=tol,
        rtol=tol,
    )


def _rotation_is_unidentifiable(source_means, target_means, classes, tol=1e-8):
    """
    Check whether the class means contain enough orientation information
    to estimate a rotation.

    If both source and target class means are scalar multiples of identity,
    then there is no meaningful orientation to align.
    """

    source_all_isotropic = True
    target_all_isotropic = True

    for label in classes:
        if not _is_isotropic(source_means[label], tol=tol):
            source_all_isotropic = False

        if not _is_isotropic(target_means[label], tol=tol):
            target_all_isotropic = False

    return source_all_isotropic and target_all_isotropic


def _project_to_orthogonal(matrix):
    """
    Project a square matrix to the nearest orthogonal matrix.

    This uses the polar decomposition through SVD:

        matrix = L S R.T
        Q = L R.T

    Q is the closest orthogonal matrix to matrix in Frobenius norm.
    """

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")

    if not np.all(np.isfinite(matrix)):
        warnings.warn(
            "Optimizer returned invalid rotation values. "
            "Falling back to identity rotation.",
            RuntimeWarning,
        )

        return np.eye(matrix.shape[0])

    try:
        left, _, right_t = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        warnings.warn(
            "SVD projection failed. Falling back to identity rotation.",
            RuntimeWarning,
        )

        return np.eye(matrix.shape[0])

    orthogonal = left @ right_t

    return orthogonal


def _euclidean_distance_squared_autograd(A, B):
    """
    Squared Frobenius distance between two matrices.

    This is compatible with Autograd and is used as the default
    rotation objective.
    """

    difference = A - B

    return anp.sum(difference * difference)


def estimate_rotation_from_class_means(
    source_means,
    target_means,
    weights=None,
    max_iterations=10000,
    min_gradient_norm=1e-9,
):
    """
    Estimate an orthogonal rotation matrix from source and target class means.

    The optimized objective is:

        sum_k w_k ||source_mean_k - U.T @ target_mean_k @ U||_F^2

    This follows the paper-style Procrustes idea while using an
    autograd-compatible Euclidean/Frobenius objective for the class-mean
    matching step.
    """

    classes = _validate_class_mean_dictionaries(
        source_means,
        target_means,
    )

    prepared_weights = _prepare_weights(classes, weights)

    n_channels = source_means[classes[0]].shape[0]

    if n_channels == 1:
        return np.eye(n_channels)

    if _rotation_is_unidentifiable(
        source_means,
        target_means,
        classes,
    ):
        return np.eye(n_channels)

    initial_loss = 0.0

    for weight, label in zip(prepared_weights, classes):
        difference = source_means[label] - target_means[label]
        initial_loss += weight * np.sum(difference * difference)

    if np.isclose(initial_loss, 0.0, atol=1e-12):
        return np.eye(n_channels)

    source_stack = anp.stack([
        anp.asarray(source_means[label])
        for label in classes
    ])

    target_stack = anp.stack([
        anp.asarray(target_means[label])
        for label in classes
    ])

    weight_vector = anp.asarray(prepared_weights)

    manifold = Stiefel(n_channels, n_channels)

    @pymanopt.function.autograd(manifold)
    def cost(U):
        total = anp.array(0.0)

        for index in range(len(classes)):
            rotated_target_mean = (
                U.T @ target_stack[index] @ U
            )

            distance_squared = _euclidean_distance_squared_autograd(
                source_stack[index],
                rotated_target_mean,
            )

            total = total + weight_vector[index] * distance_squared

        return total

    problem = Problem(
        manifold=manifold,
        cost=cost,
    )

    optimizer = SteepestDescent(
        max_iterations=max_iterations,
        min_gradient_norm=min_gradient_norm,
        verbosity=0,
    )

    initial_point = np.eye(n_channels)

    result = optimizer.run(
        problem,
        initial_point=initial_point,
    )

    U = _project_to_orthogonal(result.point)

    if not np.allclose(U.T @ U, np.eye(n_channels), atol=1e-8):
        warnings.warn(
            "Estimated rotation matrix is not perfectly orthogonal.",
            RuntimeWarning,
        )

    return U


def estimate_rotation(
    source_covariances,
    source_labels,
    target_covariances,
    target_labels,
    weights=None,
    max_iterations=10000,
    min_gradient_norm=1e-9,
):
    """
    Estimate the RPA rotation matrix from labeled source and target data.

    This function first computes class-wise Riemannian means for source
    and target data, then estimates the orthogonal matrix U.
    """

    source_means = class_means(
        source_covariances,
        source_labels,
    )

    target_means = class_means(
        target_covariances,
        target_labels,
    )

    U = estimate_rotation_from_class_means(
        source_means,
        target_means,
        weights=weights,
        max_iterations=max_iterations,
        min_gradient_norm=min_gradient_norm,
    )

    return U


def rotation_objective_value(
    source_covariances,
    source_labels,
    target_covariances,
    target_labels,
    U,
    weights=None,
    metric="euclid",
):
    """
    Compute the class-mean matching loss for a given rotation matrix.

    Parameters
    ----------
    metric : {"euclid", "riemann"}
        If "euclid", compute squared Frobenius mismatch.
        If "riemann", compute squared Riemannian distance.
        The optimization itself currently uses "euclid" because it is
        compatible with Autograd.
    """

    source_means = class_means(
        source_covariances,
        source_labels,
    )

    target_means = class_means(
        target_covariances,
        target_labels,
    )

    classes = _validate_class_mean_dictionaries(
        source_means,
        target_means,
    )

    prepared_weights = _prepare_weights(classes, weights)

    rotated_target_covariances = rotate_covariances(
        target_covariances,
        U,
    )

    rotated_target_means = class_means(
        rotated_target_covariances,
        target_labels,
    )

    total = 0.0

    for weight, label in zip(prepared_weights, classes):
        if metric == "euclid":
            difference = source_means[label] - rotated_target_means[label]
            distance_squared = np.sum(difference * difference)

        elif metric == "riemann":
            distance = riemannian_distance(
                source_means[label],
                rotated_target_means[label],
            )

            distance_squared = distance**2

        else:
            raise ValueError("metric must be either 'euclid' or 'riemann'.")

        total += weight * distance_squared

    return total