import numpy as np

from rpa.transfer_learning.procrustes import (
    estimate_rotation,
    rotation_objective_value,
)

from rpa.transfer_learning.alignment import rotate_covariances


def test_estimate_rotation_returns_orthogonal_matrix():

    source_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        [
            [1.0, 0.0],
            [0.0, 3.0],
        ],
    ])

    source_labels = np.array([0, 1])

    target_covariances = source_covariances.copy()
    target_labels = source_labels.copy()

    U = estimate_rotation(
        source_covariances,
        source_labels,
        target_covariances,
        target_labels,
    )

    assert U.shape == (2, 2)
    assert np.allclose(U.T @ U, np.eye(2), atol=1e-6)


def test_estimate_rotation_identity_case_keeps_loss_low():

    source_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        [
            [1.0, 0.0],
            [0.0, 3.0],
        ],
    ])

    source_labels = np.array([0, 1])

    target_covariances = source_covariances.copy()
    target_labels = source_labels.copy()

    U = estimate_rotation(
        source_covariances,
        source_labels,
        target_covariances,
        target_labels,
    )

    loss = rotation_objective_value(
        source_covariances,
        source_labels,
        target_covariances,
        target_labels,
        U,
    )

    assert loss < 1e-8


def test_estimate_rotation_reduces_loss_for_rotated_target():

    source_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        [
            [1.0, 0.0],
            [0.0, 3.0],
        ],
    ])

    source_labels = np.array([0, 1])

    angle = np.pi / 4.0

    true_U = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])

    target_covariances = np.array([
        true_U @ covariance @ true_U.T
        for covariance in source_covariances
    ])

    target_labels = source_labels.copy()

    identity = np.eye(2)

    loss_before = rotation_objective_value(
        source_covariances,
        source_labels,
        target_covariances,
        target_labels,
        identity,
    )

    estimated_U = estimate_rotation(
        source_covariances,
        source_labels,
        target_covariances,
        target_labels,
    )

    loss_after = rotation_objective_value(
        source_covariances,
        source_labels,
        target_covariances,
        target_labels,
        estimated_U,
    )

    assert loss_after < loss_before


def test_estimated_rotation_can_align_rotated_target():

    source_covariances = np.array([
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        [
            [1.0, 0.0],
            [0.0, 3.0],
        ],
    ])

    source_labels = np.array([0, 1])

    angle = np.pi / 4.0

    true_U = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])

    target_covariances = np.array([
        true_U @ covariance @ true_U.T
        for covariance in source_covariances
    ])

    target_labels = source_labels.copy()

    estimated_U = estimate_rotation(
        source_covariances,
        source_labels,
        target_covariances,
        target_labels,
    )

    rotated_target = rotate_covariances(
        target_covariances,
        estimated_U,
    )

    assert rotated_target.shape == target_covariances.shape