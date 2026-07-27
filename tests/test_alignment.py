import numpy as np

from rpa.transfer_learning.alignment import (
    center_covariances,
    recolor_covariances
)
from rpa.core.riemann_base import riemannian_mean
from rpa.core.spd_matrices import is_spd

def test_center_covariances_output_shape():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    centered, mean = center_covariances(covariances)

    assert centered.shape == covariances.shape
    assert mean.shape == covariances.shape[1:]

def test_centered_covariances_are_spd():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    centered, _ = center_covariances(covariances)

    for matrix in centered:
        assert is_spd(matrix)

def test_centered_mean_is_identity():

    covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    centered, _ = center_covariances(covariances)

    centered_mean = riemannian_mean(centered)

    assert np.allclose(centered_mean, np.eye(2), atol=1e-6)

def test_recolor_covariances_output_shape():

    centered_covariances = np.array([
        [
            [1.0, 0.1],
            [0.1, 1.5],
        ],
        [
            [1.2, 0.2],
            [0.2, 1.8],
        ],
    ])

    target_mean = np.array([
        [2.0, 0.3],
        [0.3, 3.0],
    ])

    recolored = recolor_covariances(
        centered_covariances,
        target_mean,
    )

    assert recolored.shape == centered_covariances.shape

def test_recolored_covariances_are_spd():

    centered_covariances = np.array([
        [
            [1.0, 0.1],
            [0.1, 1.5],
        ],
        [
            [1.2, 0.2],
            [0.2, 1.8],
        ],
    ])

    target_mean = np.array([
        [2.0, 0.3],
        [0.3, 3.0],
    ])

    recolored = recolor_covariances(
        centered_covariances,
        target_mean,
    )

    for matrix in recolored:
        assert is_spd(matrix)

def test_recolor_identity_gives_target_mean():

    centered_covariances = np.array([
        np.eye(2),
    ])

    target_mean = np.array([
        [2.0, 0.3],
        [0.3, 3.0],
    ])

    recolored = recolor_covariances(
        centered_covariances,
        target_mean,
    )

    assert np.allclose(
        recolored[0],
        target_mean,
        atol=1e-8,
    )

def test_center_then_recolor_mean_matches_target_mean():

    source_covariances = np.array([
        [
            [2.0, 0.2],
            [0.2, 3.0],
        ],
        [
            [4.0, 0.5],
            [0.5, 5.0],
        ],
        [
            [3.0, 0.1],
            [0.1, 6.0],
        ],
    ])

    target_covariances = np.array([
        [
            [6.0, 0.4],
            [0.4, 7.0],
        ],
        [
            [8.0, 0.6],
            [0.6, 9.0],
        ],
        [
            [7.0, 0.3],
            [0.3, 10.0],
        ],
    ])

    centered_source, source_mean = center_covariances(
        source_covariances
    )

    target_mean = riemannian_mean(target_covariances)

    recolored_source = recolor_covariances(
        centered_source,
        target_mean,
    )

    recolored_mean = riemannian_mean(recolored_source)

    assert np.allclose(
        recolored_mean,
        target_mean,
        atol=1e-6,
    )

