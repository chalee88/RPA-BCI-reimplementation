import numpy as np

from rpa.transfer_learning.alignment import center_covariances
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
    