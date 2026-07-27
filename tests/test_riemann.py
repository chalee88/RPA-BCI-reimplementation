import numpy as np
from rpa.core.riemann_base import (
    riemann_distance,
    log_map,
    exp_map, 
    riemannian_mean,
) 
from rpa.core.spd_matrices import is_spd

def test_distance_zero():
    A = np.array([
        [2.0, 0.5],
        [0.5, 3.0]
    ])

    d = riemann_distance(A, A)

    assert np.isclose(d, 0.0)

def test_distance_symmetric():

    A = np.array([
        [2, 1],
        [1, 2],
    ], dtype=float)

    B = np.array([
        [5, 2],
        [2, 4],
    ], dtype=float)

    d1 = riemann_distance(A, B)
    d2 = riemann_distance(B, A)

    assert np.allclose(d1, d2)

def test_distance_positive():

    A = np.eye(2)
    B = np.array([
        [3, 0],
        [0, 2]
    ])

    d = riemann_distance(A, B)

    assert d >= 0

def test_mean_of_one_matrix():
     
    A = np.array([
        [2.0, 0.5],
        [0.5, 3.0],
    ])

    mean = riemannian_mean(np.array([A]))

    assert np.allclose(mean, A, atol=1e-8)

def test_mean_of_identical_matrices():

    A = np.array([
        [3.0, 1.0],
        [1.0, 4.0],
    ])

    matrices = np.array([A, A, A])

    mean = riemannian_mean(matrices)

    assert np.allclose(mean, A, atol=1e-8)


def test_riemannian_mean_is_spd():

    matrices = np.array([
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

    mean = riemannian_mean(matrices)

    assert is_spd(mean)

def test_mean_of_two_diagonal_matrices():

    A = np.array([
        [1.0, 0.0],
        [0.0, 4.0],
    ])

    B = np.array([
        [9.0, 0.0],
        [0.0, 16.0],
    ])

    expected = np.array([
        [3.0, 0.0],
        [0.0, 8.0],
    ])

    mean = riemannian_mean(np.array([A, B]))

    assert np.allclose(mean, expected, atol=1e-8)