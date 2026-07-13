import numpy as np
from rpa.core.riemann_base import riemann_distance 

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