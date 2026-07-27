import numpy as np 

from rpa.core.spd_matrices import is_spd, matrix_log, matrix_exp, matrix_sqrt, matrix_inv_sqrt
from rpa.core.spd_matrices import matrix_power

def test_spd():

    A = np.array([
        [2, 1],
        [1, 2]
    ])

    assert is_spd(A) 

def test_not_spd():
    
    A = np.array([
        [1, 2],
        [3, 4]
    ])

    assert not is_spd(A)

def test_log_exp_inverse():

    A = np.array([
        [2.0, 0.1],
        [0.0, 5.0]
    ])

    L = matrix_log(A)
    A2 = matrix_exp(L)

    assert np.allclose(A, A2)

def test_matrix_sqrt():

    A =  np.array([
        [4.0, 0.0],
        [0.0, 9.0]
    ])

    S = matrix_sqrt(A)

    assert np.allclose(S @ S, A)

def test_inverse_sqrt():

    A = np.array([
            [4.0, 0.0],
        [0.0, 9.0]
    ])

    S = matrix_inv_sqrt(A)

    identity = S @ A @ S

    assert np.allclose(identity, np.eye(2))

def test_matrix_power_half():

    A = np.array([
        [4.0, 0.0],
        [0.0, 9.0],
    ])

    A_half = matrix_power(A, 0.5)

    expected = np.array([
        [2.0, 0.0],
        [0.0, 3.0],
    ])

    assert np.allclose(A_half, expected, atol=1e-8)


def test_matrix_power_zero():

    A = np.array([
        [4.0, 0.0],
        [0.0, 9.0],
    ])

    result = matrix_power(A, 0.0)

    assert np.allclose(result, np.eye(2), atol=1e-8)


def test_matrix_power_one():

    A = np.array([
        [4.0, 0.0],
        [0.0, 9.0],
    ])

    result = matrix_power(A, 1.0)

    assert np.allclose(result, A, atol=1e-8)