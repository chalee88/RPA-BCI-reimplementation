import numpy as np 

from rpa.core.spd_matrices import is_spd, matrix_log, matrix_exp, matrix_sqrt, matrix_inv_sqrt

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