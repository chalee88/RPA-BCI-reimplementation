import numpy as np  
from scipy.linalg import norm
from rpa.core.spd_matrices import (
    matrix_inv_sqrt,
    matrix_log,
    is_spd
)

def  riemann_distance(A, B):
    """
    Affine-Invariant Riemannian Distance (AIRM)
    """

    if not is_spd(A):
        raise ValueError("A is not SPD.")

    if not is_spd(B):
        raise ValueError("B is not SPD.")

    A_inv_sqrt = matrix_inv_sqrt(A)

    C = A_inv_sqrt @ B @ A_inv_sqrt

    return norm(matrix_log(C), "fro")