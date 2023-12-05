import numpy as np
import scipy as sp
from scipy.sparse.linalg import norm
import matplotlib.pyplot as plt


def myPCG(A, b, L, U, tol, maxit):
    # A:       Matrix produced by diffusionMatrix
    # b:       RHS calculated by the source function
    # L and U: factors of the preconditioner from sp.spilu
    # tol:     tolerance for convergence
    # maxit:   maximum number of iterations

    # Initialize parameters
    M = L @ U
    x = np.zeros_like(b)  # Initial guess
    r0 = b - (A @ x)  # Initial residual
    z0 = sp.sparse.linalg.spsolve(M, r0)  # Apply preconditioner
    d0 = z0.copy()
    i = 0
    res = []

    # PCG Loop
    while i < maxit:
        alpha = (r0.T @ z0) / (d0.T @ A @ d0)
        x += alpha * d0
        r1 = r0 - alpha * (A @ d0)
        norm = np.linalg.norm(r1, 2)
        res.append(norm)
        # Check for convergence
        if norm < tol:
            break

        z1 = sp.sparse.linalg.spsolve(M, r1)
        beta = (r1.T @ z1) / (r0.T @ z0)
        d0 = z1 + beta * d0
        r0, z0 = r1, z1
        i += 1

    print(f"Number of iterations = {i}, Norm = {norm}")

    return x
