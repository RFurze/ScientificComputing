import numpy as np
import scipy as sp
from scipy.sparse.linalg import norm
import matplotlib.pyplot as plt
import time


def myPCG(A, b, L, U, tol, maxit):
    # A:       Matrix produced by diffusionMatrix
    # b:       RHS calculated by the source function
    # L and U: factors of the preconditioner from sp.spilu
    # tol:     tolerance for convergence
    # maxit:   maximum number of iterations
    # Returns tuple: x solution matrix, res residuals per iteration

    # Initialize parameters
    start_time = time.time()
    M = L @ U  # Preconditioner matrix
    x = np.zeros_like(b)
    r0 = b - (A @ x)
    z0 = sp.sparse.linalg.spsolve(M, r0)
    rz0 = r0.T @ z0
    d0 = z0.copy()
    i = 0
    res = []
    i = 0

    # PCG Loop
    while i <= maxit:
        Ad0 = A @ d0
        alpha = (rz0) / (d0.T @ Ad0)
        x += alpha * d0
        r0 -= alpha * Ad0
        norm = np.linalg.norm(r0, 2)
        res.append(norm)

        # Check for convergence
        if norm < tol:
            break

        z0 = sp.sparse.linalg.spsolve(M, r0)
        rz1 = r0.T @ z0
        beta = rz1 / rz0
        d0 = z0 + beta * d0
        rz0 = rz1
        i += 1
    end_time = time.time()
    print(
        f"myPCG::: Number of iterations = {i}, Norm = {norm}, Solution time = {end_time - start_time}"
    )

    return x, res
