def myPCGSolver(mux, muy, h, tol, maxit, drop_tol):
    # mux:       thermal diffusion coefficient in x direction
    # muy:       thermal diffusion coefficient in y direction
    # h:         grid spacing
    # tol:       tolerance for convergence
    # maxit:     maximum number of iterations
    # drop_tol:  drop tolerance for calculating ILU factors
    # Returns tuple: sol solution matrix, res residuals per iteration

    n = int(1.0 / h + 1)
    N = n**2
    intNodes, extNodes = boundaryConditions(n)
    res = []

    # Generate diffusion matrix
    A = diffusionMatrix(mux, muy, h)
    A_int = A[intNodes][:, intNodes]

    # Create ILU preconditioner using scipy
    M = sp.sparse.linalg.spilu(A_int, drop_tol=drop_tol, permc_spec="natural")
    L = M.L

    # Define source term and initialize vectors
    b = np.zeros(N, dtype=np.float64)
    for k in range(0, N):
        i, j = reverseIndexFD(k, n)
        b[k] = source_function(i * h, j * h)
    b_int = b[intNodes]

    sol = np.zeros_like(b)
    # A is a symmetric positive definite matrix and the efficiency is gained using L@L.T as the preconditioner.
    sol[intNodes], res = myPCG(A_int, b_int, L, L.T, tol, maxit)

    return sol, res
