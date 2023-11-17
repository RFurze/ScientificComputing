def momgraddesc(fnon, jac, x0, gamma, tol, maxk, *fnonargs):
    # fnon     - name of the nonlinear function f(x)
    # jac      - name of the Jacobian function J(x)
    # x0       - initial guess for the solution x0
    # tol      - stopping tolerance for Newton's iteration
    # maxk     - maximum number of Newton iterations before stopping
    # fnonargs - optional arguments that will be passed to the nonlinear function (useful for additional function parameters)
    k = 0
    x = x0
    lmb = 0.01
    iterates = np.array(x)
    delta0 = 0
    delta = 0

    F = eval(fnon)(x, *fnonargs)

    print(" k    f(xk)")

    # Main Newton loop
    while np.linalg.norm(F) > tol and k < maxk:
        # Evaluate Jacobian matrix
        J = eval(jac)(x, *fnonargs)

        # Calculate the direction - the previous value of delta has been incorporated with the momentum factor gamma
        delta = delta * gamma + 2 * np.transpose(J) @ F

        x = x - lmb * delta

        F = eval(fnon)(x, *fnonargs)
        # the current value of delta is stored over he previous value for the next iteration.
        # delta0 = delta
        iterates = np.hstack([iterates, x])
        k += 1
        print("{0:2.0f}  {1:2.2e}".format(k, np.linalg.norm(F)))

    if k == maxk:
        print("Not converged")
    else:
        print("Converged to ")
        print(x)
    return iterates
