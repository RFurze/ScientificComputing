def myGradientDescentMom(fnon, jac, x0, gamma, tol, maxk, *fnonargs):
    # fnon     - name of the nonlinear function f(x)
    # jac      - name of the Jacobian function J(x)
    # gamma    - momentum factor
    # x0       - initial guess for the solution x0
    # tol      - stopping tolerance
    # maxk     - maximum number of Newton iterations before stopping
    # fnonargs - optional arguments
    
    k = 0
    x = x0
    lmb = 0.01

    # Note: handling of xArray is updated out of personal preference
    # - xArray.append is replaced with hstack further down
    xArray = np.array(x)
    delta = 0

    F = eval(fnon)(x, *fnonargs)

    print(" k    f(xk)")

    # Main gradient descent loop
    # Update the stopping criteria to stop when k == maxk
    # (i.e account for zero based indexing)
    while np.linalg.norm(F, 2) > tol and k < maxk:
        # Evaluate Jacobian matrix
        J = eval(jac)(x, *fnonargs)

        # Take gradient descent step
        # UPDATE: calculate a new gradient descent step using
        # the momentum modification
        delta = delta * gamma + 2 * np.dot(np.transpose(J), F)

        x = x - lmb * delta

        F = eval(fnon)(x, *fnonargs)
        xArray = np.hstack([xArray, x])
        k += 1
        print("{0:2.0f}  {1:2.2e}".format(k, np.linalg.norm(F)))

    if k == maxk:
        print("Not converged")
    else:
        print("Converged to ")
        print(x)
    return xArray
