def dekker(fnon, xbounds, tol, maxk, *fnonargs):
    # fnon     - name of the nonlinear function f(x)
    # xbounds  - initial bracket [a, b]
    # tol      - stopping tolerance
    # maxk     - maximum number of iterations before stopping
    # fnonargs - optional arguments that will be passed to the nonlinear function (useful for additional function parameters)

    k = 0
    a, b = xbounds
    print(" k  xk          f(xk)")
    b0 = a
    fb = eval(fnon)(b, *fnonargs)
    iterates = []

    # Main loop
    while abs(fb) > tol and k < maxk:
        fa = eval(fnon)(a, *fnonargs)
        fb0 = eval(fnon)(b0, *fnonargs)
        iterates.append(b)

        # Bisection
        m = (a + b) / 2

        # Implement secant method
        if fb == fb0:
            s = m
        else:
            s = b - fb * (b - b0) / (fb - fb0)

        # Check if secant result is between the midpoint and previous
        # right bracket. Also check the case where the convergence is
        # moving in the opposite direction.

        if (m < s < b) or (m > s > b):
            b1 = s
        else:
            b1 = m

        fb1 = eval(fnon)(b1, *fnonargs)

        # Check whether f(a_k) and f(b_k+1) have the same sign.
        if fa * fb1 > 0:
            a = b

        # Check which extreme of the new bracket is the best guess.
        if abs(eval(fnon)(a, *fnonargs)) < abs(fb1):
            a, b1 = b1, a

        # Update the right hand side of the bracket.
        b0, b = b, b1
        fb = eval(fnon)(b, *fnonargs)

        k += 1
        print("{0:2.0f}  {1:2.8f}  {2:2.2e}".format(k, b, abs(fb)))

    if k == maxk:
        print("Not converged")
    else:
        print("Converged")

    return iterates
