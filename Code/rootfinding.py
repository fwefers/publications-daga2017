#
#  Root finding algorithms
#  Author: Frank Wefers (fwefers@fwefers.de)
#

def findRootNewton(f, x0, maxError=1e-8, epsilon=0.1, verbose=False, returnNumIterations=False):
    x = x0
    n = 0
    while True:
        y = f(x)

        if verbose: print("x[%d] = %0.12f, |f(x[%d])| = %0.12f" % (n, x, n, abs(y)))

        if abs(y) <= maxError:
            if verbose: print("%d iterations\n" % n)
            if returnNumIterations:
                return x, n
            else:
                return x

        yl = f(x-epsilon)
        yr = f(x + epsilon)
        m = (yr-yl)/(2*epsilon)

        x = x - y/m
        n += 1

def findRootSecant(f, x0, x1, maxError=1e-8, verbose=False, returnNumIterations=False):
    y0 = f(x0)
    n = 0
    while True:
        y1 = f(x1)

        if verbose: print("x[%d] = %0.12f, |f(x[%d])| = %0.12f" % (n, x1, n, abs(y1)))

        if abs(y1) <= maxError:
            if verbose: print("%d iterations\n" % n)
            if returnNumIterations:
                return x1, n
            else:
                return x1

        x2 = x1 - y1*(x1-x0)/(y1-y0)
        x0 = x1
        y0 = y1
        x1 = x2

        n += 1
