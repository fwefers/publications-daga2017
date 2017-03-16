#
#  Spline classes and functions
#  Author: Frank Wefers (fwefers@fwefers.de)
#

import numpy as np
from numpy.polynomial.polynomial import Polynomial

def finiteDifferenceNonuniform(x, y, dtype=np.float):
    """Computes non-uniform finite differences (second-order accurate)"""
    # Source: http://cfd.mace.manchester.ac.uk/twiki/pub/Main/TimCraftNotes_All_Access/cfd1-findiffs.pdf

    x = np.asarray(x)
    y = np.asarray(y)
    m = np.zeros_like(x, dtype=np.float)

    m[0] = (y[1]-y[0]) / (x[1]-x[0])
    m[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    for i in range(1, len(m)-1):
        a = x[i] - x[i-1]
        b = x[i+1] - x[i]
        m[i] = -b/(a*(a+b))*y[i-1] + (b-a)/(b*a)*y[i] + a/(b*(a+b))*y[i+1]
    return m

def cubicHermiteCoeffs(y0, y1, m0, m1):
    """Computes the coefficients of a cubic Hermite polynomial p(x)
       over the unit interval x in [0,1] with boundary values p(0)=y0, p(1)=y1
       and boundary derivatives [dp/dx](0)=m0, [dp/dx](1)=m1"""
    return np.dot( np.array([[ 1,  0,  0,  0],
                             [ 0,  0,  1,  0],
                             [-3,  3, -2, -1],
                             [ 2, -2,  1,  1]]),
                   np.array([y0, y1, m0, m1]) )

class CubicHermitePolynomial:
    def __init__(self, y0, y1, m0, m1):
        c = np.dot( np.array([[ 1,  0,  0,  0],
                             [ 0,  0,  1,  0],
                             [-3,  3, -2, -1],
                             [ 2, -2,  1,  1]]),
                    np.array([y0, y1, m0, m1]) )
        self.__P = Polynomial(c)

    def __call__(self, x):
        return self.__P(x)

class CubicHermiteSpline:
    def __init__(self, x, y, m):
        assert all(np.diff(x) > 0), "x must be a strictly increasing sequence"
        self.__x = np.asarray(x)
        print("x = ", self.__x)
        self.__dx = self.__x[1:] - self.__x[0:-1]
        self.__P = []
        for i in range(len(self.__dx)):
            print(self.__dx[i])
            self.__P.append( CubicHermitePolynomial(y[i], y[i+1], m[i]*self.__dx[i], m[i+1]*self.__dx[i]) )

    def __call__(self, xi):
        if not isinstance(xi, np.ndarray): xi = np.asarray(xi, dtype=np.float)
        yi = np.zeros_like(xi, dtype=np.float)

        # Continue left boundary value
        yi[xi < self.__x[0]] = self.__P[0](0)

        # Compute the values in the intervals
        for i in range(len(self.__dx)):
            # Mask of the interval
            mask = (xi >= self.__x[i]) & (xi < self.__x[i+1])
            # Compute the values (inside unit interval)
            yi[mask] = self.__P[i]((xi[mask]-self.__x[i])/self.__dx[i])

        yi[xi >= self.__x[-1]] = self.__P[-1](1)
        return yi

class CatmullRomSpline:
    def __init__(self, x, y):
        # Compute the derivatives using finite differences
        m = finiteDifferenceNonuniform(x, y)
        self.__s = CubicHermiteSpline(x, y, m)

    def __call__(self, xi):
        return self.__s(xi)
