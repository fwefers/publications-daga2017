#
#  Virtual audio renderer with varying propagation delay
#  (used to evaluate the numerical computation of the propagation delay)
#  Author: Frank Wefers (fwefers@fwefers.de)
#

from rootfinding import findRootNewton
import numpy as np

class Renderer:
    """A simple audio renderer"""

    def __init__(self, rs, rr, ss, c, t0, fs, B):
        self.rs = rs
        self.rr = rr
        self.ss = ss
        self.c = c
        self.t0 = t0
        self.fs = fs
        self.B = B
        self.sampleOffset = 0

        # Initial propagation time
        self.tau = self.solveTau(t0, np.linalg.norm(self.rr.p(t0) - self.rs.p(t0))/c)
        print("Initial tau = %0.3f ms, delay = %0.2f samples" % (self.tau*1e3, self.tau*self.fs))

    def solveTau(self, t, tauStart):
        return findRootNewton(lambda tau: np.linalg.norm(self.rr.p(t) - self.rs.p(t - tau)) - self.c * tau, tauStart, maxError = 1/self.fs/10, epsilon=1e-3)

    def process(self):
        # Current time
        t = self.sampleOffset / float(self.fs) + self.t0
        print("Sample = %d, time = %0.3f ms" % (self.sampleOffset, t))

        # Solve the propagation time tau(t) starting from the last solution
        self.tau = self.solveTau(t, self.tau)
        delay = self.tau * self.fs

        print("Tau = %0.3f ms, delay = %0.2f samples" % (self.tau * 1e3, delay))

        self.sampleOffset += self.B

        return [t, self.tau]
