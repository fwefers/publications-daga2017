#
#  Plots tau(t) and the number of requires root finding iterations
#  Author: Frank Wefers (fwefers@fwefers.de)
#

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=20, suppress=True)

t, tau_newton, iterations_newton = np.loadtxt("newton.txt", unpack=True)
t, tau_secant, iterations_secant = np.loadtxt("secant.txt", unpack=True)

fig = plt.figure()
ax = plt.subplot(211)
plt.title(r"Propagation delay $\tau$(t)")
plt.plot(t, tau_newton, 'k')
plt.xlabel(r"t [s]")
plt.ylabel(r"$\tau$(t) [s]")
plt.xlim(np.min(t), np.max(t))

ax = plt.subplot(212)
plt.hold(True)
plt.plot(t, iterations_newton, 'r', label="Newton method")
plt.title(r"Required iterations ($|e|<10^{-8}$)")
plt.plot(t, iterations_secant, 'b', label="Secant method")
plt.xlabel(r"t [s]")
plt.ylabel(r"Iterations")
plt.hold(False)
plt.xlim(np.min(t), np.max(t))
plt.ylim(0, 5)
plt.legend()
fig.subplots_adjust(hspace=.6)
plt.show()

print("--= Newton =--")
print("Min iterations: %d" % min(iterations_newton))
print("Avg iterations: %0.3f" % np.mean(iterations_newton))
print("Max iterations: %d" % max(iterations_newton))

print("--= Secant =--")
print("Min iterations: %d" % min(iterations_secant))
print("Avg iterations: %0.3f" % np.mean(iterations_secant))
print("Max iterations: %d" % max(iterations_secant))
