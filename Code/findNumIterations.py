#
#  Computes the number of root finding iterations
#  Author: Frank Wefers (fwefers@fwefers.de)
#

import numpy as np
from rootfinding import *
from trajectory import *
np.set_printoptions(precision=20, suppress=True)

c = 343
t = np.linspace(-10, 10, 20*340)


if False:
    # Dual car pass-by
    v = 30  # Source velocity [m/s]
    d = 10  # Receiver displacement [m]
    rs = AnalyticTrajectory(lambda t: [v*t, 0])
    rr = AnalyticTrajectory(lambda t: [-v*t, -d])

if False:
    # Train car pass-by
    v = 70  # Source velocity [m/s]
    d = 30  # Receiver displacement [m]
    rs = AnalyticTrajectory(lambda t: [v * t, 0])
    rr = AnalyticTrajectory(lambda t: [0, -d])

if False:
    # Train car pass-by
    vs = 140  # Source velocity [m/s]
    v = vs
    h = 5000
    vr = 50  # Receiver velocity [m/s]
    d = 30  # Receiver displacement [m]
    phi = 60
    rs = AnalyticTrajectory(lambda t: [vs * (t+2), 0, h])
    rr = AnalyticTrajectory(lambda t: [np.cos(phi / 180 * np.pi)*vr*t, np.sin(phi / 180 * np.pi)*vr*t, 0])


if True:
    # Curve1
    rsd = SampledTrajectory.load("data/curve1.traj")
    rs = CatmullRomTrajectory(rsd)
    rr = AnalyticTrajectory(lambda t: [-20, 50, 0])

    startTime = np.min(rsd.t)
    endTime = np.max(rsd.t)
    duration = endTime - startTime
    rate = 44100/128
    t = np.linspace(startTime, endTime, duration * rate)

def f(rs, rr, t, tau, c):
    return np.linalg.norm(rr.p(t) - rs.p(t-tau)) - c*tau

print(rs.p(1))
print(rr.p(1))
#print(f(rs, rr, 1, 0.5, 343))

threshold = 1e-8
print("Error threshold = %g" % threshold)

# Newton
tau_newton = np.zeros_like(t)
last = 0
iterations_newton = []
for i in range(len(t)):
    print("Solving %d of %d" % (i, len(t)))
    tau_newton[i], n = findRootNewton(lambda tau: f(rs, rr, t[i], tau, c), last, maxError=threshold, returnNumIterations=True)
    last = tau_newton[i]
    iterations_newton.append(n)

data = np.vstack((t, tau_newton, iterations_newton)).T
np.savetxt("newton.txt", data, fmt='%12.6f', delimiter="\t", header="Time [s], tau [s], num iterations")

# Secant
tau_secant = np.zeros_like(t)
last = 1e-3
iterations_secant = []
for i in range(len(t)):
    print("Solving %d of %d" % (i, len(t)))
    tau_secant[i], n = findRootSecant(lambda tau: f(rs, rr, t[i], tau, c), 0, last, maxError=threshold, returnNumIterations=True)
    last = tau_secant[i]
    iterations_secant.append(n)

data = np.vstack((t, tau_secant, iterations_secant)).T
np.savetxt("secant.txt", data, fmt='%12.6f', delimiter="\t", header="Time [s], tau [s], num iterations")

fig = plt.figure()
ax = plt.subplot(211)

plt.plot(t, tau_newton)

ax = plt.subplot(212)
plt.hold(True)
plt.plot(t, iterations_newton, 'r', label="Newton")
plt.plot(t, iterations_secant, 'orange', label="Secant")
plt.hold(False)
plt.ylim(0, 5)
plt.legend()
plt.show()


print("--= Newton =--")
print("Min iterations: %d" % min(iterations_newton))
print("Avg iterations: %0.3f" % np.mean(iterations_newton))
print("Max iterations: %d" % max(iterations_newton))

print("--= Secant =--")
print("Min iterations: %d" % min(iterations_secant))
print("Avg iterations: %0.3f" % np.mean(iterations_secant))
print("Max iterations: %d" % max(iterations_secant))
