#
#  Visualize the root finding function f(tau) = |rR(t)-rS(t-tau)|-ct
#  for a simple passby with straight-line motion and resting receiver
#
#  Author: Frank Wefers (fwefers@fwefers.de)
#

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from rootfinding import findRootNewton, findRootSecant
from trajectory import AnalyticTrajectory

mpl.rcParams['lines.linewidth'] = 1.5

c = 343     # Speed of sound [m/s]

if True:
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

#rr = AnalyticTrajectory(lambda t: [0, np.cos(t/2)*-d])

#rs = AnalyticTrajectory(lambda t: [v*t, 20*np.sin(t)])
#rr = AnalyticTrajectory(lambda t: [-v*t, -20+30*np.cos(t*1.4)*np.exp(-0.1*t)])

def f(rs, rr, t, tau, c):
    return np.linalg.norm(rr.p(t) - rs.p(t-tau)) - c*tau

# Times modeled like a sine for a nice animation
#t = -3*np.cos(np.linspace(0, 2*np.pi, 300))
t = np.linspace(-5, 5, 344)
tau = np.linspace(0, 1, 100)
y = []
tau0 = []
ytau0 = []
iterations = []
last = 1

for time in t:
    yv = np.zeros_like(tau)
    for i in range(len(tau)):
        yv[i] = f(rs, rr, time, tau[i], c)
    y.append(yv)
    #tau0v, n = findRootNewton(lambda tau: f(rs, rr, time, tau, c), 0, maxError=1e-8, epsilon=1e-6)

    # Newton mit letzter Position
    #tau0v, n = findRootNewton(lambda tau: f(rs, rr, time, tau, c), last, maxError=1e-8, epsilon=1e-6)

    # Sekantenverfahren mit letzter Position
    tau0v, n = findRootSecant(lambda tau: f(rs, rr, time, tau, c), last, 0, maxError=1e-8, returnNumIterations=True)

    tau0.append(tau0v)
    last = tau0v
    iterations.append(n)
    ytau0.append(f(rs, rr, time, tau0[-1], c))

#print("f(0) = ", f(rs, rr, t, 0, c))
#print("tau0 = ", tau0)

def plot(i):
    plt.hold(True)
    plt.plot(tau, y[i], linewidth=1.5, color='#0c5ac4')
    plt.plot(tau0[i], ytau0[i], '.', color='#e3260d', markersize=16)
    plt.plot(tau, -c * tau, '--', linewidth=1.5, color='gray')
    plt.hold(False)
    plt.grid(True)
    plt.axhline(y=0, color='k', zorder=-1)
    plt.axvline(x=0, color='k', zorder=-1)
    plt.show()

def animate():

    fig = plt.figure(figsize=(12,8))

    wavefront = plt.Circle((0, 0), 10, color='black', fill=False, linestyle=':')

    ax = plt.subplot(211)
    trange = np.linspace(-5, 5, 100)

    plt.hold(True)
    ps = rs.p(trange)
    plt.plot(ps[:,0], ps[:,1], color='#0c5ac4')
    pr = rr.p(trange)
    plt.plot(pr[:, 0], pr[:, 1], color='#0c5ac4')
    srcPoint, = ax.plot(ps[0, 0], ps[0, 1], '.', color='#d5ff86', markersize=24, markeredgecolor='black', markeredgewidth=1.25)
    rcvPoint, = ax.plot(pr[0, 0], pr[0, 1], '.', color='#f19d88', markersize=24, markeredgecolor='black', markeredgewidth=1.25)
    orgPoint, = ax.plot(ps[0, 0], ps[0, 1], '.', color='black', markersize=16)
    ax.add_artist(wavefront)

    plt.title("Trajectories")

    plt.hold(False)
    plt.ylim(-d*0.7, d*0.7)
    plt.xlim(-np.max(t)*v*1.2, np.max(t)*v*1.2)
    plt.axis('equal')
    plt.xlabel(r"x [m]")
    plt.xlabel(r"y [m]")

    ax = plt.subplot(212)
    plt.hold(True)
    curve, = ax.plot(tau, y[0], color='#539b00')
    zero, = ax.plot(tau0[i], ytau0[i], '.', color='#e3260d', markersize=16)
    point, = ax.plot(tau[0], y[i][0], '.k', markersize=12)
    ax.plot(tau, -c * tau, '--', color='gray')
    plt.title(r"Function f($\tau$)")
    plt.hold(False)
    plt.grid(True)
    plt.axhline(y=0, color='k', zorder=-1)
    plt.axvline(x=0, color='k', zorder=-1)
    plt.ylim(0, 0.6)
    plt.ylim(-300, 300)
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel(r"f($\tau$) [m]")
    fig.subplots_adjust(hspace=.7)
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()

    def update(i):
        ps = rs.p(t[i])
        srcPoint.set_xdata(ps[0])
        srcPoint.set_ydata(ps[1])

        pr = rr.p(t[i])
        rcvPoint.set_xdata(pr[0])
        rcvPoint.set_ydata(pr[1])

        psr = rs.p(t[i]-tau0[i])
        orgPoint.set_xdata(psr[0])
        orgPoint.set_ydata(psr[1])
        wavefront.center = psr
        wavefront.set_radius(c*tau0[i])

        curve.set_ydata(y[i])
        zero.set_xdata(tau0[i])
        zero.set_ydata(ytau0[i])
        point.set_ydata(y[i][0])

        plt.savefig('images/img_%04d.png' % i, bbox_inches='tight')

        return orgPoint, wavefront, srcPoint, rcvPoint, curve, zero, point

    ani = animation.FuncAnimation(fig, update, np.arange(len(t)), interval=25, blit=False, repeat=False)
    #writer = animation.writers['ffmpeg']
    #ani.save('demo.mp4', writer=writer, dpi=100)

    plt.show()

    return 0

animate()

#plt.plot(iterations)
#plt.ylim(0, np.max(iterations)+1)
#plt.axhline(np.mean(iterations), color='r', zorder=-1)
#plt.show()
