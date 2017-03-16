#
#  Visualizes the source and listener trajectory
#  Author: Frank Wefers (fwefers@fwefers.de)
#

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from trajectory import *

mpl.rcParams['lines.linewidth'] = 1.5


if True:
    # Curve1
    rsd = SampledTrajectory.load("data/curve1.traj")
    timeRange = [rsd.t[0], rsd.t[-1]]
    rs = CatmullRomTrajectory(rsd)
    rr = AnalyticTrajectory(lambda t: [-20, 50, 0])

def animateSourceAndReceiverTrajectory(rs, rr, timeRange, speed=1, frameRate=25):

    startTime = np.min(timeRange)
    endTime = np.max(timeRange)
    duration = endTime - startTime

    dt = speed/frameRate
    nframes = int(np.ceil(duration / dt))
    t = np.arange(nframes) * dt

    print("%d frames" % nframes)

    ps = np.zeros((nframes, rs.dims))
    pr = np.zeros((nframes, rs.dims))

    for i in range(nframes):
        ps[i, :] = rs.p(t[i])
        pr[i, :] = rr.p(t[i])

    # --= Animation =--

    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')

    plt.hold(True)
    ax.plot(ps[:, 0], ps[:, 1], ps[:, 2], '-', color="#2465c1")
    ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], '-', color="#2465c1")
    srcPoint, = ax.plot([ps[0, 0]], [ps[0, 1]], [ps[0, 2]], '.', color='#d5ff86', markersize=24, markeredgecolor='black', markeredgewidth=1.25)
    rcvPoint, = ax.plot([pr[0, 0]], [pr[0, 1]], [pr[0, 2]], '.', color='#f19d88', markersize=24, markeredgecolor='black', markeredgewidth=1.25)
    plt.hold(False)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')

    # Enforce equal axes
    xmin = np.min( (np.min(ps[:, 0]), np.min(pr[:, 0])) )
    xmax = np.max( (np.max(ps[:, 0]), np.max(pr[:, 0])) )
    ymin = np.min( (np.min(ps[:, 1]), np.min(pr[:, 1])) )
    ymax = np.max( (np.max(ps[:, 1]), np.max(pr[:, 1])) )
    zmin = np.min( (np.min(ps[:, 2]), np.min(pr[:, 2])) )
    zmax = np.max( (np.max(ps[:, 2]), np.max(pr[:, 2])) )
    m = np.max((xmax - xmin, ymax - ymin, zmax - zmin))

    ax.set_xlim3d([(xmax + xmin) / 2 - m / 2, (xmax + xmin) / 2 + m / 2])
    ax.set_ylim3d([(ymax + ymin) / 2 - m / 2, (ymax + ymin) / 2 + m / 2])
    #ax.set_zlim3d([(zmax + zmin) / 2 - m / 2, (zmax + zmin) / 2 + m / 2])
    ax.set_zlim3d([zmin, zmin + m])

    def update(i):
        #print("Update %d" % i)
        srcPoint.set_data([ps[i, 0], ps[i, 1]])
        srcPoint.set_3d_properties(ps[i, 2])
        #srcLabel.set_x(ps[i, 0])
        #srcLabel.set_y(ps[i, 1])
        #srcLabel.set_3d_properties(ps[i, 2])

        rcvPoint.set_data([pr[i, 0], pr[i, 1]])
        rcvPoint.set_3d_properties(pr[i, 2])

        plt.savefig('images/img_%04d.png' % i, bbox_inches='tight')

        return srcPoint, rcvPoint,

    ani = animation.FuncAnimation(fig, update, np.arange(len(t)), interval=frameRate, blit=False, repeat=False)

    plt.show()

    return 0


animateSourceAndReceiverTrajectory(rs, rr, timeRange, speed=6)


#plt.plot(iterations)
#plt.ylim(0, np.max(iterations)+1)
#plt.axhline(np.mean(iterations), color='r', zorder=-1)
#plt.show()
