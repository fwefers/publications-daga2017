#
#  Various helper functions
#  Author: Frank Wefers (fwefers@fwefers.de)
#

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numbers
import numpy as np
import decimal

def isRealScalar(obj):
    """Checks for a real scalar or vector"""
    return isinstance(obj, numbers.Real) or (np.isscalar(obj) and np.isfinite(obj) and np.isreal(obj))

def isRealScalarOrVector(obj):
    """Checks for a real scalar or vector"""
    if obj is None: return False
    if isinstance(obj, numbers.Real):
        return True

    if isinstance(obj, list):
        if len(obj) == 0: return False
        return all(isinstance(x, numbers.Real) for x in obj)

    if isinstance(obj, np.float32):
        return True

    if isinstance(obj, np.ndarray):
        if obj.ndim > 2: return False
        if np.isscalar(obj):
            return np.isfinite(obj) and np.isreal(obj)
        else:
            return all(np.isfinite(obj)) and all(np.isreal(obj))

    return False

def isNumber(obj):
    """Returns if an object is of a number type"""
    # TODO: Refine and consider more types
    return isinstance(obj, (int, float, decimal.Decimal, np.number))

def asNumber(obj, dtype=np.float, errormsg="Expecting a finite number"):
    if isNumber(obj):
        result = dtype(obj)
        if np.isfinite(result): return result
    raise Exception("Expecting a finite number, but got " + str(obj))

def asRealNumber(obj, dtype=np.float):
    if isNumber(obj):
        result = dtype(obj)
        if np.isfinite(result) and np.isreal(result): return result
    raise Exception("Expecting a finite number, but got " + str(obj))

def asPositiveRealNumber(obj, dtype=np.float, errormsg="Expecting a positive real-valued number"):
    result = asNumber(obj, dtype, errormsg)
    if np.isreal(result) and result > 0: return result
    raise Exception(errormsg)

def asScalarOrVector(obj, dtype=np.float, errormsg="Expecting a scalar or vector"):
    if isNumber(obj):
        return asNumber(obj, dtype=dtype, errormsg=errormsg)
    else:
        return asVector(obj, dtype=dtype, errormsg=errormsg)

def asRealScalarOrVector(obj, dtype=np.float, errormsg="Expecting a real-valued scalar or vector"):
    result = asScalarOrVector(obj, dtype=dtype, errormsg=errormsg)
    if np.isscalar(obj):
        if np.isreal(obj): return result
    else:
        if all(np.isreal(obj)): return result
    raise Exception(errormsg)

def asVector(obj, dtype=np.float, errormsg="Expecting a vector"):
    """Converts a number, list of numbers or array into a 1-D NumPy ndarray"""
    if obj is None:
        raise Exception(errormsg + ", but got None")

    if isNumber(obj):
        raise Exception(errormsg + ", but got a number")

    # Python lists
    if isinstance(obj, list):
        if not obj:
            raise Exception(errormsg + ", but got an empty list")

        if not all(isNumber(item) for item in obj):
            raise Exception(errormsg + ", but got non numeric values")

        result = np.asarray(obj, dtype=dtype)
        if not all(np.isfinite(result)):
            raise Exception(errormsg + ", but got an infinite values")
        return result

    # Python tuples
    if isinstance(obj, tuple):
        if not obj:
            raise Exception(errormsg + ", but got an empty tuple")

        if not all(isNumber(item) for item in obj):
            raise Exception(errormsg + ", but got non numeric values")

        result = np.asarray(obj, dtype=dtype)
        if not all(np.isfinite(result)):
            raise Exception(errormsg + ", but got an infinite values")
        return result

    # NumPy ndarrays
    if isinstance(obj, np.ndarray):
        if not all(np.isfinite(obj)):
            raise Exception(errormsg + ", but got an infinite value")

        if obj.ndim == 0:
            raise Exception(errormsg + ", but got a zero-dimensional ndarray")
        if obj.ndim == 1:
            if obj.shape[0] == 0:
                raise Exception(errormsg + ", but got a empty ndarray")
            if obj.dtype == dtype:
                return obj
            else:
                return np.asarray(obj, dtype=dtype)

        elif obj.ndim == 2:
            if obj.shape[0] == 1:
                if obj.dtype == dtype:
                    return obj[0,:]
                else:
                    return np.asarray(obj[0,:], dtype=dtype)

            if obj.shape[1] == 1:
                if obj.dtype == dtype:
                    return obj[:,0]
                else:
                    return np.asarray(obj[:,0], dtype=dtype)

            return np.asarray(obj, dtype=dtype)

        raise Exception(errormsg + ", but got a multidimensional array")

    raise Exception(errormsg + ", but got some incompatible data")

def asRealVector(obj, dtype=np.float, errormsg="Expecting a positive real-valued vector"):
    result = asVector(obj, dtype, errormsg)
    if all(np.isreal(result)): return result
    raise Exception(errormsg)

def pcm2float(sig):
    """Convert a PCM signal to floating points with a range from -1 to 1"""
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype('float64')

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def plotTX(t, x, v=None, linestyle='o-'):
    [M, N] = x.shape
    plt.hold(True)
    plt.plot(t, x[:,0], linestyle, linewidth=2, markersize=4, color='#e3260d', markerfacecolor='#e3260d',
             markeredgecolor='#e3260d', label='X', zorder=3)
    if N >= 2:
        plt.plot(t, x[:,1], linestyle, linewidth=2, markersize=4, color='#3fc20b', markerfacecolor='#3fc20b',
                 markeredgecolor='#3fc20b', label='Y', zorder=2)
    if N == 3:
        plt.plot(t, x[:,2], linestyle, linewidth=2, markersize=4, color='#0c5ac4', markerfacecolor='#0c5ac4',
                 markeredgecolor='#0c5ac4', label='Z', zorder=1)
    if v is not None:
        plt.plot(t, v, linestyle, linewidth=2, markersize=4, color='black', label='P', zorder=0)
    plt.hold(False)

    a = np.min(t)
    b = np.max(t)
    s = b - a
    plt.xlim(a - s / 15, b + s / 15)

    a = np.min(x)
    b = np.max(x)
    if v is not None:
        a = np.min([a, np.min(v)])
        b = np.max([b, np.max(v)])
    s = b - a
    plt.ylim(a - s / 15, b + s / 15)

def plotTrajectory(c, lines=True, markers=True, indices=False):
    fig = plt.figure()
    linestyle = ''
    if markers: linestyle += 'o'
    if lines: linestyle += '-'

    if c.dims == 3:
        # 3-D trajectory
        ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2, projection='3d', title='Curve')
        ax1.plot(c.p[:, 0], c.p[:, 1], c.p[:, 2], linestyle)
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('y [m]')
        ax1.set_zlabel('z [m]')

        # Enforce equal axes
        xmin = np.min(c.p[:, 0])
        xmax = np.max(c.p[:, 0])
        ymin = np.min(c.p[:, 1])
        ymax = np.max(c.p[:, 1])
        zmin = np.min(c.p[:, 2])
        zmax = np.max(c.p[:, 2])
        m = np.max((xmax-xmin, ymax-ymin, zmax-zmin))

        ax1.set_xlim3d([(xmax + xmin) / 2 - m / 2, (xmax + xmin) / 2 + m / 2])
        ax1.set_ylim3d([(ymax + ymin) / 2 - m / 2, (ymax + ymin) / 2 + m / 2])
        ax1.set_zlim3d([(zmax + zmin) / 2 - m / 2, (zmax + zmin) / 2 + m / 2])

        if indices:
            for i in range(c.nsamples):
                ax1.text(c.p[i, 0], c.p[i, 1], c.p[i, 2], '%d' % i, size=12, zorder=1, color='#C0C0C0')
    else:
        # 2-D trajectory
        ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2, title='Curve')
        if c.dims == 2:
            ax1.plot(c.p[:, 0], c.p[:, 1], linestyle)
            ax1.set_ylabel('y [m]')

            if indices:
                for i in range(c.nsamples):
                    ax1.annotate('%d' % i, xy=(c.p[i, 0], c.p[i, 1]), xytext=(0, 5), color='#C0C0C0',
                                 textcoords='offset points')
        if c.dims == 1:
            ax1.plot(c.p, np.zeros_like(c.p), linestyle)

            if indices:
                for i in range(c.nsamples):
                    ax1.annotate('%d' % i, xy=(c.p[i], 0), xytext=(0, 5), color='#C0C0C0',
                                 textcoords='offset points')
        ax1.set_xlabel('x [m]')
        ax1.axis('equal')



    # Positions
    # ax2 = fig.add_subplot(222, title='Position')
    ax2 = plt.subplot2grid((3, 3), (0, 2), title='Position')
    plotTX(c.t, c.p, linestyle=linestyle)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.grid(True)
    # plt.legend()

    # Velocities
    # ax3 = fig.add_subplot(223, title='Velocity')
    ax3 = plt.subplot2grid((3, 3), (1, 2), title='Velocity')
    plotTX(c.t, c.v, c.vp, linestyle=linestyle)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [ms^-1]')
    plt.grid(True)
    # plt.legend()

    # Accelerations
    # ax4 = fig.add_subplot(224, title='Acceleration')
    ax4 = plt.subplot2grid((3, 3), (2, 2), title='Acceleration')
    plotTX(c.t, c.a, c.ap, linestyle=linestyle)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [ms^-2]')
    plt.grid(True)
    # plt.legend()

