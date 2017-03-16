#
#  Trajectory classes
#  Author: Frank Wefers (fwefers@fwefers.de)
#

from helpers import *
from splines import CatmullRomSpline

# TODO:
# 1-D import does not work (Unterscheidung ndarray, Scalar|Vector)

class SampledTrajectory:
    """A trajectory consisting of discrete positions sampled over time"""

    def __init__(self, times=None, positions=None):
        """Creates an empty trajectory"""
        self.__t = None     # sample times
        self.__p = None     # sampled position (vectors)

        if times is not None:
            assert isRealScalarOrVector(times), "Times must be None or a real-valued scalar or vector"

            # Convert the times into a ndarray
            if isRealScalar(times):
                self.__t = np.array([times], dtype=np.float32)
            else:
                self.__t = np.asarray(times, dtype=np.float32)
            assert np.all(np.diff(self.__t) > 0), "Times must be increasing"

            assert positions is not None, "Positions expected"
            positions = np.asarray(positions)

            if positions.ndim == 1:
                positions = np.reshape(positions, (1, positions.shape[0]))
            [M, N] = positions.shape

            assert M == len(self.__t), "Position rows must match length of times"
            assert N >= 1 and N <= 3, "Positions can be 1-D, 2-D or 3-D"
            if isRealScalar(positions):
                self.__p = np.array([positions], dtype=np.float32)
            else:
                self.__p = np.asarray(positions, dtype=np.float32)
        else:
            assert positions is None, "Times must be specified along positions"

    def load(path):
        """Loads a sampled trajectory from a text file"""
        data = np.loadtxt(path, unpack=True)
        t = data[0,:]
        p = data[1:4,:].T
        return SampledTrajectory(t, p)

    def store(self, path):
        """Stored the trajectory into a text file"""
        n = self.__p.shape[0]
        data = np.hstack((self.__t.reshape(n, 1), self.__p))
        np.savetxt(path, data, fmt='%12.6f', delimiter="\t", header="Time [s], X [m], Y [m], Z [m]")

    @property
    def nsamples(self):
        """Returns the number of samples"""
        return len(self.__t)

    @property
    def dims(self):
        """Returns the number of spatial dimensions (e.g. 1-D, 2-D, 3-D)"""
        if self.__p is None: return 0
        return self.__p.shape[1]

    @property
    def t(self):
        """Returns the sample times as a NumPy array"""
        return self.__t;

    @property
    def dt(self):
        """Returns the time's first-order derivative as a NumPy array"""
        return np.gradient(self.__t);

    @property
    def p(self):
        """Returns the sample positions as a NumPy matrix"""
        return self.__p;

    @property
    def L(self):
        """Returns the segment lengths as a NumPy array"""
        return np.linalg.norm(self.dp, axis=1)

    @property
    def v(self):
        """Returns the velocity vectors as a NumPy array"""
        return np.apply_along_axis(lambda x: np.gradient(x, self.dt), 0, self.__p)

    @property
    def vp(self):
        """Returns the velocities in forward direction of the path as a NumPy array"""
        return np.linalg.norm(self.v, axis=1)

    @property
    def a(self):
        """Returns the acceleration vectors as a NumPy array"""
        return np.apply_along_axis(lambda x: np.gradient(x, self.dt), 0, self.v)

    @property
    def ap(self):
        """Returns the acceleration in forward direction of the path as a NumPy array"""
        return np.linalg.norm(self.a, axis=1)

    @property
    def center(self):
        """Returns the mean position as a NumPy vector"""
        return np.mean(self.__p, axis=0)

    def add(self, time, position):
        """Appends another position sample to the trajectory"""

        assert isRealScalar(time), "Time must be a real-valued number"
        assert isRealScalarOrVector(position), "Position must be a real-valued scalar or vector"

        if self.__t is None:
            # Case: Empty trajectory
            self.__t = np.array([time], dtype=np.float32)
            self.__p = np.asarray([position], dtype=np.float32)

        else:
            # Case: Append
            assert (isRealScalar(position) and self.dims == 1) or (len(position) == self.dims), "Position must be a %d-element vector" % self.dims
            assert time > self.__t[-1], "Times must be strictly increasing"
            self.__t = np.append(self.__t, time)
            self.__p = np.vstack((self.__p, position))

    def timeshift(self, dt):
        """Shift the time by adding the given offset"""
        dt = asRealNumber(dt)
        self.__t += dt

    def move(self, v):
        """Moves the trajectory by adding the vector v to each point"""
        v = asRealScalarOrVector(v)
        assert v.shape[0] == self.dims, "Vector has wrong dimension"
        self.__p += v

    def trim(self, startTime, endTime):
        """Extracts a specific time range as an independent trajectory"""
        assert isRealScalar(startTime), "Start time must be a real-valued number"
        assert isRealScalar(endTime), "End time must be a real-valued number"
        a = np.min(np.where(self.__t >= startTime))
        b = np.max(np.where(self.__t <= endTime))
        if b-a <= 0: raise ('Invalid time range')
        return SampledTrajectory(self.__t[a:b], self.__p[a:b,:])

    def from_position_function(t, p_function):
        """Creates a trajectory by sampling a function time->position in the given time points"""
        c = SampledTrajectory()
        t = np.asarray(t, dtype=np.float32)
        for i in range(len(t)):
            c.add(t[i], np.asarray(p_function(t[i]), dtype=np.float32))
        return c

    def from_velocity_function(t, p0, v_function):
        """Creates a trajectory by integrating a function time->velocity in the given time points"""
        c = SampledTrajectory()
        t = np.asarray(t, dtype=np.float32)
        dt = np.gradient(t)
        p = np.asarray(p0, dtype=np.float32)
        for i in range(len(t)):
            c.add(t[i], p)
            v = np.asarray(v_function(t[i]), dtype=np.float32)
            p += v * dt[i]
        return c

    def from_acceleration_function(t, p0, v0, a_function):
        """Creates a trajectory by double-integrating a function time->acceleration in the given time points"""
        c = SampledTrajectory()
        t = np.asarray(t, dtype=np.float32)
        dt = np.gradient(t)
        p = np.asarray(p0, dtype=np.float32)
        v = np.asarray(v0, dtype=np.float32)
        for i in range(len(t)):
            c.add(t[i], p)
            a = np.asarray(a_function(t[i]), dtype=np.float32)
            v += a * dt[i]
            p += v * dt[i]
        return c

    def __str__(self):
        s = "# Trajectory, %d samples\n" % self.nsamples
        for i in range(self.nsamples):
            s += "%0.3f -> %s\n" % (self.__t[i], str(self.__p[i,:]))
        return s


class AnalyticTrajectory:
    """A continuous trajectory defined by a function time->space"""

    def __init__(self, function):
        self.__f = function
        # Determine the spatial dimensions
        p0 = asScalarOrVector(self.__f(np.float(0)))
        if isinstance(p0, np.float):
            self.__dims = 1
        else:
            self.__dims = p0.shape[0]

    @property
    def dims(self):
        """Returns the number of spatial dimensions (e.g. 1-D, 2-D, 3-D)"""
        return self.__dims

    def p(self, t):
        """Computes the position(s) for the given time(s) and returns it as a NumPy array"""
        t = asScalarOrVector(t)
        if isinstance(t, np.float):
            # Case: Number
            x = asScalarOrVector(self.__f(t))
            if isinstance(x, np.float):
                assert self.__dims == 1, "Function result has invalid dimension"
            else:
                assert x.shape[0] == self.__dims, "Function result has invalid dimension"
        else:
            # Case: Vector
            x = np.empty((len(t), self.__dims))
            for i in range(len(t)):
                y = asScalarOrVector(self.__f(t[i]))
                if isinstance(y, np.float):
                    assert self.__dims == 1, "Function result has invalid dimension"
                else:
                    assert y.shape[0] == self.__dims, "Function result has invalid dimension"
                x[i,:] = y

        return x

    def __eval(self, t):
        # Evaluate the function at a single time t
        # and cast the results to NumPy types
        return asScalarOrVector(self.__f(np.float(t)))

class LinearTrajectory:
    """A continuous trajectory interpolated in space over time"""

    def __init__(self, data):
        """Creates an empty trajectory"""
        assert isinstance(data, SampledTrajectory), "Data must be sampled trajectory"
        assert np.all(np.diff(data.t) > 0), "Data times must be increasing"
        self.__dims = data.dims
        self.__t = np.copy(data.t)
        self.__p = np.copy(data.p)

    @property
    def dims(self):
        """Returns the number of spatial dimensions (e.g. 1-D, 2-D, 3-D)"""
        return self.__dims

    def p(self, t):
        """Computes the position(s) for the given time(s) and returns it as a NumPy array"""
        t = asRealScalarOrVector(t)

        # Convert the evaluation points into a ndarray
        if isinstance(t, np.float):
            t = np.array([t], dtype=np.float)
        else:
            t = np.asarray(t, dtype=np.float)

        N = len(t)
        c = []
        for i in range(self.__dims):
            u = np.interp(t, self.__t, self.__p[:,i], left=self.__p[0,i], right=self.__p[-1,i]);
            c.append(np.reshape(u, (N,1)))

        return np.hstack(c)

    def v(self):
        """Computes the velocity for the given time and returns it as a NumPy array"""
        return np.apply_along_axis(lambda x: np.gradient(x, self.dt), 0, self.__p)

    def vp(self):
        """Computes the velocity forward direction of the path for the given time"""
        return 0

    def a(self):
        """Returns the acceleration vectors as a NumPy array"""
        return np.apply_along_axis(lambda x: np.gradient(x, self.dt), 0, self.v)

    def ap(self):
        """Computes the acceleration in forward direction of the path for the given time"""
        return np.linalg.norm(self.a, axis=1)

class CatmullRomTrajectory:
    """A continuous trajectory interpolated in space over time"""

    def __init__(self, data):
        assert isinstance(data, SampledTrajectory), "Data must be sampled trajectory"
        # Create individual Catmull-Rom splines for each coordinate axis (dimension)
        self.__splines = [CatmullRomSpline(data.t, data.p[:,i]) for i in range(data.dims)]

    @property
    def dims(self):
        """Returns the number of spatial dimensions (e.g. 1-D, 2-D, 3-D)"""
        return len(self.__splines)

    def p(self, t):
        """Computes the position(s) for the given time(s) and returns it as a NumPy array"""
        assert isRealScalarOrVector(t), "t must be a real-valued scalar or vector"

        # Convert the evaluation points into a ndarray
        if isRealScalar(t):
            t = np.array([t], dtype=np.float)
        else:
            t = np.asarray(t, dtype=np.float)

        N = len(t)
        c = [np.reshape(s(t), (N,1)) for s in self.__splines]
        return np.hstack(c)
