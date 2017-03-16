from abc import ABC, abstractmethod, abstractproperty
from helpers import *
from inspect import isfunction
from scipy.interpolate import interp1d
from scipy.io import wavfile

class SampledSignal:
    """A finite discrete-time signal (single channel)"""

    def __init__(self, samples, samplingRate):
        self.samples = asRealVector(samples)
        self.samplerate = asPositiveRealNumber(samplingRate)

    @classmethod
    def load(cls, path, verbose=False):
        """Loads a signal from an audio file"""
        samplerate, samples = wavfile.read(path)
        if samples.ndim == 0:
            raise Exception("Audio file contains no data")
        if samples.ndim > 1:
            raise Exception("Audio file contains multiple channels")

        signal = SampledSignal(pcm2float(samples), samplerate)
        if verbose:
            print("Loaded audio file \"%s\": %d samples @ %0.f Hz, duration %0.3f s, peak %0.2f dB" %
                  (path, signal.length, signal.samplerate, signal.duration, 20*np.log10(signal.peak)))
        return signal

    @property
    def length(self):
        """Returns the number of samples"""
        return len(self.samples)

    @property
    def duration(self):
        """Returns the duration in seconds"""
        return len(self.samples)/self.samplerate

    @property
    def peak(self):
        """Returns the peak value"""
        return np.max(np.abs(self.samples))

    def __getitem__(self, key):
        """Returns the sample of the given key"""
        return self.samples[key]

    def __setitem__(self, key, value):
        self.samples[key] = value

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return 'SampledSignal ({})'.format(self.samples)

    # Comparison operators (useful for NumPy ndarray compatability)
    def __eq__(self, other): return self.samples == other
    def __ne__(self, other): return self.samples != other
    def __le__(self, other): return self.samples <= other
    def __lt__(self, other): return self.samples < other
    def __ge__(self, other): return self.samples >= other
    def __gt__(self, other): return self.samples > other

class ContinuousTimeSignal(ABC):
    @abstractmethod
    def __call__(self, times):
        """Returns the signal value(s) at the given time(s) [seconds]"""
        pass

class InterpolatedSignal(ContinuousTimeSignal):
    """A continuous time view on a sampled signal based on interpolation"""

    def __init__(self, signal, timeOffset=0):
        if not isinstance(signal, SampledSignal):
            raise Exception("Expecting a SampledSignal")
        if signal.length == 0:
            raise Exception("Signal must not be empty")

        # Keep an autonomous copy of the required data (no references)
        times = np.arange(signal.length) / signal.samplerate + timeOffset
        self.__timespan = [times[0], times[-1]]
        self.__interpolator = interp1d(times, signal.samples, kind='cubic')

        # Newer SciPy versions:
        #self.__interpolator = interp1d(times, signal.samples,
        #                               assume_sorted=True,
        #                               bounds_error=False,
        #                               fill_value=(signal.samples[0], signal.samples[-1]))

    @property
    def timespan(self):
        """Returns the timespan of the signal [seconds]"""
        return self.__timespan

    def __call__(self, times):
        """Returns the (interpolated) value at the given time(s) [seconds]"""
        # Note: Before/after the definition range the first/last value is continued, hence 'clip'
        return self.__interpolator(np.clip(times, self.__timespan[0], self.__timespan[-1]))
        # Newer SciPy versions: return self.__interpolator(times)


class AnalyticSignal(ContinuousTimeSignal):
    """A continuous time signal defined by a function s(t)"""

    def __init__(self, function):
        if not isfunction(function):
            raise Exception("Function expected")

        self.__function = function

        # Check the return type for a scalar and a vector
        s = self(0)
        if not isNumber(s):
            raise Exception("Function must map a time to a value")

        msg = "Function must map an array of times to an equally-sized array of values"
        v = asVector(self([-1, 0, 1]), errormsg=msg)
        if len(v) != 3: raise Exception(msg)

    def __call__(self, times):
        """Returns the value at the given time(s) [seconds]"""
        return self.__function(asScalarOrVector(times))
