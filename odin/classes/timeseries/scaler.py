import abc
from numbers import Number
import numpy as np

from odin.classes.timeseries import ScalerInterface
from odin.classes.strings import err_type


class StandardScaler(ScalerInterface):
    # TODO extend for multivariate
    def __init__(self, mean, std):
        """
        The StandardScaler class can be used to standardize the data.

        Parameters
        ----------
        mean: Number
            Mean of the data to be standardized.
        std: Number
            Standard deviation of the data to be standardized.
        """
        if not isinstance(mean, Number) or not isinstance(std, Number):
            raise TypeError(err_type.format("mean | std"))
        self._mean = mean
        self._std = std

    def transform(self, x):
        """Returns the data standardized

        Parameters
        ----------
        x: Number or list of Number
            Data to be standardized

        Returns
        -------
        Number or list of Number
            Data standardized

        """
        data = np.array(x)
        return (data - self._mean)/self._std
    
    def inverse_transform(self, x):
        """Returns the data by inverse the standardization

        Parameters
        ----------
        x: Number or list of Number
            Data to be inversed

        Returns
        -------
        Number or list of Number
            Data inversed

        """
        data = np.array(x)
        return (data * self._std) + self._mean


class MinMaxScaler(ScalerInterface):
    # TODO extend for multivariate
    def __init__(self, min_v, max_v, min=0, max=1):
        """
        The MinMaxScaler class can be used to normalize the data.

        Parameters
        ----------
        min_v: Number
            Minimum value of the data.
        max_v: Number
            Maximum value of the data.
        min: Number, optional
            Lower bound for the normalization (default is 0)
        max: Number, optional
            Upper bound for the normalization (default is 1)
        """
        if not isinstance(min_v, Number) or not isinstance(max_v, Number):
            raise TypeError(err_type.format("min_v | max_v"))
        if not isinstance(min, Number) or not isinstance(max, Number):
            raise TypeError(err_type.format("min | max"))
        self._min_v = min_v
        self._max_v = max_v
        self._min = min
        self._max = max

    def transform(self, x):
        """Returns the data normalized

        Parameters
        ----------
        x: Number or list of Number
            Data to be normalized

        Returns
        -------
        Number or list of Number
            Data normalized

        """
        data = np.array(x)
        data = (data - self._min_v) / (self._max_v - self._min_v)
        return data * (self._max - self._min) + self._min
    
    def inverse_transform(self, x):
        """Returns the data by inverse the normalization

        Parameters
        ----------
        x: Number or list of Number
            Data to be inversed

        Returns
        -------
        Number or list of Number
            Data inversed

        """
        data = np.array(x)
        data = (data - self._min)/(self._max - self._min)
        return data * (self._max_v - self._min_v) + self._min_v