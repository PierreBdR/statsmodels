"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains the multi-variate KDE meta-method.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from copy import copy as shallow_copy
from .fast_linbin import fast_linbin as fast_bin
from . import kernels
from .kde_utils import numpy_trans_method, atleast_2df
from . import bandwidths

class AxesType(object):
    _valid_types = 'cou'
    _dtype = np.dtype(np.str_).char+'1'
    def __init__(self, value='c'):
        self._types = np.empty((), dtype=self._dtype)
        self.set(value)

    def set(self, value):
        value = np.array(list(value), dtype=self._types.dtype)
        if any(v not in AxesType._valid_types for v in value):
            raise ValueError("Error, an axis type must be one of 'c', 'o' or 'u'")
        self._types = value

    def copy(self):
        return AxesType(self)

    def __len__(self):
        return len(self._types)

    def __repr__(self):
        return "AxesType('{}')".format(''.join(self._types))

    def __str__(self):
        return ''.join(self._types)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ''.join(self._types[idx])
        return self._types[idx]

    def __iter__(self):
        return iter(self._types)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            value = list(value)
            if any(v not in AxesType._valid_types for v in value):
                raise ValueError("Error, invalid axis type: '{}'".format(''.join(value)))
            self._types[idx] = value
        else:
            if value not in AxesType._valid_types:
                raise ValueError("Error, invalid axis type: '{}'".format(value))
            self._types[idx] = value

    def __delitem__(self, idx):
        del self._types[idx]

    def resize(self, nl, default='c'):
        cur_l = len(self)
        if nl < cur_l:
            self._types = self._types[nl:]
        elif nl > cur_l:
            self._types = np.resize(self._types, nl)
            self._types[cur_l:] = default

    def __eq__(self, other):
        if isinstance(other, AxesType):
            return self._types == other._types
        return self._types == other

    def __ne__(self, other):
        if isinstance(other, AxesType):
            return self._types != other._types
        return self._types != others


class KDEMethod(object):
    """
    This is the base class for KDE methods.

    Although inheriting from it is not required, it is recommended as it will provide quite a few useful services.
    """
    def __init__(self):
        self._exog = None
        self._upper = None
        self._lower = None
        self._axis_type = AxesType('c')
        self._kernel = kernels.normal_kernel()
        self._bandwidth = bandwidths.MultivariateBandwidth()
        self._weights = None
        self._adjust = None
        self._total_weights = None
        self._fitted = None

    @property
    def exog(self):
        """
        ndarray
            Exogenous data set. Its shape is NxD for N points in D dimension.
        """
        return self._exog

    @exog.setter
    def exog(self, value):
        value = atleast_2df(value).astype(float)
        self._exog = value.reshape(self._exog.shape)

    @property
    def ndim(self):
        """
        int
            Number of dimensions of the problem
        """
        return self._exog.shape[1]

    @property
    def npts(self):
        """
        int
        Number of points in the exogenous dataset.
        """
        return self._exog.shape[0]

    @property
    def kernel(self):
        r"""
        Kernel class. This must be an object modeled on :py:class:`kernels.Kernels` or on :py:class:`kernels.Kernel1D`
        for 1D kernels. It is recommended to inherit one of these classes to provide numerical approximation for all
        methods.

        By default, the kernel class is :py:class:`pyqt_fit.kernels.normal_kernel`
        """
        return self._kernel

    @kernel.setter
    def kernel(self, k):
        self._kernel = k

    @property
    def axis_type(self):
        """
        AxesType
            Type of each axis. Each axis type is defined by a letter:
                - c for continuous
                - u for unordered (discrete)
                - o for ordered (discrete)
        """
        return self._axis_type

    @axis_type.setter
    def axis_type(self, value):
        self._axis_type.set(value)

    @axis_type.deleter
    def axis_type(self):
        self._axis_type[:] = 'c'

    @property
    def bandwidth(self):
        """
        Bandwidth of the kernel.
        Can be set either as a fixed value or using a bandwidth calculator,
        that is a function of signature ``w(model)`` that returns the bandwidth.

        See the actual method used for details on the acceptable values.
        """
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        value = np.asarray(value)
        self._bandwidth = value.reshape(self._bandwidth.shape)

    @property
    def weights(self):
        """
        Weigths associated to each data point. It can be either a single value,
        or a 1D-array with a value per data point. If a single value is provided,
        the weights will always be set to 1.
        """
        return self._weights

    @weights.setter
    def weights(self, ws):
        try:
            ws = float(ws)
            self._weights = np.asarray(1.)
            if self._fitted:
                self._total_weights = self.npts
        except TypeError:
            ws = np.atleast_1d(ws).astype(float)
            ws = ws.reshape((self.npts,))
            self._weights = ws
            if self._fitted:
                self._total_weights = sum(ws)

    @weights.deleter
    def weights(self):
        self._weights = np.asarray(1.)
        self._total_weights = self.npts

    @property
    def total_weights(self):
        return self._total_weights

    @property
    def adjust(self):
        """
        Scaling of the bandwidth, per data point. It can be either a single
        value or an array with one value per data point. The real bandwidth 
        then becomes: bandwidth * adjust

        When deleted, the adjusting is reset to 1.
        """
        return self._adjust

    @adjust.setter
    def adjust(self, ls):
        try:
            self._adjust = np.asarray(float(ls))
        except TypeError:
            ls = np.atleast_1d(ls).astype(float)
            ls = ls.reshape((self.npts,))
            self._adjust = ls

    @adjust.deleter
    def adjust(self):
        self._adjust = np.asarray(1.)

    @property
    def lower(self):
        r"""
        Lower bound of the density domain. If deleted, becomes :math:`-\infty` on all dimension.

        Note that for discrete dimensions, the lower bounds will also be reset to 0.
        """
        return self._lower

    @lower.setter
    def lower(self, val):
        self._lower = np.atleast_1d(val)

    @lower.deleter
    def lower(self):
        self._lower = None

    @property
    def upper(self):
        r"""
        Upper bound of the density domain. If deleted, becomes set to :math:`\infty`

        Note that for discrete dimensions, if the upper dimension is 0, it will be set to the maximum observed element.
        """
        return self._upper

    @upper.setter
    def upper(self, val):
        self._upper = np.atleast_1d(val)

    @upper.deleter
    def upper(self):
        self._upper = None

