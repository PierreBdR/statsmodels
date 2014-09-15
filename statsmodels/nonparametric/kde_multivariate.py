"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains the multi-variate KDE meta-method.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from copy import copy as shallow_copy
from .fast_linbin import fast_linbin as fast_bin
from . import kernels, kernelsnc
from . import kde1d_methods, kdenc_methods, bandwidths
from .kde_utils import numpy_trans_method

class KDEAdaptor(object):
    def __init__(self, kde, bws, kernels, axis=None):
        self._axis = axis
        self._kde = kde
        self._kernels = kernels
        self._bw = bws

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, val):
        val = int(val)
        if val < 0 or val >= self._kde.ndim:
            raise ValueError("Error, invalid axis")
        self._axis = val

    @property
    def bandwidth(self):
        return self._bw[self._axis]

    @property
    def kernel(self):
        return self._kernels[self._axis]

    @property
    def bandwidth(self):
        return self._bws[self._axis]

    @property
    def ndim(self):
        return 1

    def fit(self):
        raise NotImplementedError()

    _array_attributes = ['lower', 'upper', 'exog', 'axis_type']

    _constant_attributes = ['weights', 'adjust', 'total_weights', 'npts']

def _add_fwd_array_attr(cls, attr):
    def getter(self):
        return getattr(self._kde, attr)[..., self._axis]
    setattr(cls, attr, property(getter))

def _add_fwd_attr(cls, attr):
    def getter(self):
        return getattr(self._kde, attr)
    setattr(cls, attr, property(getter))

for attr in KDEAdaptor._array_attributes:
    _add_fwd_array_attr(KDEAdaptor, attr)

for attr in KDEAdaptor._constant_attributes:
    _add_fwd_attr(KDEAdaptor, attr)

class MultivariateKDE(object):
    """
    This class works as an adaptor for various 1D methods to work together.
    """
    def __init__(self, **kwords):
        self._exog = None
        self._methods = {}
        self._weights = None
        self._total_weights = None
        self._bw = None
        self._adjust = None
        self._kernels = {}
        self._axis_type = None
        self._kernels_type = dict(c=kernels.normal_kernel1d(),
                                  o=kernelsnc.WangRyzin(),
                                  u=kernelsnc.AitchisonAitken())
        self._methods_type = dict(c=kde1d_methods.Reflection(),
                                  o=kdenc_methods.OrderedKDE(),
                                  u=kdenc_methods.UnorderedKDE())
        for k in kwords:
            if hasattr(self, k):
                setattr(self, k, kwords[k])
            else:
                raise ValueError("Error, unknown attribute '{}'".format(k))

    def copy(self):
        return shallow_copy(self)

    @property
    def axis_type(self):
        """
        String defining the kind of axis, it must be composed of the letters:
            c - continuous
            o - ordered (discrete)
            u - unordered (discrete)
        """
        return self._axis_type

    @property
    def axis_type(self):
        return self._axis_type

    @property
    def methods(self):
        return self._methods

    @property
    def kernels(self):
        return self._kernels

    @property
    def continuous_method(self):
        return self._methods_type['c']

    @continuous_method.setter
    def continuous_method(self, m):
        self._methods_type['c'] = m

    @property
    def ordered_method(self):
        return self._methods_type['o']

    @ordered_method.setter
    def ordered_method(self, m):
        self._methods_type['o'] = m

    @property
    def unordered_method(self):
        return self._methods_type['u']

    @unordered_method.setter
    def unordered_method(self, m):
        self._methods_type['u'] = m

    @property
    def adjust(self):
        return self._adjust

    @adjust.setter
    def adjust(self, val):
        try:
            self._adjust = np.asarray(float(val))
        except TypeError:
            val = np.atleast_1d(val).astype(float)
            if val.shape != (self.npts,):
                raise ValueError("Error, adjust must be a single value or a value per point")
            self._adjust = val
        if self._methods:
            for m in self._methods:
                m.adjust = self._adjust

    @adjust.deleter
    def adjust(self):
        self.adjust = np.asarray(1.)

    @property
    def npts(self):
        return self._exog.shape[0]

    @property
    def ndim(self):
        return self._exog.shape[1]

    @property
    def bandwidth(self):
        return self._bw

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def kernels(self):
        return self._kernels

    def get_methods(self, axis_types):
        m = [None]*len(axis_types)
        k = [None]*len(axis_types)
        for i, t in enumerate(axis_types):
            try:
                m[i] = self._methods[i]
            except (IndexError, KeyError):
                m[i] = self._methods_type[t].copy()
            try:
                k[i] = self._kernels[i]
            except (IndexError, KeyError):
                k[i] = self._kernels_type[t].for_ndim(1)
        return m, k

    def fit(self, kde):
        methods, kernels = self.get_methods(kde.axis_type)
        if kde.ndim == 1:
            return methods[0].fit(kde)
        if callable(kde.bandwidth):
            bw = kde.bandwidth(kde)
        else:
            bw = np.atleast_1d(kde.bandwidth)
        if bw.shape != (kde.ndim,):
            raise ValueError("There should be one bandwidth per dimension")
        k = KDEAdaptor(kde, bw, kernels)
        fitted = self.copy()
        fitted._bw = bw
        fitted._methods = methods
        fitted._axis_type = kde.axis_type
        fitted._kernels = kernels
        fitted._exog = kde.exog
        k._bw = bw
        for i, m in enumerate(methods):
            k.axis = i
            fitted.methods[i] = m.fit(k)
        fitted._lower = np.array([m.lower for m in fitted.methods])
        fitted._upper = np.array([m.upper for m in fitted.methods])
        fitted._bin_data = np.concatenate([m.to_bin[:,None] for m in fitted.methods], axis=1)
        fitted._weights = kde.weights
        fitted._adjust = kde.adjust
        fitted._total_weights = kde.total_weights
        return fitted

    @numpy_trans_method('ndim', 1)
    def pdf(self, points, out):
        full_out = np.empty_like(points)
        for i in range(self.ndim):
            self._methods[i].pdf(points, out=full_out[...,i], dims=i)
        np.prod(full_out, axis=1, out=out)
        return out
