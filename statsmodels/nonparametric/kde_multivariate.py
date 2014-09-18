"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains the multi-variate KDE meta-method.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from statsmodels.compat.python import range
from copy import copy as shallow_copy
from .fast_linbin import fast_linbin as fast_bin
from . import kernels, kernelsnc
from . import kde1d_methods, kdenc_methods, bandwidths
from .kde_utils import numpy_trans_method, AxesType
from .kde_methods import KDEMethod, _array_arg
from .bandwidths import KDE1DAdaptor

def _compute_bandwidth(kde, default):
    """
    Compute the bandwidth and covariance for the estimated model, based of its 
    exog attribute
    """
    n = kde.ndim
    if kde.bandwidth is not None:
        bw = kde.bandwidth
    else:
        bw = default
    if callable(bw):
        bw = bw(kde)
    else:
        adapt = KDE1DAdaptor(kde)
        for i in range(n):
            local_bw = bw[i]
            if callable(local_bw):
                adapt.axis = i
                local_bw = float(local_bw(adapt))
            else:
                local_bw = float(local_bw)
            bn[i] = local_bw
    bw = np.asarray(bw, dtype=float)
    if bw.shape != (n,):
        raise ValueError("Error, there must be one bandwidth per variable")
    return bw

class MultivariateKDE(KDEMethod):
    """
    This class works as an adaptor for various 1D methods to work together.
    """
    def __init__(self, **kwords):
        KDEMethod.__init__(self)
        self._methods = {}
        self._kernels = {}
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

    @property
    def methods(self):
        return self._methods

    def fit(self, kde):
        if len(kde.axis_type) == 1:
            axis_type = AxesType(kde.axis_type[0]*kde.ndim)
        else:
            axis_type = AxesType(kde.axis_type)
        if len(axis_type) != kde.ndim:
            raise ValueError("You must specify exacltly one axis type, or as many as there are axis")
        methods, kernels = self.get_methods(axis_type)
        ndim = kde.ndim
        if ndim == 1:
            return methods[0].fit(kde)
        bw = _compute_bandwidth(kde, self.bandwidth)
        fitted = self.copy()
        fitted._bandwidth = bw
        fitted._axis_type = kde.axis_type
        fitted._kernels = kernels
        fitted._exog = kde.exog
        new_kde = kde.copy()
        new_kde.kernels = kernels
        new_kde.bandwidth = bw
        adapt = KDE1DAdaptor(new_kde)
        for d, m in enumerate(methods):
            adapt.axis = d
            f = m.fit(adapt)
            methods[d] = f
        fitted._methods = methods
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
            self._methods[i].pdf(points, out=full_out, dims=i)
        np.prod(full_out, axis=1, out=out)
        return out

    def __call__(self, points, out=None):
        return self.pdf(points, out)

