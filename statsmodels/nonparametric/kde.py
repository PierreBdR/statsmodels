r"""
Module implementing kernel-based estimation of density of probability.

Given a kernel :math:`K`, the density function is estimated from a sampling 
:math:`X = \{X_i \in \mathbb{R}^n\}_{i\in\{1,\ldots,m\}}` as:

.. math::

    f(\mathbf{z}) \triangleq \frac{1}{hW} \sum_{i=1}^m \frac{w_i}{\lambda_i}
    K\left(\frac{X_i-\mathbf{z}}{h\lambda_i}\right)

    W = \sum_{i=1}^m w_i

where :math:`h` is the bandwidth of the kernel, :math:`w_i` are the weights of 
the data points and :math:`\lambda_i` are the adaptation factor of the kernel 
width.

The kernel is a function of :math:`\mathbb{R}^n` such that:

.. math::

    \begin{array}{rclcl}
       \idotsint_{\mathbb{R}^n} f(\mathbf{z}) d\mathbf{z} 
       & = & 1 & \Longleftrightarrow & \text{$f$ is a probability}\\
       \idotsint_{\mathbb{R}^n} \mathbf{z}f(\mathbf{z}) d\mathbf{z} &=& 
       \mathbf{0} & \Longleftrightarrow & \text{$f$ is 
       centered}\\
       \forall \mathbf{u}\in\mathbb{R}^n, \|\mathbf{u}\| 
       = 1\qquad\int_{\mathbb{R}} t^2f(t \mathbf{u}) dt &\approx& 
       1 & \Longleftrightarrow & \text{The co-variance matrix of $f$ is close 
       to be the identity.}
    \end{array}

The constraint on the covariance is only required to provide a uniform meaning 
for the bandwidth of the kernel.

If the domain of the density estimation is bounded to the interval 
:math:`[L,U]`, the density is then estimated with:

.. math::

    f(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{\lambda_i}
    \hat{K}(x;X,\lambda_i h,L,U)

where :math:`\hat{K}` is a modified kernel that depends on the exact method 
used. Currently, only 1D KDE supports bounded domains.

References
----------
Wasserman, L. All of Nonparametric Statistics Springer, 2005

http://en.wikipedia.org/wiki/Kernel_%28statistics%29

Silverman, B.W.  Density Estimation for Statistics and Data Analysis.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from . import kernels, bandwidths
from .kde_methods import KDEMethod
from . import kde1d_methods, kdend_methods, kdend_methods
from . import kde_multivariate
from .kde_utils import atleast_2df
from ..compat.python import string_types

#default_method = kde1d_methods.Reflection
#default_method = kdend_methods.KDEnDMethod
default_method = kde_multivariate.MultivariateKDE

class KDE(object):
    r"""
    Prepare a nD kernel density estimation, possible on a bounded domain.

    :param ndarray exog: 2D array DxN with the N input points in D dimension.
    :param dict kwords: setting attributes at construction time.
        Any named argument will be equivalent to setting the property
        after the fact. For example::

            >>> xs = [1,2,3]
            >>> k = KDE1D(xs, lower=0)

        will be equivalent to::

            >>> k = KDE1D(xs)
            >>> k.lower = 0
    """
    def __init__(self, exog, method=None, **kwords):

        self._method = None
        if method is None:
            self.method = default_method
        else:
            self.method = method

        self.exog = exog

        for n in kwords:
            if hasattr(self, n):
                setattr(self, n, kwords[n])
            else:
                raise AttributeError("Error, unknown attribute: '{}'".format(n))

    def copy(self):
        """
        Shallow copy of the KDE object
        """
        res = KDE.__new__(KDE)
        # Copy private members: start with a single '_'
        res._method = self._method.copy()
        return res

    @property
    def method(self):
        """
        Method used to estimate the KDE.
        """
        return self._method

    @method.setter
    def method(self, m):
        old_method = self._method
        if isinstance(m, type):
            self._method = m()
        else:
            self._method = m
        if old_method is not None:
            self._method.set_from(old_method)

    def fit(self):
        """
        Compute the various parameters needed by the kde method
        """
        return self.method.fit()

    _r_attrs = ['npts', 'ndim', 'total_weights']
    _rw_attrs = ['exog', 'kernel', 'bandwidth']
    _rwd_attrs = ['axis_type', 'weights', 'adjust', 'lower', 'upper']

def _fwd_to_method(cls, attr, read=True, write=False, delete=False):
    def getter(self):
        return getattr(self._method, attr)
    def setter(self, value):
        setattr(self._method, attr, value)
    def deleter(self):
        delattr(self._method, attr)
    if not read:
        getter = None
    if not write:
        setter = None
    if not delete:
        deleter = None
    setattr(cls, attr, property(getter, setter, deleter, getattr(KDEMethod, attr).__doc__))
for attr in KDE._r_attrs:
    _fwd_to_method(KDE, attr, True)
for attr in KDE._rw_attrs:
    _fwd_to_method(KDE, attr, True, True)
for attr in KDE._rwd_attrs:
    _fwd_to_method(KDE, attr, True, True, True)

