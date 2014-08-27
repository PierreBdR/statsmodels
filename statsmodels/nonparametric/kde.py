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
from . import kde1d_methods
#from . import kdend_methods
from .kde_utils import numpy_method_idx

default_method = kde1d_methods.Reflection

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
    def __init__(self, exog, **kwords):
        self._exog = None
        self._upper = None
        self._lower = None
        self._kernel = kernels.normal_kernel()

        self._bw = None
        self._covariance = None
        self._method = None

        self.weights = 1.
        self.adjust = 1.

        for n in kwords:
            setattr(self, n, kwords[n])

        self.exog = exog

        if self._bw is None and self._covariance is None:
            self.covariance = bandwidths.scotts_covariance

        if self._method is None:
            self.method = default_method

    def copy(self):
        """
        Shallow copy of the KDE object
        """
        res = KDE.__new__(KDE)
        # Copy private members: start with a single '_'
        for m in self.__dict__:
            if len(m) > 1 and m[0] == '_' and m[1] != '_':
                setattr(res, m, getattr(self, m))
        return res

    @property
    def exog(self):
        return self._exog

    @exog.setter
    def exog(self, xs):
        self._exog = np.atleast_2d(xs)
        assert len(self._exog.shape) == 2, "The attribute 'exog' must be a one-dimension array"

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
    def kernel(self, val):
        self._kernel = val

    @property
    def lower(self):
        r"""
        Lower bound of the density domain. If None, this is :math:`-\infty` on all dimension.
        """
        if self._lower is None:
            return -np.inf*np.ones((self.ndims,), dtype=float)
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
        Upper bound of the density domain. If deleted, becomes set to
        :math:`\infty`
        """
        if self._upper is None:
            return np.inf*np.ones((self.ndims,), dtype=float)
        return self._upper

    @upper.setter
    def upper(self, val):
        self._upper = np.atleast_1d(val)

    @upper.deleter
    def upper(self):
        self._upper = None

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
            self._total_weights = None
        except TypeError:
            ws = np.atleast_1d(ws).astype(float)
            self._weights = ws
            self._total_weights = sum(ws)

    @weights.deleter
    def weights(self):
        self._weights = np.asarray(1.)
        self._total_weights = None

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
            self._adjust = ls

    @adjust.deleter
    def adjust(self):
        self._adjust = np.asarray(1.)

    @property
    def bandwidth(self):
        """
        Bandwidth of the kernel.
        Can be set either as a fixed value or using a bandwidth calculator,
        that is a function of signature ``w(data)`` that returns a single
        value.

        .. note::

            A ndarray with a single value will be converted to a floating point
            value.
        """
        return self._bw

    @bandwidth.setter
    def bandwidth(self, bw):
        self._bw  = bw
        self._covariance = None

    @property
    def covariance(self):
        """
        Covariance of the gaussian kernel.
        Can be set either as a fixed value or using a bandwidth calculator,
        that is a function of signature ``w(data)`` that returns a single
        value.

        .. note::

            A ndarray with a single value will be converted to a floating point
            value.
        """
        return self._covariance

    @covariance.setter
    def covariance(self, cov):
        self._covariance = cov
        self._bw = None

    @property
    def ndims(self):
        """
        Return the number of dimensions of the problem
        """
        return self._exog.shape[0]

    @property
    def npts(self):
        """
        Return the number of points in the exogenous dataset.
        """
        return self._exog.shape[1]

    @property
    def method(self):
        """
        Method used to estimate the KDE.
        """
        return self._method

    @method.setter
    def method(self, m):
        if isinstance(m, type):
            self._method = m()
        else:
            self._method = m

    def fit(self):
        """
        Compute the various parameters needed by the kde method
        """
        if self._weights.shape:
            assert self._weights.shape[0] == self.npts, \
                "There must be either one or as many weights as data points"
        else:
            self._total_weights = self.npts
        if self._adjust.shape:
            assert self._adjust.shape[0] == self.npts, \
                "There must be either one or as many 'adjust' values as data points"
        return self.method.fit(self)

