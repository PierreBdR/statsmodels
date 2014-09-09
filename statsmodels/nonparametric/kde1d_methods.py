r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains a set of methods to compute univariate KDEs.

These methods provide various variations on :math:`\hat{K}(x;X,h,L,U)`, the 
modified kernel evaluated on the point :math:`x` based on the estimation points 
:math:`X`, a bandwidth :math:`h` and on the domain :math:`[L,U]`.

The definitions of the methods rely on the following definitions:

.. math::

   \begin{array}{rcl}
     a_0(l,u) &=& \int_l^u K(z) dz\\
     a_1(l,u) &=& \int_l^u zK(z) dz\\
     a_2(l,u) &=& \int_l^u z^2K(z) dz
   \end{array}

These definitions correspond to:

- :math:`a_0(l,u)` -- The partial cumulative distribution function
- :math:`a_1(l,u)` -- The partial first moment of the distribution. In 
  particular, :math:`a_1(-\infty, \infty)` is the mean of the kernel (i.e. and 
  should be 0).
- :math:`a_2(l,u)` -- The partial second moment of the distribution. In 
  particular, :math:`a_2(-\infty, \infty)` is the variance of the kernel (i.e. 
  which should be close to 1, unless using higher order kernel).

References:
```````````
.. [1] Jones, M. C. 1993. Simple boundary correction for kernel density
    estimation. Statistics and Computing 3: 135--146.
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import fftpack, integrate, optimize
from .kde_utils import make_ufunc, namedtuple, numpy_trans1d_method, numpy_trans1d, finite
from .grid import Grid
from .fast_linbin import fast_linbin as fast_bin
from copy import copy as shallow_copy
from .kernels import Kernel1D
from .grid_interpolation import GridInterpolator

def generate_grid(kde, N=None, cut=None):
    r"""
    Helper method returning a regular grid on the domain of the KDE.

    Parameters
    ----------
    kde: KDE1DMethod
        Fitted KDE object
    N: int
        Number of points in the grid
    cut: float
        For unbounded domains, how far past the maximum should
        the grid extend to, in term of KDE bandwidth

    Returns
    -------
    A vector of N regularly spaced points
    """
    N = kde.grid_size(N)
    if cut is None:
        cut = kde.kernel.cut
    if kde.lower == -np.inf:
        lower = np.min(kde.exog) - cut * kde.bandwidth
    else:
        lower = kde.lower
    if kde.upper == np.inf:
        upper = np.max(kde.exog) + cut * kde.bandwidth
    else:
        upper = kde.upper
    mesh, step = np.linspace(lower, upper, N, endpoint=False, retstep=True)
    mesh += step/2
    return Grid(mesh, bounds=[lower, upper])

def _compute_bandwidth(kde):
    """
    Compute the bandwidth and covariance for the estimated model, based of its 
    exog attribute
    """
    if kde.bandwidth is not None:
        if callable(kde.bandwidth):
            bw = float(kde.bandwidth(kde))
        else:
            bw = float(kde.bandwidth)
        return bw, bw*bw
    elif kde.covariance is not None:
        if callable(kde.covariance):
            cov = float(kde.covariance(kde))
        else:
            cov = float(kde.covariance)
        return np.sqrt(cov), cov
    raise ValueError("Bandwidth or covariance needs to be specified")

class KDE1DMethod(object):
    """
    Base class providing a default grid method and a default method for 
    unbounded evaluation of the PDF and CDF. It also provides default methods 
    for the other metrics, based on PDF and CDF calculations.

    :Note:
        - It is expected that all grid methods will return the same grid if 
          used with the same arguments.
        - It is fair to assume all array-like arguments will be at least 1D 
          arrays.
    """

    name = 'unbounded'

    def __init__(self):
        self._exog = None
        self._upper = None
        self._lower = None
        self._kernel = None
        self._weights = None
        self._adjust = None
        self._total_weights = None
        self._bw = None
        self._cov = None

    def fit(self, kde, compute_bandwidth=True):
        """
        Extract the parameters required for the computation and returns 
        a stand-alone estimator capable of performing most computations.

        Parameters
        ----------
        kde: pyqt_fit.kde.KDE
            KDE object being fitted
        compute_bandwidth: bool
            If true (default), the bandwidth is computed

        Returns
        -------
        An estimator object that doesn't depend on the KDE object.


        Notes
        -----
        By default, most values can be adjusted after estimation. However, it 
        is not allowed to change the number of exogenous variables or the 
        dimension of the problem.
        """
        assert kde.ndim == 1, "Error, this is a 1D method, expecting a 1D problem"
        fitted = self.copy()
        if compute_bandwidth:
            bw, cov = _compute_bandwidth(kde)
            fitted._bw = bw
            fitted._cov = cov
        fitted._exog = kde.exog.reshape((kde.npts,))
        fitted._upper = float(kde.upper)
        fitted._lower = float(kde.lower)
        fitted._kernel = kde.kernel.for_ndim(1)
        fitted._weights = kde.weights
        fitted._adjust = kde.adjust
        fitted._total_weights = kde.total_weights
        return fitted

    def copy(self):
        return shallow_copy(self)

    @property
    def adjust(self):
        return self._adjust

    @adjust.setter
    def adjust(self, val):
        try:
            self._adjust = np.asarray(float(val))
        except TypeError:
            val = np.atleast_1d(val).astype(float)
            assert val.shape == (self.npts,), \
                    "Adjust must be a single values or a 1D array with value per input point"
            self._adjust = val

    @adjust.deleter
    def adjust(self):
        self._adjust = np.asarray(1.)

    @property
    def ndim(self):
        """
        Dimension of the problem
        """
        return 1

    @property
    def npts(self):
        """
        Number of points in the setup
        """
        return self._exog.shape[0]

    @property
    def bandwidth(self):
        """
        Selected bandwidth.

        Unlike the bandwidth for the KDE, this must be an actual value and not 
        a method.
        """
        return self._bw

    @bandwidth.setter
    def bandwidth(self, val):
        val = float(val)
        assert val > 0, "The bandwidth must be strictly positive"
        self._bw = val
        self._cov = val*val

    @property
    def covariance(self):
        """
        Square of the selected bandwidth

        Unlike the covariance for the KDE, this must be an actual value and not 
        a method.
        """
        return self._cov

    @covariance.setter
    def covariance(self, val):
        val = float(val)
        assert val > 0, "The covariance must be strictly positive"
        self._cov = val
        self._bw = np.sqrt(val)

    def update_inputs(self, exog, weights=1., adjust=1.):
        """
        Update all the variable lengths inputs at once to ensure consistency
        """
        exog = np.atleast_1d(exog)
        if exog.ndim != 1:
            raise ValueError("Error, exog must be a 1D array (nb dimensions: {})".format(exog.ndim))
        weights = np.asarray(weights)
        adjust = np.asarray(adjust)
        if weights.ndim != 0 and weights.shape != exog.shape:
            raise ValueError("Error, weights must be either a single number, or an array the same shape as exog")
        if adjust.ndim != 0 and adjust.shape != exog.shape:
            raise ValueError("Error, adjust must be either a single number, or an array the same shape as exog")
        self._exog = exog
        self._weights = weights
        self._adjust = adjust
        if weights.ndim > 0:
            self._total_weights = weights.sum()
        else:
            self._total_weights = self.npts

    @property
    def exog(self):
        """
        Input points.

        Notes
        -----
        At that point, you are not allowed to change the number of exogenous 
        points.
        """
        return self._exog

    @exog.setter
    def exog(self, value):
        value = np.atleast_1d(value).reshape(self._exog.shape)
        self._exog = value

    @property
    def lower(self):
        """
        Lower bound of the problem domain
        """
        return self._lower

    @lower.setter
    def lower(self, val):
        val = float(val)
        self._lower = val

    @property
    def upper(self):
        """
        Upper bound of the problem domain
        """
        return self._upper

    @upper.setter
    def upper(self, val):
        val = float(val)
        self._upper = val

    @property
    def kernel(self):
        """
        Kernel used for the estimation
        """
        return self._kernel

    @kernel.setter
    def kernel(self, ker):
        self._kernel = ker

    @property
    def weights(self):
        """
        Weights for the points in ``KDE1DMethod.exog``
        """
        return self._weights

    @weights.setter
    def weights(self, val):
        val = np.asarray(val)
        if val.shape:
            val = val.reshape(self._exog.shape)
            self._weights = val

    @property
    def total_weights(self):
        """
        Sum of the point weights
        """
        return self._total_weights

    @property
    def closed(self):
        """
        Returns true if the density domain is closed (i.e. lower and upper
        are both finite)
        """
        return self.lower > -np.inf and self.upper < np.inf

    @property
    def bounded(self):
        """
        Returns true if the density domain is actually bounded
        """
        return self.lower > -np.inf or self.upper < np.inf


    @numpy_trans1d_method()
    def pdf(self, points, out):
        """
        Compute the PDF of the estimated distribution.

        Parameters
        ----------
        points: ndarray
            Points to evaluate the distribution on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        Returns the ``out`` variable, updated with the PDF.

        :Default: Direct implementation of the formula for unbounded pdf
            computation.
        """
        exog = self.exog
        points = points[..., np.newaxis]

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw

        kernel = self.kernel

        terms = kernel(z)

        terms *= self.weights / bw

        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    def __call__(self, points, out=None):
        """
        Call the :py:meth:`pdf` method.
        """
        return self.pdf(points, out)

    @numpy_trans1d_method()
    def cdf(self, points, out):
        r"""
        Compute the CDF of the estimated distribution, defined as:

        .. math::

            cdf(x) = P(X \leq x) = \int_l^x p(t) dt

        where :math:`l` is the lower bound of the distribution domain and 
        :math:`p` the density of probability

        Parameters
        ----------
        points: ndarray
            Points to evaluate the CDF on
        out: ndarray
            Result object. If must have the same shapes as ``points``

        Returns
        -------
        The ``out`` variable, updated with the CDF.

        :Default: Direct implementation of the formula for unbounded CDF
            computation.
        """
        exog = self.exog
        points = points[..., np.newaxis]
        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw

        kernel = self.kernel

        terms = kernel.cdf(z)
        terms *= self.weights

        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    @numpy_trans1d_method()
    def icdf(self, points, out):
        r"""
        Compute the inverse cumulative distribution (quantile) function, 
        defined as:

        .. math::

            icdf(p) = \inf\left\{x\in\mathbb{R} : cdf(x) \geq p\right\}

        :param ndarray points: Points to evaluate the iCDF on
        :param ndarray out: Result object. If must have the same shapes as ``points``
        :rtype: ndarray
        :return: Returns the ``out`` variable, updated with the iCDF.
        :Default: First approximate the result using linear interpolation on
            the CDF and refine the result numerically using the Newton method.
        """
        xs, ys = self.cdf_grid()
        xs = xs.linear()
        coarse_result = np.interp(points, ys, xs, self.lower, self.upper)
        lower = self.lower
        upper = self.upper
        cdf = self.cdf
        pdf_out = np.empty(1, dtype=float)
        def pdf(x):
            if x <= lower:
                return 0
            if x >= upper:
                return 0
            return self.pdf(np.atleast_1d(x), pdf_out)

        @make_ufunc()
        def find_inverse(p, approx):
            if p > 1-1e-10:
                return upper
            if p < 1e-10:
                return lower
            if approx >= xs[-1] or approx <= xs[0]:
                return approx
            cdf_out = np.empty(1, dtype=float)
            def f(x):
                if x <= lower:
                    return -p
                elif x >= upper:
                    return 1-p
                return cdf(np.atleast_1d(x), cdf_out) - p
            return optimize.newton(f, approx, fprime=pdf, tol=1e-6)

        return find_inverse(points, coarse_result, out=out)

    @numpy_trans1d_method()
    def sf(self, points, out):
        r"""
        Compute the survival function, defined as:

        .. math::

            sf(x) = P(X \geq x) = \int_x^u p(t) dt = 1 - cdf(x)

        :param ndarray points: Points to evaluate the survival function on
        :param ndarray out: Result object. If must have the same shapes as ``points``
        :rtype: ndarray
        :return: Returns the ``out`` variable, updated with the survival function.
        :Default: Compute explicitly :math:`1 - cdf(x)`
        """
        self.cdf(points, out)
        out -= 1
        out *= -1
        return out

    @numpy_trans1d_method()
    def isf(self, points, out):
        r"""
        Compute the inverse survival function, defined as:

        .. math::

            isf(p) = \sup\left\{x\in\mathbb{R} : sf(x) \leq p\right\}

        :param ndarray points: Points to evaluate the iSF on
        :param ndarray out: Result object. If must have the same shapes as ``points``
        :rtype: ndarray
        :return: Returns the ``out`` variable, updated with the inverse survival function.
        :Default: Compute :math:`icdf(1-p)`
        """
        return self.icdf(1-points, out)

    @numpy_trans1d_method()
    def hazard(self, points, out):
        r"""
        Compute the hazard function evaluated on the points.

        The hazard function is defined as:

        .. math::

            h(x) = \frac{p(x)}{sf(x)}

        where :math:`p(x)` is the probability density function and 
        :math:`sf(x)` is the survival function.

        :param ndarray points: Points to evaluate the hazard function on
        :param ndarray out: Result object. If must have the same shapes as ``points``
        :rtype: ndarray
        :return: Returns the ``out`` variable, updated with the hazard function
        :Default: Compute explicitly :math:`pdf(x) / sf(x)`
        """
        self.pdf(points, out=out)
        sf = np.empty(out.shape, dtype=out.dtype)
        self.sf(points, sf)
        sf[sf < 0] = 0 # Some methods can produce negative sf
        out /= sf
        return out

    @numpy_trans1d_method()
    def cumhazard(self, points, out):
        r"""
        Compute the cumulative hazard function evaluated on the points.

        The hazard function is defined as:

        .. math::

            ch(x) = \int_l^x h(t) dt = -\ln sf(x)

        where :math:`l` is the lower bound of the domain, :math:`h` the hazard 
        function and :math:`sf` the survival function.

        :param ndarray points: Points to evaluate the cumulative hazard function on
        :param ndarray out: Result object. If must have the same shapes as ``points``
        :rtype: ndarray
        :return: Returns the ``out`` variable, updated with the cumulative hazard function
        :Default: Compute explicitly :math:`-\ln sf(x)`
        """
        self.sf(points, out)
        out[out < 0] = 0 # Some methods can produce negative sf
        np.log(out, out=out)
        out *= -1
        return out

    def grid(self, N=None, cut=None):
        """
        Evaluate the PDF of the distribution on a regular grid with at least 
        ``N`` elements.

        :param int N: minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        :param float cut: for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.
        :rtype: (ndarray, ndarray)
        :returns: The array of positions the PDF has been estimated on, and the
            estimations.
        :Default: Evaluate :math:`pdf(x)` on a grid generated using
            :py:func:`generate_grid`
        """
        N = self.grid_size(N)
        g = generate_grid(self, N, cut)
        out = np.empty(g.shape, dtype=float)
        return g, self.pdf(g.full(), out)

    def cdf_grid(self, N=None, cut=None):
        """
        Evaluate the CDF of the distribution on a regular grid with at least 
        ``N`` elements.

        :param int N: minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        :param float cut: for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.
        :rtype: (ndarray, ndarray)
        :returns: The array of positions the CDF has been estimated on, and the
            estimations.
        :Default: Evaluate :math:`cdf(x)` on a grid generated using
            :py:func:`generate_grid`
        """
        N = self.grid_size(N)
        if N <= 2**11:
            g = generate_grid(self, N, cut)
            out = np.empty(g.shape, dtype=float)
            return g, self.cdf(g.full(), out)
        return self.numeric_cdf_grid(N, cut)

    def icdf_grid(self, N=None, cut=None):
        """
        Compute the inverse cumulative distribution (quantile) function on 
        a grid.

        :Note: The default implementation is not as good an approximation as
            the plain icdf default method.

        :param int N: minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        :param float cut: for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.
        :rtype: (ndarray, ndarray)
        :returns: The array of positions the CDF has been estimated on, and the
            estimations.
        :Default: Linear interpolation of the inverse CDF on a grid
        """
        xs, ys = self.cdf_grid(N, cut)
        xs = xs.linear()
        N = len(xs)
        points = np.linspace(0, 1, N)
        icdf = np.interp(points, ys, xs, self.lower, self.upper)
        return Grid(points, bounds=[0,1]), icdf

    def sf_grid(self, N=None, cut=None):
        r"""
        Compute the survival function on a grid.

        :param int N: minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        :param float cut: for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.
        :rtype: (ndarray, ndarray)
        :returns: The array of positions the survival function has been
            estimated on, and the estimations.
        :Default: Compute explicitly :math:`1 - cdf(x)`
        """
        points, out = self.cdf_grid(N, cut)
        out -= 1
        out *= -1
        return points, out

    def isf_grid(self, N=None, cut=None):
        """
        Compute the inverse survival function on a grid.

        :Note: The default implementation is not as good an approximation as
            the plain isf default method.

        :param int N: minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        :param float cut: for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.
        :rtype: (ndarray, ndarray)
        :returns: The array of positions the CDF has been estimated on, and the
            estimations.
        :Default: Linear interpolation of the inverse survival function on a grid
        """
        xs, ys = self.sf_grid(N, cut)
        xs = xs.full()
        N = len(xs)
        points = np.linspace(0, 1, N)
        isf = np.interp(points, ys[::-1], xs[::-1], self.upper, self.lower)
        return Grid(points, bounds=[0,1]), isf

    def hazard_grid(self, N=None, cut=None):
        r"""
        Compute the hazard function on a grid.

        :param int N: minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        :param float cut: for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.
        :rtype: (ndarray, ndarray)
        :returns: The array of positions the hazard function has been
            estimated on, and the estimations.
        :Default: Compute explicitly :math:`pdf(x) / sf(x)`
        """
        points, out = self.grid(N, cut)
        _, sf = self.sf_grid(N, cut)
        sf[sf < 0] = 0 # Some methods can produce negative sf
        out /= sf
        return points, out

    def cumhazard_grid(self, N=None, cut=None):
        r"""
        Compute the hazard function on a grid.

        :param int N: minimum number of element in the returned grid. Most
            methods will want to round it to the next power of 2.
        :param float cut: for unbounded domains, how far from the last data
            point should the grid go, as a fraction of the bandwidth.
        :rtype: (ndarray, ndarray)
        :returns: The array of positions the hazard function has been
            estimated on, and the estimations.
        :Default: Compute explicitly :math:`-\ln sf(x)`
        """
        points, out = self.sf_grid(N, cut)
        out[out < 0] = 0 # Some methods can produce negative sf
        np.log(out, out=out)
        out *= -1
        return points, out

    def __str__(self):
        """
        Return the name of the method
        """
        return self.name

    @numpy_trans1d_method()
    def numeric_cdf(self, points, out):
        """
        Provide a numeric approximation of the CDF based on integrating the pdf 
        using :py:func:`scipy.integrate.quad`.
        """
        pts = points.ravel()

        pts[pts < self.lower] = self.lower
        pts[pts > self.upper] = self.upper

        ix = pts.argsort()

        sp = pts[ix]

        pdf_out = np.empty((1,), dtype=float)
        def pdf(x):
            return self.pdf(np.array([x]), pdf_out)

        @make_ufunc()
        def comp_cdf(i):
            low = self.lower if i == 0 else sp[i-1]
            return integrate.quad(pdf, low, sp[i])[0]

        parts = np.empty(sp.shape, dtype=float)
        comp_cdf(np.arange(len(sp)), out=parts)

        ints = parts.cumsum()

        out.put(ix, ints)
        return out

    def numeric_cdf_grid(self, N=None, cut=None):
        """
        Compute the CDF on a grid using a trivial, but fast, numeric 
        integration of the pdf.
        """
        pts, pdf = self.grid(N, cut)
        return pts, pts.cum_integrate(pdf)

    def grid_size(self, N=None):
        if N is None:
            return 2**10
        return N

def fftdensity(exog, kernel_rfft, bw, lower, upper, N, weights, total_weights):
    """
    Compute the density estimate using a FFT approximation.

    Parameters
    ----------
    exog: ndarray
        1D array with the data to fit
    kernel_rfft: function
        Function computing the rFFT for the kernel
    lower: float
        Lower bound on which to compute the density
    upper: float
        Upper bound on which to compute the density
    N: int
        Number of buckets to compute
    weights: ndarray or None
        Weights of the data, or None if they all have the same weight.
    total_weights: float
        Sum of the weights, or len(exog) if weights is None

    Returns
    -------
    mesh: ndarray
        Points on which the density had been evaluated
    density: ndarray
        Density evaluated on the mesh

    Notes
    -----
    No checks are made to ensure the consistency of the input!
    """
    R = upper - lower
    mesh, DataHist = fast_bin(exog, [lower, upper], N, weights=weights, bin_type='C')
    DataHist /= total_weights * mesh.start_interval[0]
    FFTData = np.fft.rfft(DataHist)

    smth = kernel_rfft(len(DataHist), mesh.start_interval[0]/bw)

    SmoothFFTData = FFTData * smth
    density = np.fft.irfft(SmoothFFTData, len(DataHist))
    return mesh, density

class Cyclic(KDE1DMethod):
    r"""
    This method assumes cyclic boundary conditions and works only for closed 
    boundaries.

    The estimation is done with a modified kernel given by:

    .. math::

        \hat{K}(x; X, h, L, U) \triangleq K(z)
        + K\left(z - \frac{U-L}{h}\right)
        + K\left(z + \frac{U-L}{h}\right)

    where:

    .. math::

        z = \frac{x-X}{h}

    When computing grids, if the bandwidth is constant, the result is computing 
    using FFT.
    """

    name = 'cyclic'

    @numpy_trans1d_method()
    def pdf(self, points, out):
        if not self.bounded:
            return KDE1DMethod.pdf(self, points, out)
        if not self.closed:
            raise ValueError("Cyclic boundary conditions can only be used with "
                             "closed or un-bounded domains.")

        exog = self.exog
        points = np.atleast_1d(points)[..., np.newaxis]

        # Make sure points are between the bounds
        if any(points < self.lower) or any(points > self.upper):
            points = points - self.lower
            points %= self.upper - self.lower
            points += self.lower

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw
        L = self.lower
        U = self.upper

        span = (U - L) / bw

        kernel = self.kernel

        terms = kernel(z)
        terms += kernel(z + span) # Add points to the left
        terms += kernel(z - span) # Add points to the right

        terms *= self.weights / bw
        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    @numpy_trans1d_method()
    def cdf(self, points, out):
        if not self.bounded:
            return KDE1DMethod.cdf(self, points, out)
        if not self.closed:
            raise ValueError("Cyclic boundary conditions can only be used with "
                             "closed or unbounded domains.")

        exog = self.exog
        points = np.atleast_1d(points)[..., np.newaxis]

        # Make sure points are between the bounds
        if any(points < self.lower) or any(points > self.upper):
            points = points - self.lower
            points %= self.upper - self.lower
            points += self.lower

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw
        L = self.lower
        U = self.upper

        span = (U - L) / bw

        kernel = self.kernel

        terms = kernel.cdf(z)
        terms -= kernel.cdf((L - exog) / bw) # Remove the parts left of the lower bound

        terms += kernel.cdf(z + span) # Repeat on the left
        terms -= kernel.cdf((L - exog) / bw + span) # Remove parts left of lower bounds

        terms += kernel.cdf(z - span) # Repeat on the right

        terms *= self.weights
        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    def grid(self, N=None, cut=None):
        """
        FFT-based estimation of KDE estimation, i.e. with cyclic boundary
        conditions. This works only for closed domains, fixed bandwidth
        (i.e. adjust = 1) and gaussian kernel.
        """
        if self.adjust.shape:
            return KDE1DMethod.grid(self, N, cut)
        if self.bounded and not self.closed:
            raise ValueError("Error, cyclic boundary conditions require "
                             "a closed or un-bounded domain.")
        bw = self.bandwidth * self.adjust
        exog = self.exog
        N = self.grid_size(N)

        lower = self.lower
        upper = self.upper

        if upper == np.inf:
            if cut is None:
                cut = self.kernel.cut
            lower = np.min(exog) - cut * self.bandwidth
            upper = np.max(exog) + cut * self.bandwidth

        weights = self.weights
        if not weights.shape:
            weights = None

        return fftdensity(exog, self.kernel.rfft, bw, lower, upper, N, weights, self.total_weights)

    def grid_size(self, N=None):
        if N is None:
            if self.adjust.shape:
                return 2**10
            return 2**16
        return N # 2 ** int(np.ceil(np.log2(N)))

Unbounded = Cyclic

def dctdensity(exog, kernel_dct, bw, lower, upper, N, weights, total_weights):
    """
    Compute the density estimate using a DCT approximation.

    Parameters
    ----------
    exog: ndarray
        1D array with the data to fit
    kernel_dct: function
        Function computing the DCT of the kernel
    lower: float
        Lower bound on which to compute the density
    upper: float
        Upper bound on which to compute the density
    N: int
        Number of buckets to compute
    weights: ndarray or None
        Weights of the data, or None if they all have the same weight.
    total_weights: float
        Sum of the weights, or len(exog) if weights is None

    Returns
    -------
    mesh: ndarray
        Points on which the density had been evaluated
    density: ndarray
        Density evaluated on the mesh

    Notes
    -----
    No checks are made to ensure the consistency of the input!
    """
    R = upper - lower

    # Histogram the data to get a crude first approximation of the density
    mesh, DataHist = fast_bin(exog, [lower, upper], N, weights=weights, bin_type='R')

    DataHist /= total_weights * (2*R)
    DCTData = fftpack.dct(DataHist, norm=None)

    smth = kernel_dct(len(DataHist), mesh.start_interval/bw)

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData * smth
    # Inverse DCT to get density
    density = fftpack.idct(SmDCTData, norm=None)

    return mesh, density

class Reflection(KDE1DMethod):
    r"""
    This method consist in simulating the reflection of the data left and 
    right of the boundaries. If one of the boundary is infinite, then the 
    data is not reflected in that direction. To this purpose, the kernel is 
    replaced with:

    .. math::

        \hat{K}(x; X, h, L, U) \triangleq K(z)
        + K\left(\frac{x+X-2L}{h}\right)
        + K\left(\frac{x+X-2U}{h}\right)

    where:

    .. math::

        z = \frac{x-X}{h}


    See the :py:mod:`pyqt_fit.kde1d_methods` for a description of the various symbols.

    When computing grids, if the bandwidth is constant, the result is computing 
    using CDT.
    """

    name = 'reflection'

    @numpy_trans1d_method()
    def pdf(self, points, out):
        if not self.bounded:
            return KDE1DMethod.pdf(self, points, out)

        exog = self.exog
        points = points[..., np.newaxis]

        # Make sure points are between the bounds, with reflection if needed
        if any(points < self.lower) or any(points > self.upper):
            span = self.upper - self.lower
            points = points - (self.lower + span)
            points %= 2*span
            points -= self.lower + span
            points = np.abs(points)

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw
        z1 = (points + exog) / bw
        L = self.lower
        U = self.upper

        kernel = self.kernel

        terms = kernel(z)

        if L > -np.inf:
            terms += kernel(z1 - (2 * L / bw))

        if U < np.inf:
            terms += kernel(z1 - (2 * U / bw))

        terms *= self.weights / bw
        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    @numpy_trans1d_method()
    def cdf(self, points, out):
        if not self.bounded:
            return KDE1DMethod.cdf(self, points, out)

        exog = self.exog
        points = points[..., np.newaxis]

        # Make sure points are between the bounds, with reflection if needed
        if any(points < self.lower) or any(points > self.upper):
            span = self.upper - self.lower
            points = points - (self.lower + span)
            points %= 2*span
            points -= self.lower + span
            points = np.abs(points)

        bw = self.bandwidth * self.adjust

        z = (points - exog) / bw
        z1 = (points + exog) / bw
        L = self.lower
        U = self.upper

        kernel = self.kernel

        terms = kernel.cdf(z)

        if L > -np.inf:
            terms -= kernel.cdf((L - exog) / bw) # Remove the truncated part on the left
            terms += kernel.cdf(z1 - (2 * L / bw)) # Add the reflected part
            terms -= kernel.cdf((exog - L) / bw) # Remove the truncated part from the reflection

        if U < np.inf:
            terms += kernel.cdf(z1 - (2 * U / bw)) # Add the reflected part

        terms *= self.weights
        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    def grid(self, N=None, cut=None):
        """
        DCT-based estimation of KDE estimation, i.e. with reflection boundary 
        conditions. This works only for fixed bandwidth (i.e. adjust = 1) and 
        gaussian kernel.

        For open domains, the grid is taken with 3 times the bandwidth as extra 
        space to remove the boundary problems.
        """
        if self.adjust.shape:
            return KDE1DMethod.grid(self, N, cut)

        bw = self.bandwidth * self.adjust
        exog = self.exog
        N = self.grid_size(N)

        if cut is None:
            cut = self.kernel.cut

        if self.lower == -np.inf:
            lower = np.min(exog) - cut * self.bandwidth
        else:
            lower = self.lower
        if self.upper == np.inf:
            upper = np.max(exog) + cut * self.bandwidth
        else:
            upper = self.upper

        weights = self.weights
        if not weights.shape:
            weights = None

        return dctdensity(exog, self.kernel.dct, bw, lower, upper, N, weights, self.total_weights)

    def grid_size(self, N=None):
        if N is None:
            if self.adjust.shape:
                return 2**10
            return 2**16
        return N # 2 ** int(np.ceil(np.log2(N)))

class Renormalization(Unbounded):
    r"""
    This method consists in using the normal kernel method, but renormalize 
    to only take into account the part of the kernel within the domain of the 
    density [1]_.

    The kernel is then replaced with:

    .. math::

        \hat{K}(x;X,h,L,U) \triangleq \frac{1}{a_0(u,l)} K(z)

    where:

    .. math::

        z = \frac{x-X}{h} \qquad l = \frac{L-x}{h} \qquad u = \frac{U-x}{h}

    """

    name = 'renormalization'

    @numpy_trans1d_method()
    def pdf(self, points, out):
        if not self.bounded:
            return Cyclic.pdf(self, points, out)

        exog = self.exog
        points = points[..., np.newaxis]

        bw = self.bandwidth * self.adjust

        l = (points - self.lower) / bw
        u = (points - self.upper) / bw
        z = (points - exog) / bw

        kernel = self.kernel

        a1 = (kernel.cdf(l) - kernel.cdf(u))

        terms = kernel(z) * ((self.weights / bw) / a1)

        terms.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    @numpy_trans1d_method()
    def cdf(self, points, out):
        if not self.bounded:
            return super(Renormalization, self).cdf(points, out)
        return self.numeric_cdf(points, out)

    def grid(self, N=None, cut=None):
        if self.adjust.shape:
            return KDE1DMethod.grid(self, N, cut)
        if not self.bounded:
            return super(Renormalization, self).grid(N, cut)

        if cut is None:
            cut = self.kernel.cut
        N = self.grid_size(N)

        bw = self.bandwidth * self.adjust
        lower = self.lower
        upper = self.upper
        exog = self.exog
        if not finite(lower):
            lower = exog.min() - cut*self.bandwidth
        if not finite(upper):
            upper = exog.max() + cut*self.bandwidth
        R = upper - lower
        weights = self.weights
        if not weights.shape:
            weights = None
        kernel = self.kernel

        # Compute the FFT with enough margin to avoid side effects
        # here we assume that bw << est_R / 8 otherwise the FFT approximation is bad anyway
        shift_N = N // 8
        comp_N = N + N // 4
        comp_lower = lower - R / 8
        comp_upper = upper + R / 8

        mesh, density = fftdensity(exog, kernel.rfft, bw, comp_lower, comp_upper, comp_N, weights, self.total_weights)

        mesh = mesh.full()
        mesh = mesh[shift_N:shift_N+N]
        density = density[shift_N:shift_N+N]

        # Apply renormalization
        l = (mesh - lower) / bw
        u = (mesh - upper) / bw
        a1 = (kernel.cdf(l) - kernel.cdf(u))

        density /= a1

        return Grid(mesh, bounds=[lower, upper]), density

class _LinearCombinationKernel(Kernel1D):
    def __init__(self, ker):
        self._kernel = ker

    def pdf(self, x, out = None):
        out = self._kernel(x, out)
        out *= x
        return out

    __call__ = pdf

class LinearCombination(Unbounded):
    r"""
    This method uses the linear combination correction published in [1]_.

    The estimation is done with a modified kernel given by:

    .. math::

        \hat{K}(x;X,h,L,U) \triangleq \frac{a_2(l,u) - a_1(-u, -l) z}{a_2(l,u)a_0(l,u)
        - a_1(-u,-l)^2} K(z)

    where:

    .. math::

        z = \frac{x-X}{h} \qquad l = \frac{L-x}{h} \qquad u = \frac{U-x}{h}

    """

    name = 'linear combination'

    @numpy_trans1d_method()
    def pdf(self, points, out):
        if not self.bounded:
            return KDE1DMethod.pdf(self, points, out)

        exog = self.exog
        points = np.atleast_1d(points)[..., np.newaxis]

        bw = self.bandwidth * self.adjust

        l = (self.lower - points) / bw
        u = (self.upper - points) / bw
        z = (points - exog) / bw

        kernel = self.kernel

        a0 = kernel.cdf(u) - kernel.cdf(l)
        a1 = kernel.pm1(-l) - kernel.pm1(-u)
        a2 = kernel.pm2(u) - kernel.pm2(l)

        denom = a2 * a0 - a1 * a1
        upper = a2 - a1 * z

        upper /= denom
        upper *= (self.weights / bw) * kernel(z)

        upper.sum(axis=-1, out=out)
        out /= self.total_weights

        return out

    def cdf(self, points, out=None):
        if not self.bounded:
            return super(LinearCombination, self).cdf(points, out)
        return self.numeric_cdf(points, out)

    def grid(self, N=None, cut=None):
        if self.adjust.shape:
            return KDE1DMethod.grid(self, N, cut)
        if not self.bounded:
            return super(LinearCombination, self).grid(N, cut)

        if cut is None:
            cut = self.kernel.cut
        N = self.grid_size(N)

        bw = self.bandwidth * self.adjust
        lower = self.lower
        upper = self.upper
        exog = self.exog
        weights = self.weights
        if not weights.shape:
            weights = None
        kernel = self.kernel

        # Range on which the density is to be estimated
        est_lower = lower if finite(lower) else exog.min() - cut*self.bandwidth
        est_upper = upper if finite(upper) else exog.max() + cut*self.bandwidth
        est_R = est_upper - est_lower

        # Compute the FFT with enough margin to avoid side effects
        # here we assume that bw << est_R / 8 otherwise the FFT approximation is bad anyway
        shift_N = N // 8
        comp_N = N + N // 4
        comp_lower = est_lower - est_R / 8
        comp_upper = est_upper + est_R / 8
        total_weights = self.total_weights

        mesh, density = fftdensity(exog, kernel.rfft, bw, comp_lower, comp_upper, comp_N, weights, total_weights)
        _, z_density = fftdensity(exog, kernel.rfft_xfx, bw, comp_lower, comp_upper, comp_N, weights, total_weights)

        grid = mesh.full()
        grid = grid[shift_N:shift_N+N]
        density = density[shift_N:shift_N+N]
        z_density = z_density[shift_N:shift_N+N]

        # Apply linear combination approximation
        l = (lower - grid) / bw
        u = (upper - grid) / bw
        a0 = kernel.cdf(u) - kernel.cdf(l)
        a1 = kernel.pm1(-l) - kernel.pm1(-u)
        a2 = kernel.pm2(u) - kernel.pm2(l)

        density *= a2
        density -= a1*z_density
        density /= a2 * a0 - a1 * a1
        grid[0] = 0.

        return Grid(grid), density

Transform = namedtuple('Tranform', ['__call__', 'inv', 'Dinv'])

def _inverse(x, out=None):
    return np.divide(1, x, out)

LogTransform = Transform(np.log, np.exp, np.exp)
ExpTransform = Transform(np.exp, np.log, _inverse)


def transform_distribution(xs, ys, Dinv, out):
    r"""
    Transform a distribution into another one by a change a variable.

    Parameters
    ----------
    xs: ndarray
        Evaluation points of the distribution
    ys: ndarray
        Distribution value on the points xs
    Dinv: func
        Function evaluating the derivative of the inverse transformation  
        function
    out: ndarray
        Array in which to store the result

    Returns
    -------
    ndarray
        The variable ``out``, updated wih the transformed distribution

    Notes
    -----
    Given a random variable :math:`X` of distribution :math:`f_X`, the random
    variable :math:`Y = g(X)` has a distribution :math:`f_Y` given by:

    .. math::

        f_Y(y) = \left| \frac{1}{g'(g^{-1}(y))} \right| \cdot f_X(g^{-1}(y))

    """
    Dinv(xs, out=out)
    np.abs(out, out=out)
    _inverse(out, out=out)
    np.multiply(out, ys, out=out)
    return out


def create_transform(obj, inv=None, Dinv=None):
    """
    Create a transform object.

    :param fun obj: This can be either simple a function, or a function-object with an 'inv' and/or 'Dinv' attributes
        containing the inverse function and its derivative (respectively)
    :param fun inv: If provided, inverse of the main function
    :param fun Dinv: If provided, derivative of the inverse function
    :rtype: Transform
    :returns: A transform object with function, inverse and derivative of the inverse

    The inverse function must be provided, either as argument or as attribute to the object. The derivative of the 
    inverse will be estimated numerically if not provided.

    :Note: All the functions should accept an ``out`` argument to store the result.
    """
    if isinstance(obj, Transform):
        return obj
    fct = obj.__call__
    if inv is None:
        if not hasattr(obj, 'inv'):
            raise AttributeError("Error, transform object must have a 'inv' "
                                 "attribute or you must specify the 'inv' argument")
        inv = obj.inv
    if Dinv is None:
        if hasattr(obj, 'Dinv'):
            Dinv = obj.Dinv
        else:
            @numpy_trans1d()
            def Dinv(x):
                dx = x * 1e-9
                dx[x == 0] = np.min(dx[x != 0])
                return (inv(x + dx) - inv(x - dx)) / (2 * dx)
    return Transform(fct, inv, Dinv)

class _transKDE(object):
    def __init__(self, method):
        self.method = method.method

class TransformKDE(KDE1DMethod):
    r"""
    Compute the Kernel Density Estimate of a dataset, transforming it first to 
    a domain where distances are "more meaningful".

    Often, KDE is best estimated in a different domain. This object takes a KDE 
    object (or one compatible), and 
    a transformation function.

    Given a random variable :math:`X` of distribution :math:`f_X`, the random 
    variable :math:`Y = g(X)` has a distribution :math:`f_Y` given by:

    .. math::

        f_Y(y) = \left| \frac{1}{g'(g^{-1}(y))} \right| \cdot f_X(g^{-1}(y))

    In our term, :math:`Y` is the random variable the user is interested in, 
    and :math:`X` the random variable we can estimate using the KDE. In this 
    case, :math:`g` is the transform from :math:`Y` to :math:`X`.

    So to estimate the distribution on a set of points given in :math:`x`, we 
    need a total of three functions:

        - Direct function: transform from the original space to the one in 
          which the KDE will be perform (i.e. :math:`g^{-1}: y \mapsto x`)
        - Invert function: transform from the KDE space to the original one 
          (i.e. :math:`g: x \mapsto y`)
        - Derivative of the invert function

    If the derivative is not provided, it will be estimated numerically.

    :param trans: Either a simple function, or a function object with
        attributes `inv` and `Dinv` to use in case they are not provided as 
        arguments. The helper :py:func:`create_transform` will provide numeric 
        approximation of the derivative if required.
    :param method: instance of KDE1DMethod used in the transformed domain.
        Default is :py:class:`Reflection`
    :param inv: Invert of the function. If not provided, `trans` must have
        it as attribute.
    :param Dinv: Derivative of the invert function.

    :Note: all given functions should accept an optional ``out`` argument to
        get a pre-allocated array to store its result. Also the ``out`` 
        parameter may be one of the input argument.
    """
    def __init__(self, trans, method=None, inv=None, Dinv=None):
        super(TransformKDE, self).__init__()
        self.trans = create_transform(trans, inv, Dinv)
        if method is None:
            method = Reflection()
        self._method = method
        self._fitted = False
        self._clean_attrs()

    _to_clean = [ '_bw', '_cov', '_adjust'
                , '_weights', '_kernel', '_total_weights' ]
    def _clean_attrs(self):
        """
        Remove attributes not needed for this class
        """
        for attr in TransformKDE._to_clean:
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def method(self):
        """
        Method used in the transformed space.

        Notes
        -----
        The method can only be changed before fitting!
        """
        return self._method

    @method.setter
    def method(self, m):
        if self._fitted:
            raise ValueError("You cannot change the method of a fitted TransformKDE")
        self._method = m

    @property
    def exog(self):
        return self._exog

    @exog.setter
    def exog(self, val):
        val = np.atleast_1d(val).reshape(self._exog.shape)
        self.method.exog = self.trans(val)
        self._exog = val

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, val):
        val = float(val)
        trans_val = self.trans(val)
        self.method.lower = trans_val
        self._lower = val

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, val):
        val = float(val)
        trans_val = self.trans(val)
        self.method.upper = trans_val
        self._upper = val

    # List of attributes to forward to the method object
    _fwd_attrs = [ 'weights', 'adjust', 'kernel'
                 , 'bandwidth', 'covariance'
                 , 'total_weights', 'ndim', 'npts' ]

    def fit(self, kde):
        """
        Method called by the KDE object right after fitting to allow for 
        one-time calculation.

        This method copy, and transform, the various attributes of the KDE.

        :param pyqt_fit.self.KDE kde: KDE object being fitted
        """
        fitted = super(TransformKDE, self).fit(kde, False)
        fitted._clean_attrs()
        trans_kde = _transKDE(self)
        trans_kde.lower = self.trans(fitted.lower)
        trans_kde.upper = self.trans(fitted.upper)
        trans_kde.exog = self.trans(fitted.exog)

        copy_attrs = [ 'weights', 'adjust', 'kernel'
                     , 'bandwidth', 'covariance'
                     , 'total_weights', 'ndim', 'npts' ]

        for attr in copy_attrs:
            setattr(trans_kde, attr, getattr(kde, attr))

        trans_method = self.method.fit(trans_kde)
        fitted.method = trans_method
        fitted._fitted = True

        return fitted

    @numpy_trans1d_method()
    def pdf(self, points, out):
        trans = self.trans
        pts = trans(points)
        pdf = np.empty(points.shape, points.dtype)
        self.method(pts, out=pdf)
        return transform_distribution(pts, pdf, trans.Dinv, out=out)

    def grid(self, N=None, cut=None):
        xs, ys = self.method.grid(N, cut)
        trans = self.trans
        out = np.empty(ys.shape, ys.dtype)
        transform_distribution(xs.full(), ys, trans.Dinv, out=out)
        return xs.transform(trans.inv), out

    def cdf(self, points, out=None):
        return self.method.cdf(self.trans(points), out)

    def cdf_grid(self, N=None, cut=None):
        xs, ys = self.method.cdf_grid(N, cut)
        return xs.transform(self.trans.inv), ys

    def sf(self, points, out=None):
        return self.method.sf(self.trans(points), out)

    def sf_grid(self, kde, N=None, cut=None):
        xs, ys = self.method.sf_grid(N, cut)
        return xs.transform(self.trans.inv), ys

    def icdf(self, points, out=None):
        out = self.method.icdf(points, out)
        self.trans.inv(out, out=out)
        return out

    def icdf_grid(self, N=None, cut=None):
        xs, ys = self.method.icdf_grid(N, cut)
        self.trans.inv(ys, out=ys)
        return xs, ys

    def isf(self, points, out = None):
        out = self.method.isf(points, out)
        self.trans.inv(out, out=out)
        return out

    def isf_grid(self, N=None, cut=None):
        xs, ys = self.method.isf_grid(N, cut)
        self.trans.inv(ys, out=ys)
        return xs, ys

def _add_fwd_attr(cls, to_fwd, attr):
    try:
        fwd_obj = getattr(cls, to_fwd)
        doc = getattr(fwd_obj, '__doc__')
    except AttributeError:
        doc = 'Attribute forwarded to {}'.format(to_fwd)
    def getter(self):
        return getattr(getattr(self, to_fwd), attr)

    def setter(self):
        setattr(getattr(self, to_fwd), attr)

    setattr(cls, attr, property(getter, setter, doc=doc))

for attr in TransformKDE._fwd_attrs:
    _add_fwd_attr(TransformKDE, 'method', attr)

