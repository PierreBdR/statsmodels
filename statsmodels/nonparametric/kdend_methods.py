r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains a set of methods to compute multivariates KDEs.
"""

import numpy as np
from scipy import linalg
from statsmodels.compat.python import range
from .kde_utils import numpy_trans_method, atleast_2df, Grid
from numpy import newaxis
from . import kde1d_methods
from copy import copy as shallow_copy
from numpy import s_
from .fast_linbin import fast_linbin_nd as fast_bin_nd

def generate_grid(kde, N=None, cut=None):
    r"""
    Helper method returning a regular grid on the domain of the KDE.

    Parameters
    ----------
    kde: KDE1DMethod
        Fitted KDE object
    N: int or list of int
        Number of points in the grid
    cut: float
        For unbounded domains, how far past the maximum should
        the grid extend to, in term of KDE bandwidth

    Returns
    -------
    A vector of N regularly spaced points
    """
    N = np.asarray(kde.grid_size(N), dtype=int)
    if N.ndim == 0:
        N = N * np.ones(kde.ndim, dtype=int)
    elif N.ndim != 1 or N.shape[0] != kde.ndim:
        raise ValueError("N must be a single integer, or a 1D array with as many element as dimensions in the KDE")
    if cut is None:
        cut = kde.kernel.cut
    if kde.bandwidth.ndim == 0:
        cut = kde.bandwidth * cut * np.ones(kde.ndim, dtype=float)
    elif kde.bandwidth.ndim == 1:
        cut = kde.bandwidth * cut
    else:
        cut = np.dot(kde.bandwidth, cut * np.ones(kde.ndim, dtype=float))
    lower = np.array(kde.lower)
    upper = np.array(kde.upper)
    ndim = kde.ndim
    for i in range(ndim):
        if lower[i] == -np.inf:
            lower[i] = np.min(kde.exog[:,i]) - cut[i]
        if upper[i] == np.inf:
            upper[i] = np.max(kde.exog[:,i]) + cut[i]
    xi = tuple(s_[lower[i] : upper[i] : N[i]*1j] for i in range(ndim))
    gr = np.mgrid[xi]
    return np.concatenate([g[...,None] for g in gr], axis=-1)

def _compute_bandwidth(kde):
    """
    Compute the bandwidth and covariance for the estimated model, based of its 
    exog attribute
    """
    n = kde.ndim
    if kde.bandwidth is not None:
        if callable(kde.bandwidth):
            bw = kde.bandwidth(kde)
        else:
            bw = kde.bandwidth
        return bw, None
    elif kde.covariance is not None:
        if callable(kde.covariance):
            cov = kde.covariance(kde)
        else:
            cov = kde.covariance
        return None, cov
    return None, None

def fftdensity(exog, kernel_rfft, bw_inv, lower, upper, N, weights, total_weights):
    """
    Compute the density estimate using a FFT approximation.

    Parameters
    ----------
    exog: ndarray
        2D array with the data to fit
    kernel_rfft: function
        Function computing the rFFT for the kernel
    lower: float
        Lower bound on which to compute the density
    upper: float
        Upper bound on which to compute the density
    N: int or list of int
        Number of buckets to compute, for each dimension
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
    DataHist, mesh = fast_bin_nd(exog, np.c_[lower, upper], N, weights=weights, bin_types='C')
    DataHist = DataHist / total_weights
    FFTData = np.fft.rfftn(DataHist)
    ndim = exog.shape[1]

    dx = np.asarray(mesh.interval)
    if bw_inv.ndim == 2:
        dx = dot(dx, bw_inv)
    else:
        dx *= bw_inv

    smth = kernel_rfft(DataHist.shape, dx)

    SmoothFFTData = FFTData * smth
    volume = np.prod(mesh.interval)
    density = np.fft.irfftn(SmoothFFTData, DataHist.shape) / volume
    return mesh, density


class KDEnDMethod(object):
    """
    Base class providing a default grid method and a default method for unbounded evaluation of the PDF and CDF. It also 
    provides default methods for the other metrics, based on PDF and CDF calculations.

    The default class can only deal with open, continuous, multivariate data.

    :Note:
        - It is expected that all grid methods will return the same grid if 
          used with the same arguments.
        - It is fair to assume all array-like arguments will be at least 2D arrays, with the first dimension denoting 
          the dimension.


    Attributes
    ----------
    base_p2: int
        Log2 of the number of points wanted in a grid.
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
        self._inv_bw = None
        self._det_inv_bw = None
        self._cov = None
        self.base_p2 = 10

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
        ndim = kde.ndim
        if ndim == 1 and type(self) == KDEnDMethod:
            method = kde1d_methods.Reflection()
            return method.fit(kde, compute_bandwidth)
        npts = kde.npts
        fitted = self.copy()
        fitted._exog = kde.exog
        assert kde.upper.shape == (ndim,)
        fitted._upper = kde.upper
        assert kde.lower.shape == (ndim,)
        fitted._lower = kde.lower
        fitted._kernel = kde.kernel.for_ndim(ndim)
        assert kde.weights.ndim == 0 or kde.weights.shape == (npts,)
        fitted._weights = kde.weights
        assert kde.adjust.ndim == 0 or kde.adjust.shape == (npts,)
        fitted._adjust = kde.adjust
        fitted._total_weights = kde.total_weights
        if compute_bandwidth:
            bw, cov = _compute_bandwidth(kde)
            if bw is not None:
                fitted.bandwidth = bw
            elif cov is not None:
                fitted.covariance = cov
            else:
                raise ValueError("Error, no bandwidth or covariance have been specified")
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
        return self._exog.shape[1]

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
    def bandwidth(self, bw):
        bw = np.asarray(bw).squeeze()
        if bw.ndim == 0:
            cov = bw*bw
            inv_bw = 1/bw
            det_inv_bw = inv_bw
        elif bw.ndim == 1:
            assert bw.shape[0] == self.ndim
            cov = bw * bw
            inv_bw = 1 / bw
            det_inv_bw = np.product(inv_bw)
        elif bw.ndim == 2:
            assert bw.shape == (self.ndim, self.ndim)
            cov = np.dot(bw, bw)
            inv_bw = linalg.inv(bw)
            det_inv_bw = linalg.det(inv_bw)
        else:
            raise ValueError("Error, specified bandiwdth has more than 2 dimension")
        self._bw = bw
        self._cov = cov
        self._inv_bw = inv_bw
        self._det_inv_bw = det_inv_bw

    @property
    def inv_bandwidth(self):
        """
        Inverse of the selected bandwidth
        """
        return self._inv_bw

    @property
    def det_inv_bandwidth(self):
        """
        Inverse of the selected bandwidth
        """
        return self._det_inv_bw

    @property
    def covariance(self):
        """
        Square of the selected bandwidth

        Unlike the covariance for the KDE, this must be an actual value and not 
        a method.
        """
        return self._cov

    @covariance.setter
    def covariance(self, cov):
        cov = np.asarray(cov)
        if cov.ndim == 0:
            bw = np.sqrt(cov)
        elif cov.ndim == 1:
            assert cov.shape[0] == self.ndim
            bw = np.sqrt(cov)
        elif cov.ndim == 2:
            assert cov.shape == (self.ndim, self.ndim)
            bw = sqrtm(cov)
        else:
            raise ValueError("Error, specified covariance has more than 2 dimension")
        self.bandwidth = bw
        self._cov = cov

    def update_inputs(self, exog, weights=1., adjust=1.):
        exog = atleast_2df(exog)
        if exog.ndim != 2:
            raise ValueError("Error, exog must be at most a 2D array")
        weights = np.asarray(weights)
        adjust = np.asarray(adjust)
        if weights.ndim != 0 and weights.shape != (exog.shape[0],):
            raise ValueError("Error, weights must be either a single number, or a 1D array with the same length as exog")
        if adjust.ndim != 0 and adjust.shape != (exog.shape[0],):
            raise ValueError("Error, adjust must be either a single number, or a 1D array with the same length as exog")
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
        value = atleast_2df(value)
        assert value.shape == self._exog.shape
        self._exog = value

    @property
    def lower(self):
        """
        Lower bound of the problem domain
        """
        return self._lower

    @lower.setter
    def lower(self, val):
        val = np.atleast_1d(val)
        assert val.shape == (self.ndim,)
        self._lower = val

    @property
    def upper(self):
        """
        Upper bound of the problem domain
        """
        return self._upper

    @upper.setter
    def upper(self, val):
        val = np.atleast_1d(val)
        assert val.shape == (self.ndim,)
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
            assert val.shape == (self.npts,)
            self._weights = val
            self._total_weights = np.sum(val)
        else:
            self._weights = np.asarray(1.)
            self._total_weights = float(self.npts)

    @property
    def total_weights(self):
        """
        Sum of the point weights
        """
        return self._total_weights

    def closed(self, dim = None):
        """
        Returns true if the density domain is closed (i.e. lower and upper
        are both finite)

        Parameters
        ----------
        dim: int
            Dimension to test. If None, test of all dimensions are closed
        """
        if dim is None:
            return all(self.closed(i) for i in range(self.ndim))
        return self.lower[dim] > -np.inf and self.upper[dim] < np.inf

    def bounded(self, dim = None):
        """
        Returns true if the density domain is actually bounded

        Parameters
        ----------
        dim: int
            Dimension to test. If None, test of all dimensions are bounded
        """
        if dim is None:
            return all(self.bounded(i) for i in range(self.ndim))
        return self.lower[dim] > -np.inf or self.upper[dim] < np.inf

    @numpy_trans_method('ndim', 1)
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

        m, d = points.shape
        assert d == self.ndim

        kernel = self.kernel
        inv_bw = self.inv_bandwidth
        def scalar_inv_bw(pts):
            return (pts * inv_bw)
        def matrix_inv_bw(pts):
            return np.dot(pts, inv_bw)
        inv_bw_fct = scalar_inv_bw
        if inv_bw.ndim == 2:
            inv_bw_fct = matrix_inv_bw


        #if inv_bw.ndim == 2:
            #raise ValueError("Error, this method cannot handle non-diagonal bandwidth matrix.")
        det_inv_bw = self.det_inv_bandwidth
        weights = self.weights
        adjust = self.adjust

        if self.npts > m:
            factor = weights * det_inv_bw / adjust
            # There are fewer points that data: loop over points
            energy = np.empty((exog.shape[0],), dtype=out.dtype)
            #print("iterate on points")
            for idx in range(m):
                diff = inv_bw_fct(points[idx] - exog)
                kernel.pdf(diff, out=energy)
                energy *= factor
                out[idx] = np.sum(energy)
        else:
            weights = np.atleast_1d(weights)
            adjust = np.atleast_1d(adjust)
            out[...] = 0

            # There are fewer data that points: loop over data
            dw = 1 if weights.shape[0] > 1 else 0
            da = 1 if adjust.shape[0] > 1 else 0
            na = 0
            nw = 0
            n = self.npts
            energy = np.empty((points.shape[0],), dtype=out.dtype)
            #print("iterate on exog")
            for idx in range(n):
                diff = inv_bw_fct(points - exog[idx])
                kernel.pdf(diff, out=energy)
                energy *= weights[nw] / adjust[na]
                out += energy
                # Iteration for weights and adjust
                na += da
                nw += dw
            out *= det_inv_bw

        out /= self.total_weights
        return out

    def __call__(self, points, out=None):
        """
        Call the :py:meth:`pdf` method.
        """
        return self.pdf(points, out)

    def grid(self, N=None, cut=None):
        """
        Compute the PDF on a grid.

        Parameters
        ----------
        N: int or list of int
            Number of elements on the grid, per dimension
        """
        gr = generate_grid(self, N, cut)
        return gr, self.pdf(gr)

    @numpy_trans_method('ndim', 1)
    def cdf(self, points, out):
        bw = self.bandwidth
        if bw.ndim < 2: # We have a diagonal matrix
            exog = self.exog
        else:
            pass

    def grid_size(self, N = None):
        if N is None:
            p2 = self.base_p2 // self.ndim
            if self.base_p2 % self.ndim > 0:
                p2 += 1
            return 2**p2
        return N

class Cyclic(KDEnDMethod):
    def fit(self, kde, compute_bandwidth = True):
        if kde.ndim == 1:
            cyc = kde1d_methods.Cyclic()
            return cyc.fit(kde, compute_bandwidth)
        return super(Cyclic, self).fit(kde, compute_bandwidth)

    @numpy_trans_method('ndim', 1)
    def pdf(self, points, out):
        for i in range(self.ndim):
            if self.bounded(i) and not self.closed(i):
                raise ValueError("Error, cyclic method requires all dimensions to be closed or not bounded")
        if not self.bounded():
            return super(Cyclic, self).pdf(points, out)
        raise NotImplemented

    def grid(self, N=None, cut=None):
        if self.adjust.shape:
            return KDEnDMethod.grid(self, N, cut)
        for i in range(self.ndim):
            if self.bounded(i) and not self.closed(i):
                raise ValueError("Error, cyclic method requires all dimensions to be closed or not bounded")
        bw_inv = self.inv_bandwidth / self.adjust
        exog = self.exog
        N = self.grid_size(N)

        lower = self.lower
        upper = self.upper

        if cut is None:
            cut = self.kernel.cut
        if self.bandwidth.ndim == 0:
            cut = self.bandwidth * cut * np.ones(self.ndim, dtype=float)
        elif self.bandwidth.ndim == 1:
            cut = self.bandwidth * cut
        else:
            cut = np.dot(self.bandwidth, cut * np.ones(self.ndim, dtype=float))

        for d in range(self.ndim):
            if upper[d] == np.inf:
                if cut is None:
                    cut = self.kernel.cut
                lower[d] = np.min(exog[:,d]) - cut[d]
                upper[d] = np.max(exog[:,d]) + cut[d]

        weights = self.weights
        if not weights.shape:
            weights = None

        return fftdensity(exog, self.kernel.rfft, bw_inv, lower, upper, N, weights, self.total_weights)
