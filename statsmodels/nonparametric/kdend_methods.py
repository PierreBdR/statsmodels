r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains a set of methods to compute multivariates KDEs.
"""

import numpy as np
from scipy import linalg
from statsmodels.compat.python import range

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
    cut = dot(kde.bandwidth, cut * np.ones(kde.ndim, dtype=float))
    lower = np.array(kde.lower)
    upper = np.array(kde.upper)
    ndim = kde.ndim
    for i in range(ndim):
        if lower[i] == -np.inf:
            lower[i] = np.min(kde.exog[i]) - cut[i]
        if upper[i] == np.inf:
            upper[i] = np.max(kde.exog[i]) + cut[i]
    xi = [ np.linspace(lower[i], upper[i], N) for i in range(ndim) ]
    return np.meshgrid(*xi)


def compute_bandwidth(kde):
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
        bw = np.atleast_2d(bw)
        if bw.shape == (1,1):
            bw = bw[0,0] * np.identity(n)
        assert bw.shape == (n,n)
        return bw, np.dot(bw, bw)
    elif kde.covariance is not None:
        if callable(kde.covariance):
            cov = kde.covariance(kde)
        else:
            cov = kde.covariance
        cov = np.atleast_2d(cov)
        if cov.shape == (1,1):
            cov = cov[0,0] * np.identity(n)
        assert cov.shape == (n,n)
        return linalg.sqrtm(cov), cov
    raise ValueError("Bandwidth or covariance needs to be specified")

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
        npts = kde.npts
        fitted = self.copy()
        if compute_bandwidth:
            bw, cov = compute_bandwidth(kde)
            assert bw.shape == (ndim, ndim)
            assert cov.shape == (ndim, ndim)
            fitted._bw = bw
            fitted._cov = cov
            fitted._inv_bw = linalg.inv(bw)
        assert kde.exog.shape == (ndim, npts)
        fitted._exog = kde.exog
        assert kde.upper.shape == (ndim,)
        fitted._upper = kde.upper
        assert len(kde.lower) == (ndim,)
        fitted._lower = kde.lower
        fitted._kernel = kde.kernel.for_ndim(ndim)
        assert kde.weights.shape == (npts,) or kde.weights.shape == ()
        fitted._weights = kde.weights
        assert kde.adjust.shape == (npts,) or kde.adjust.shape == ()
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
        return self._exog.shape[0]

    @property
    def npts(self):
        """
        Number of points in the setup
        """
        return self._exog.shape[1]

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
        val = np.atleast_2d(val)
        if val.shape == (1,1):
            val.shape = ()
            cov = val*val
            inv_bw = 1/val
            det_inv_bw = inv_bw
        elif val.shape[0] == 1:
            val.shape = (val.shape[1],)
            assert val.shape[0] == self.ndim
            cov = val * val
            inv_bw = 1 / bw
            det_inv_bw = np.product(inv_bw)
        else:
            assert val.shape == (self.ndim, self.ndim)
            cov = dot(val, val)
            inv_bw = linalg.inv(val)
            det_inv_bw = linalg.det(inv_bw)
        self._bw = val
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
    def covariance(self, val):
        val = np.atleast_2d(val)
        if val.shape == (1,1):
            val = val[0,0] * np.identity(self.ndim)
        assert val.shape == (self.ndim, self.ndim)
        bw = sqrtm(val)
        self.bandwidth = bw
        self._cov = val

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
        value = np.atleast_2d(value)
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
            self._total_weights = float(np.npts)

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

    def pdf(self, points, out=None):
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
        points = np.atleast_2d(points)
        exog = self.exog

        d, m = points.shape
        assert d == self.ndim

        kernel = self.kernel
        inv_bw = self.inv_bandwidth
        det_inv_bw = self.det_inv_bandwidth
        weights = self.weights
        adjust = self.adjust

        if out is None:
            out = np.zeros(m, dtype=float)
        else:
            out.setfield(0., dtype=float)

        factor = (weight * det_inv_bw) / adjust
        if self.npts > m:
            # There are fewer points that data: loop over points
            for i in range(m):
                diff = dot(inv_bw, exog - points[:, i, np.newaxis]) / adjust)
                energy = kernel(diff)
                energy *= factor
                out[i] = sum(energy)
        else:
            # There are fewer data that points: loop over data
            _, factor, adjust = np.broadcast_arrays(exog[0], factor, adjust)
            for b in range(self.npts):
                diff = dot(inv_bw, exog[:, i, newaxis] - points) / adjust[i]
                out += factor[i] * kernel(diff)

        out /= self.total_weights

        return out

    def __call__(self, points, out=None):
        """
        Call the :py:meth:`pdf` method.
        """
        return self.pdf(points, out)


