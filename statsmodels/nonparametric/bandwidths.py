from __future__ import division, absolute_import, print_function
import numpy as np
from scipy import fftpack, optimize
from .kde_utils import large_float, finite
from statsmodels.compat.python import range

def _select_sigma(X):
    """
    Returns the smaller of std(X, ddof=1) or normalized IQR(X) over axis 0.

    References
    ----------
    Silverman (1986) p.47
    """
#    normalize = norm.ppf(.75) - norm.ppf(.25)
    normalize = 1.349
#    IQR = np.subtract.reduce(percentile(X, [75,25],
#                             axis=axis), axis=axis)/normalize
    IQR = (sap(X, 75) - sap(X, 25))/normalize
    return np.minimum(np.std(X, axis=0, ddof=1), IQR)


def variance_bandwidth(factor, exog):
    r"""
    Returns the covariance matrix:

    .. math::

        \mathcal{C} = \tau^2 cov(X)

    where :math:`\tau` is a correcting factor that depends on the method.
    """
    data_covariance = np.atleast_2d(np.cov(exog, rowvar=1, bias=False))
    sq_bandwidth = data_covariance * factor * factor
    return sq_bandwidth


def silverman_covariance(model):
    r"""
    The Silverman bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = \left( n \frac{d+2}{4} \right)^\frac{-1}{d+4}
    """
    exog = np.atleast_2d(model.exog)
    d, n = exog.shape
    return variance_bandwidth(0.9 * (n ** (-1. / (d + 4.))), exog)


def scotts_covariance(model=None):
    r"""
    The Scotts bandwidth is defined as a variance bandwidth with factor:

    .. math::

        \tau = n^\frac{-1}{d+4}
    """
    exog = np.atleast_2d(model.exog)
    d, n = exog.shape
    return variance_bandwidth((n * (d + 2.) / 4.) ** (-1. / (d + 4.)), exog)


def _botev_fixed_point(t, M, I, a2):
    l = 7
    I = large_float(I)
    M = large_float(M)
    a2 = large_float(a2)
    f = 2 * np.pi ** (2 * l) * np.sum(I ** l * a2 *
                                      np.exp(-I * np.pi ** 2 * t))
    for s in range(l, 1, -1):
        K0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
        f = 2 * np.pi ** (2 * s) * \
            np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
    return t - (2 * M * np.sqrt(np.pi) * f) ** (-2 / 5)



class botev_bandwidth(object):
    """
    Implementation of the KDE bandwidth selection method outline in:

    Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
    estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.

    Based on the implementation of Daniel B. Smith, PhD.

    The object is a callable returning the bandwidth for a 1D kernel.
    """
    def __init__(self, N=None, **kword):
        if 'lower' in kword or 'upper' in kword:
            print("Warning, using 'lower' and 'upper' for botev bandwidth is "
                  "deprecated. Argument is ignored")
        self.N = N

    def __call__(self, model):
        """
        Returns the optimal bandwidth based on the data
        """
        data = model.exog
        N = 2 ** 10 if self.N is None else int(2 ** np.ceil(np.log2(self.N)))
        lower = getattr(model, 'lower', None)
        upper = getattr(model, 'upper', None)
        if not finite(lower) or not finite(upper):
            minimum = np.min(data)
            maximum = np.max(data)
            span = maximum - minimum
            lower = minimum - span / 10 if not finite(lower) else lower
            upper = maximum + span / 10 if not finite(upper) else upper
        # Range of the data
        span = upper - lower

        # Histogram of the data to get a crude approximation of the density
        weights = model.weights
        if not weights.shape:
            weights = None
        M = len(data)
        DataHist, bins = np.histogram(data, bins=N, range=(lower, upper), weights=weights)
        DataHist = DataHist / M
        DCTData = fftpack.dct(DataHist, norm=None)

        I = np.arange(1, N, dtype=int) ** 2
        SqDCTData = (DCTData[1:] / 2) ** 2
        guess = 0.1

        try:
            t_star = optimize.brentq(_botev_fixed_point, 0, guess,
                                     args=(M, I, SqDCTData))
        except ValueError:
            t_star = .28 * N ** (-.4)

        return np.sqrt(t_star) * span

