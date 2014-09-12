"""
This module contains kernels for non-continuous data.

Unlike with continuous kernels, these ones require explicitely the evaluation point and the bandwidth.
"""

import numpy as np

class AitchisonAitken(object):
    """
    The Aitchison-Aitken kernel, used for unordered discrete random variables.

    See p.18 of [2]_ for details.  The value of the kernel L if :math:`X_{i}=x`
    is :math:`1-\lambda`, otherwise it is :math:`\frac{\lambda}{c-1}`.
    Here :math:`c` is the number of levels plus one of the RV.

    References
    ----------
    .. [1] J. Aitchison and C.G.G. Aitken, "Multivariate binary discrimination
           by the kernel method", Biometrika, vol. 63, pp. 413-420, 1976.
    .. [2] Racine, Jeff. "Nonparametric Econometrics: A Primer," Foundation
           and Trends in Econometrics: Vol 3: No 1, pp1-88., 2008.
    """
    def for_ndim(self, ndim):
        assert ndim == 1, "Error, this jkernel only works in 1D"
        return self

    def pdf(self, x, Xi, bw, levels, out=None):
        """
        Compute the PDF on the points x

        Parameters
        ----------
        x: int or ndarray
            Points to evaluated the PDF at
        Xi: ndarray
            Training dataset
        bw: float
            Bandwidth
        levels: ndarray
            List of valid levels in the set
        """
        x = np.asarray(x)
        bw = float(bw)
        num_levels = len(levels)
        Xi = np.atleast_1d(Xi)
        if out is None:
            out = np.ones_like(Xi)
        else:
            out[...] = 1
        out *= bw / (num_levels-1)
        idx = Xi == x
        out[idx] = (idx*(1-bw))[idx]
        return out
