import numpy as np
from . import _fast_linbin
from .grid import Grid

def _fast_bin(fct, X, bounds, M, weights, bin_type, out):
    X = np.atleast_1d(X).astype(float)
    if X.ndim != 1:
        raise ValueError("Error, X must be a 1D array")
    if not bin_type in ('c', 'r', 'b', 'n'):
        raise ValueError("Error, bin_type must be one of 'c', 'r', 'b' or 'n'")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
            raise ValueError("Weights must be None or an array of the same shape as X")
    else:
        weights = np.empty((0,), dtype=float)
    M = int(M)
    if out is None:
        out = np.zeros((M,), dtype=float)
    elif out.shape != (M,) or out.dtype != np.dtype(np.float):
        raise ValueError("Error, out array must be a 1D float array with length M")
    a, b = bounds
    a = float(a)
    b = float(b)
    mesh, bounds = fct(X, a, b, out, weights, bin_type)

    return Grid(mesh, bounds, bin_type), out

def fast_linbin(X, bounds, M, weights = None, bin_type = 'b', out = None):
    r"""
    Linear Binning as described in Fan and Marron (1994)

    Parameters
    ----------
    X: ndarray
        Input data
    bounds: (float, float)
        Bounds of the grid
    M: int
        Number of bins
    weights: ndarray
        Array of same size as X with weights for each point, or None if all weights are 1
    bin_type: str
        Type of bin as a string (see Notes)

    Returns
    -------
    mesh: Grid
        Grid of the points
    values: ndarray
        Values for each bin in the grid

    Notes
    -----

    The bin can be:
        - Bounded ('b')
        - Reflected ('r')
        - Cyclic ('c')
        - Non-Continuous ('n')

    Unless the bin is non-continuous, for a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions 
    :math:`p_i` and :math:`p_{i+1}`, the bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    The placement of the bin depends on the bin type:
    - For reflected or un-bounded bins, the bins are placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, 
      \ldots b-\delta/2\}` with :math:`delta = \frac{M-1}{b-a}`.
    - For cyclic bins, the bins are placed at :math:`\{a,\ldots,\}


    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    return _fast_bin(_fast_linbin.fast_linbin, X, bounds, M, weights, bin_type, out)

def fast_bin(X, bounds, M, weights = None, bin_type='b'):
    """
    Fast binning in 1D

    Parameters
    ----------
    X: ndarray
        Array of values to bin
    bounds: (float, float)
        Bounds for the bins
    M: int
        Number of bins
    weights: ndarray
        If not None, an array of same size as X for the weights of each point
    bin_types: str
        Ignore: provided for compatibility with the linbin functions

    Returns
    -------
    mesh: Grid
        Grid of the points
    values: ndarray
        Values for each bin in the grid
    """
    return _fast_bin(_fast_linbin.fast_bin, X, bounds, M, weights, bin_type, out)

def _fast_bin_nd(fct, X, bounds, M, weights, bin_types, out):
    M = np.asarray(M).astype(int)
    X = np.atleast_2d(X).astype(np.float)
    bounds = np.atleast_2d(bounds).astype(float)

    D = X.shape[1]
    max_d = _fast_linbin.MAX_DIMENSION
    if D > max_d:
        raise ValueError("Error, you cannot have more than {0} dimensions, and you have {1}".format(max_d, D))

    n = X.shape[0]
    if M.ndim == 0:
        M = M*np.ones((D,), dtype=int)
    if np.any(M<2):
        raise ValueError("You need to specify at least 2 elements per dimension")
    if (bounds.ndim != 2 or bounds.shape[0] != D or bounds.shape[1] != 2 or
        M.ndim != 1 or M.shape[0] != D):
        raise ValueError("Error, incompatible dimensions for bounds, M and X")
    tM = tuple(M)
    if weights is None:
        weights = np.empty((0,), dtype=float)
    else:
        weights = np.asarray(weights).astype(float)
        if weights.shape != (n,):
            raise ValueError("weights must be None or a 1D array of same length as X")
    if out is None:
        out = np.zeros(tM, dtype=np.float)
    elif out.shape != tM or out.dtype != np.dtype(np.float):
        raise ValueError("Output array is either of the wrong size or the wrong type")

    if len(bin_types) == 1:
        bin_types = bin_types * D
    elif len(bin_types) != D:
        raise ValueError("Error, bin_types must be a string of length 1 or D (e.g. the number of dimensions)")

    mesh, bounds = fct(X, bounds[:,0], bounds[:,1], out, weights, bin_types)
    return Grid(mesh, bounds, bin_types), out

def fast_linbin_nd(X, bounds, M, weights=None, bin_types='b', out=None):
    r"""
    Linear Binning in nD as described in Fan and Marron (1994)

    Parameters
    ----------
    X: ndarray
        Input data
    bounds: ndarray
        2D array (D,2) describing for each dimension the bounds
    M: ndarray
        Number of bins for each dimension. If this is only one value, all axis will have the same number of bins.
    weights: ndarray
        Array of same size as X with weights for each point, or None if all weights are 1
    bin_types: str
        Type of bin as a string (see Notes)

    Returns
    -------
    mesh: Grid
        Grid of the points
    values: ndarray
        Values for each bin in the grid

    Notes
    -----

    The bin can be:
        - Bounded ('b')
        - Reflected ('r')
        - Cyclic ('c')
        - Non-Continuous ('n')

    Unless the bin is non-continuous, for a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions 
    :math:`p_i` and :math:`p_{i+1}`, the bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    The placement of the bin depends on the bin type:
    - For reflected or un-bounded bins, the bins are placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, 
      \ldots b-\delta/2\}` with :math:`delta = \frac{M-1}{b-a}`.
    - For cyclic bins, the bins are placed at :math:`\{a,\ldots,\}


    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    return _fast_bin_nd(_fast_linbin.fast_linbin_nd, X, bounds, M, weights, bin_types, out)


def fast_bin_nd(X, bounds, M, weights=None, bin_types='b', out=None):
    r"""
    Simple Binning in nD

    Parameters
    ----------
    X: ndarray
        Input data
    bounds: ndarray
        2D array (D,2) describing for each dimension the bounds
    M: ndarray
        Number of bins for each dimension. If this is only one value, all axis will have the same number of bins.
    weights: ndarray
        Array of same size as X with weights for each point, or None if all weights are 1
    bin_types: str
        Bin types

    Returns
    -------
    mesh: Grid
        Grid of the points
    values: ndarray
        Values for each bin in the grid
    """
    return _fast_bin_nd(_fast_linbin.fast_bin_nd, X, bounds, M, weights, bin_types, out)

