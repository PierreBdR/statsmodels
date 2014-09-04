"""
cython -a fast_linbin.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o fast_linbin.so fast_linbin.c
"""

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport floor
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def _fast_linbin(np.ndarray[DOUBLE] X, double a, double b, int M, np.ndarray[DOUBLE] weights = None, int cyclic=0):
    r"""
    Linear Binning as described in Fan and Marron (1994)

    :param X ndarray: Input data
    :param a float: Lowest value to consider
    :param b float: Highest valus to consider
    :param M int: Number of bins
    :param weights ndarray: Array of same size as X with weights for each point, or None if all weights are 1
    :param cyclic bool: Consider the data cyclic or not

    :Returns: The weights in each bin

    For a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions :math:`p_i` and :math:`p_{i+1}`, the 
    bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    By default the bins will be placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, \ldots b-\delta/2\}` with 
    :math:`delta = \frac{M-1}{b-a}`.

    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    cdef:
        Py_ssize_t i
        int nobs = X.shape[0]
        np.ndarray[DOUBLE] gcnts = np.zeros(M, np.float)
        np.ndarray[DOUBLE] mesh
        double delta = (b - a) / M
        double inv_delta = 1 / delta
        double shift
        double rem
        double val
        double lower
        double upper
        double w
        int base_idx
        int N
        int has_weight = weights is not None

    if cyclic:
        shift = -a
        lower = 0
        upper = M
    else:
        shift = -a-delta/2
        lower = -0.5
        upper = M-0.5

    for i in range(nobs):
        val = (X[i] + shift) * inv_delta
        if val >= lower and val <= upper:
            base_idx = <int> floor(val);
            if not cyclic and val < 0:
                rem = 1
            elif not cyclic and val > M-1:
                rem = 0
            else:
                rem = val - base_idx
            if has_weight:
                w = weights[i]
            else:
                w = 1.
            if base_idx == M: # Only possible if cyclic
                gcnts[0] += w
            else:
                if base_idx >= 0:
                    gcnts[base_idx] += (1 - rem)*w
                if base_idx < M-1:
                    gcnts[base_idx+1] += rem*w
                elif base_idx == M-1:
                    gcnts[0] += rem*w

    if cyclic:
        mesh = np.linspace(a, b-delta, M)
    else:
        mesh = np.linspace(a+delta/2, b-delta/2, M)

    return gcnts, mesh

def fast_linbin(object X, double a, double b, int M, object weights = None, int cyclic=0):
    cdef:
        np.ndarray[DOUBLE] mX
    mX = np.atleast_1d(X).astype(float)
    if mX.ndim != 1:
        raise ValueError("Error, X must be a 1D array")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != mX.shape[0]:
            raise ValueError("Weights must be None or an array of the same shape as X")
    return _fast_linbin(mX, a, b, M, weights, cyclic)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def fast_bin(np.ndarray[DOUBLE] X, double a, double b, int M, np.ndarray[DOUBLE] weights = None, int cyclic=0):
    r"""
    Linear Binning as described in Fan and Marron (1994)

    :param X ndarray: Input data
    :param a float: Lowest value to consider
    :param b float: Highest valus to consider
    :param M int: Number of bins
    :param weights ndarray: Array of same size as X with weights for each point, or None if all weights are 1
    :param cyclic bool: Consider the data cyclic or not

    :Returns: The weights in each bin

    For a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions :math:`p_i` and :math:`p_{i+1}`, the 
    bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    By default the bins will be placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, \ldots b-\delta/2\}` with 
    :math:`delta = \frac{M-1}{b-a}`.

    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    cdef:
        Py_ssize_t i
        int nobs = X.shape[0]
        np.ndarray[DOUBLE] gcnts = np.zeros(M, np.float)
        np.ndarray[DOUBLE] mesh
        double delta = (b - a) / M
        double inv_delta = 1 / delta
        double shift
        double rem
        double val
        double lower
        double upper
        double w
        int base_idx
        int N
        int has_weight = weights is not None

    if has_weight:
        assert weights.shape[0] == X.shape[0], "Error, the weights must be None or an array of same size as X"

    shift = -a
    lower = 0
    upper = M

    for i in range(nobs):
        val = (X[i] + shift) * inv_delta
        if val >= lower and val <= upper:
            base_idx = <int> floor(val);
            if has_weight:
                w = weights[i]
            else:
                w = 1.
            if base_idx == M:
                base_idx -= 1
            gcnts[base_idx] += w

    mesh = np.linspace(a+delta/2, b-delta/2, M)
    return gcnts, mesh


# Note: this define is NOT the limiting factor in the algorithm. See the code for details.
# Ideally, this constant should be the number of bits in Py_ssize_t
DEF MAX_DIM = 64

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
def _fast_linbin_nd(np.ndarray[DOUBLE, ndim=2] X,
                    np.ndarray[DOUBLE] a, np.ndarray[DOUBLE] b,
                    np.ndarray[INT] M,
                    np.ndarray[DOUBLE] weights,
                    int has_weight, int cyclic,
                    np.ndarray[DOUBLE] gcnts, np.ndarray[DOUBLE] delta):
    cdef:
        Py_ssize_t D = a.shape[0]
        np.npy_intp nD = D
        Py_ssize_t i, d, c, N
        int nobs = X.shape[0]
        object mesh
        double shift[MAX_DIM]
        double rem[MAX_DIM]
        double val[MAX_DIM]
        double lower[MAX_DIM]
        double upper[MAX_DIM]
        double w
        int base_idx[MAX_DIM]
        int next_idx[MAX_DIM]
        int is_in
        Py_ssize_t nb_corner = 1 << D
        double wc
        Py_ssize_t pos

    for d in range(D):
        delta[d] = (b[d] - a[d]) / M[d]
        if cyclic:
            shift[d] = -a[d]
            lower[d] = 0
            upper[d] = <double> M[d]
        else:
            shift[d] = -a[d]-delta[d]/2
            lower[d] = -0.5
            upper[d] = <double> M[d]-0.5

    for i in range(nobs):
        is_in = 1
        for d in range(D):
            val[d] = (X[i,d] + shift[d]) / delta[d]
            if val[d] < lower[d] or val[d] > upper[d]:
                is_in = 0
                break
        if is_in:
            if has_weight:
                w = weights[i]
            else:
                w = 1.

            for d in range(D):
                base_idx[d] = <int> floor(val[d])
                if not cyclic and val[d] < 0:
                    rem[d] = 1
                elif not cyclic and val[d] > M[d]-1:
                    rem[d] = 0
                else:
                    rem[d] = val[d] - base_idx[d]

                if base_idx[d] == M[d]: # Only possible if cyclic
                    base_idx[d] = 0
                    next_idx[d] = 1
                elif base_idx[d] == M[d]-1:
                    next_idx[d] = 0
                else:
                    next_idx[d] = base_idx[d]+1

            # This uses the binary representation of the corner id (from 0 to 2**d-1) to identify where it is
            # for each bit: 0 means lower index, 1 means upper index
            # This means we are limited by the number of bits in Py_ssize_t. But also that we couldn't possibly allocate 
            # an array too big for this to work.
            for c in range(nb_corner):
                wc = w
                pos = 0
                for d in range(D):
                    pos *= M[d]
                    if c & 1:
                        wc *= 1-rem[d]
                        pos += base_idx[d]
                    else:
                        wc *= rem[d]
                        pos += next_idx[d]
                    c >>= 1
                gcnts[pos] += wc


@cython.embedsignature(True)
@cython.profile(True)
def fast_linbin_nd(object X, object a, object b, object M, object weights = None, int cyclic=0):
    r"""
    Linear Binning as described in Fan and Marron (1994)

    :param X ndarray: Input data, (N,D) array with D the dimension and N the number of points
    :param a float: Lowest value to consider
    :param b float: Highest valus to consider
    :param M int: Number of bins
    :param weights ndarray: Array of same length as X with weights for each point, or None if all weights are 1
    :param cyclic bool: Consider the data cyclic or not

    :Returns: The weights in each bin and the sparse mesh

    For a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions :math:`p_i` and :math:`p_{i+1}`, the 
    bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    By default the bins will be placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, \ldots b-\delta/2\}` with 
    :math:`delta = \frac{M-1}{b-a}`.

    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    cdef:
        Py_ssize_t D
        int has_weight = weights is not None
        int n
        np.ndarray[DOUBLE] result
        np.ndarray[DOUBLE] delta
        np.ndarray[DOUBLE, ndim=2] nX
        np.ndarray[DOUBLE] na, nb
        np.ndarray nM
    nM = np.asarray(M).astype(int)
    nX = np.atleast_2d(X).astype(np.float)
    na = np.atleast_1d(a).astype(np.float)
    nb = np.atleast_1d(b).astype(np.float)

    max_d = min(MAX_DIM, 8*sizeof(Py_ssize_t))
    D = nX.shape[1]
    if D > max_d:
        raise ValueError("Error, you cannot have more than {0} dimensions, and you have {1}".format(max_d, D))

    n = nX.shape[0]
    if nM.ndim == 0:
        nM = nM*np.ones((D,), dtype=int)
    if np.any(nM<2):
        raise ValueError("You need to specify at least 2 elements per dimension")
    if (na.ndim != 1 or na.shape[0] != D or
        nb.ndim != 1 or nb.shape[0] != D or
        nM.ndim != 1 or nM.shape[0] != D):
        raise ValueError("Error, incompatible dimensions for a, b, M and X")
    if has_weight:
        weights = np.asarray(weights).astype(float)
        if weights.shape != (n,):
            raise ValueError("weights must be None or a 1D array of same length as X")
    if not has_weight:
        weight = np.empty(())
    result = np.zeros(np.prod(nM), dtype=np.float)
    delta = np.zeros((D,), dtype=np.float)
    _fast_linbin_nd(nX, na, nb, nM, weights, has_weight, cyclic, result, delta)

    if cyclic:
        mesh = [np.linspace(na[i], nb[i]-delta[i], nM[i]) for i in range(D)]
    else:
        mesh = [ np.linspace(na[i]+delta[i]/2, nb[i]-delta[i]/2, nM[i]) for i in range(D) ]

    return result.reshape(tuple(nM)), np.meshgrid(*mesh, sparse=True, indexing='ij')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def _fast_bin_nd(np.ndarray[DOUBLE, ndim=2] X, np.ndarray[DOUBLE] a, np.ndarray[DOUBLE] b,
                 np.ndarray[INT] M, np.ndarray[DOUBLE] weights, int has_weight, int cyclic=0):
    r"""
    Linear Binning as described in Fan and Marron (1994)

    :param X ndarray: Input data
    :param a float: Lowest value to consider
    :param b float: Highest valus to consider
    :param M int: Number of bins
    :param weights ndarray: Array of same size as X with weights for each point, or None if all weights are 1
    :param cyclic bool: Consider the data cyclic or not

    :Returns: The weights in each bin

    For a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions :math:`p_i` and :math:`p_{i+1}`, the 
    bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    By default the bins will be placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, \ldots b-\delta/2\}` with 
    :math:`delta = \frac{M-1}{b-a}`.

    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    cdef:
        Py_ssize_t i, pos, d
        Py_ssize_t D = X.shape[1]
        int nobs = X.shape[0]
        np.ndarray[DOUBLE] gcnts = np.zeros(np.prod(M), np.float)
        object mesh
        np.ndarray[DOUBLE] delta = (b - a) / M
        np.ndarray[DOUBLE] inv_delta = 1 / delta
        np.ndarray[DOUBLE] shift = np.empty((D,), dtype=np.float)
        np.ndarray[DOUBLE] val = np.empty((D,), dtype=np.float)
        np.ndarray[DOUBLE] lower = np.empty((D,), dtype=np.float)
        np.ndarray[DOUBLE] upper = np.empty((D,), dtype=np.float)
        double w
        np.ndarray[INT] base_idx = np.empty((D,), dtype=np.int)
        int N, is_in

    if has_weight:
        assert weights.shape[0] == X.shape[0], "Error, the weights must be None or an array of same size as X"

    for d in range(D):
        shift[d] = -a[d]
        lower[d] = 0
        upper[d] = M[d]

    for i in range(nobs):
        is_in = 1
        for d in range(D):
            val[d] = (X[i,d] + shift[d]) / delta[d]
            if val[d] < lower[d] or val[d] > upper[d]:
                is_in = 0
                break
        if is_in:
            if has_weight:
                w = weights[i]
            else:
                w = 1.
            pos = 0
            for d in range(D):
                pos *= M[d]
                base_idx[d] = <int> floor(val[d])
                if base_idx[d] == M[d]:
                    base_idx[d] -= 1
                pos += base_idx[d]
            gcnts[pos] += w

    mesh = [ np.linspace(a[i]+delta[i]/2, b[i]-delta[i]/2, M[i]) for i in range(D) ]
    return gcnts.reshape(tuple(M)), np.meshgrid(*mesh, sparse=True, indexing='ij')

@cython.embedsignature(True)
@cython.profile(True)
def fast_bin_nd(np.ndarray[DOUBLE, ndim=2] X, np.ndarray[DOUBLE] a,
                   np.ndarray[DOUBLE] b, object M, object weights = None, int cyclic=0):
    r"""
    Linear Binning as described in Fan and Marron (1994)

    :param X ndarray: Input data, (N,D) array with D the dimension and N the number of points
    :param a float: Lowest value to consider
    :param b float: Highest valus to consider
    :param M int: Number of bins
    :param weights ndarray: Array of same length as X with weights for each point, or None if all weights are 1
    :param cyclic bool: Consider the data cyclic or not

    :Returns: The weights in each bin and the sparse mesh

    For a point :math:`x` between bins :math:`b_i` and :math:`b_{i+1}` at positions :math:`p_i` and :math:`p_{i+1}`, the 
    bins will be updated as:

    .. math::

        b_i = b_i + \frac{b_{i+1} - x}{b_{i+1} - b_i}

        b_{i+1} = b_{i+1} + \frac{x - b_i}{b_{i+1} - b_i}

    By default the bins will be placed at :math:`\{a+\delta/2, \ldots, a+k \delta + \delta/1, \ldots b-\delta/2\}` with 
    :math:`delta = \frac{M-1}{b-a}`.

    If cyclic is true, then the bins are placed at :math:`\{a, \ldots, a+k \delta, \ldots, b-\delta\}` with 
    :math:`\delta = \frac{M}{b-a}` and there is a virtual bin in :math:`b` which is fused with :math:`a`.

    """
    cdef:
        Py_ssize_t D
        int has_weight = weights is not None
        int n
    M = np.asarray(M).astype(int)
    if X.ndim != 2:
        raise ValueError("Error, X must be a 2D array")
    D = X.shape[1]
    n = X.shape[0]
    if M.ndim == 0:
        M = M*np.ones((D,), dtype=int)
    if (a.ndim != 1 or a.shape[0] != D or
        b.ndim != 1 or b.shape[0] != D or
        M.ndim != 1 or M.shape[0] != D):
        raise ValueError("Error, incompatible dimensions for a, b, M and X")
    if has_weight:
        weights = np.asarray(weights).astype(float)
        if weights.shape != (n,):
            raise ValueError("weights must be None or an array of same length as X")
    if not has_weight:
        weight = np.empty(())
    return _fast_bin_nd(X, a, b, M, weights, has_weight, cyclic)
