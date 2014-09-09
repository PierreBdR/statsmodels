"""
cython -a fast_linbin.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -o fast_linbin.so fast_linbin.c
"""

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport floor, fmod

ctypedef np.float64_t DOUBLE
ctypedef np.int_t INT

DEF UNBOUNDED = 0
DEF REFLECTED = 1
DEF CYCLIC = 2
DEF NON_CONTINUOUS = 3

cdef object bin_type_map = dict(U=UNBOUNDED,
                                R=REFLECTED,
                                C=CYCLIC,
                                N=NON_CONTINUOUS)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def fast_linbin(np.ndarray[DOUBLE] X not None,
                double a, double b,
                np.ndarray[DOUBLE] grid not None,
                np.ndarray[DOUBLE] weights not None,
                str s_bin_type):
    cdef:
        Py_ssize_t i
        int nobs = X.shape[0]
        int M = grid.shape[0]
        np.ndarray[DOUBLE] mesh
        double delta = (b - a) / M
        double inv_delta = 1 / delta
        double shift
        double rem
        double val, dval
        double lower
        double upper
        double w
        int base_idx
        int N
        int has_weight = len(weights) > 0
        int bin_type
        object bounds

    try:
        bin_type = bin_type_map[s_bin_type]
    except KeyError as err:
        raise ValueError('Error, invalid bin type: {0}'.format(err.args[0]))

    if bin_type == CYCLIC:
        shift = -a-delta/2
        lower = 0
        upper = M
    elif bin_type == NON_CONTINUOUS:
        shift = -a
        lower = 0
        upper = M-1
    else: # REFLECTED of UNBOUNDED
        shift = -a-delta/2
        lower = -0.5
        upper = M-0.5

    for i in range(nobs):
        val = (X[i] + shift) * inv_delta
        if bin_type == CYCLIC:
            if val < lower:
                rem = fmod(lower - val, M)
                val = upper - rem
            if val >= upper:
                rem = fmod(val - upper, M)
                val = lower + rem
        elif bin_type == REFLECTED:
            if val < lower:
                rem = fmod(lower - val, 2*M)
                if rem < M:
                    val = lower + rem
                else:
                    val = upper - rem + M
            elif val > upper:
                rem = fmod(val - upper, 2*M)
                if rem < M:
                    val = upper - rem
                else:
                    val = lower + rem - M
        else: # UNBOUNDED or NON_CONTINUOUS
            if val < lower or val > upper:
                continue # Skip this sample

        base_idx = <int> floor(val);
        if has_weight:
            w = weights[i]
        else:
            w = 1.
        if bin_type == NON_CONTINUOUS:
            grid[base_idx] += w
        else:
            rem = val - base_idx
            if bin_type == CYCLIC:
                grid[base_idx] += (1-rem)*w
                if base_idx == M-1:
                    grid[0] += rem*w
                else:
                    grid[base_idx+1] += rem*w
            else: # UNBOUNDED or REFLECTED
                if base_idx < 0:
                    grid[0] += w
                elif base_idx >= M-1:
                    grid[base_idx] += w
                else:
                    grid[base_idx] += (1-rem)*w
                    grid[base_idx+1] += rem*w

    if bin_type == NON_CONTINUOUS:
        mesh = np.linspace(a, b, M)
        bounds = [a, b]
    else:
        mesh = np.linspace(a+delta/2, b-delta/2, M)
        bounds = [a, b]

    return mesh, bounds


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def fast_bin(np.ndarray[DOUBLE] X not None,
             double a, double b,
             np.ndarray[DOUBLE] grid not None,
             np.ndarray[DOUBLE] weights not None,
             str s_bin_type):
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
        int M = grid.shape[0]
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
        int has_weight = len(weights) > 0
        int bin_type

    try:
        bin_type = bin_type_map[s_bin_type]
    except KeyError as err:
        raise ValueError('Error, invalid bin type: {0}'.format(err.args[0]))

    shift = -a
    lower = 0
    upper = M

    for i in range(nobs):
        val = (X[i] + shift) * inv_delta

        if bin_type == CYCLIC:
            if val < lower:
                rem = fmod(lower - val, M)
                val = upper - rem
            if val >= upper:
                rem = fmod(val - upper, M)
                val = lower + rem
        elif bin_type == REFLECTED:
            if val < lower:
                rem = fmod(lower - val, 2*M)
                if rem < M:
                    val = lower + rem
                else:
                    val = upper - rem + M
            elif val > upper:
                rem = fmod(val - upper, 2*M)
                if rem < M:
                    val = upper - rem
                else:
                    val = lower + rem - M
        else: # UNBOUNDED or NON_CONTINUOUS
            if val < lower or val > upper:
                continue # Skip this sample

        base_idx = <int> floor(val);
        if has_weight:
            w = weights[i]
        else:
            w = 1.
        if base_idx == M:
            base_idx -= 1
        grid[base_idx] += w

    return np.linspace(a+delta/2, b-delta/2, M), [a, b]


# Note: this define is NOT the limiting factor in the algorithm. See the code for details.
# Ideally, this constant should be the number of bits in Py_ssize_t
DEF MAX_DIM = 64

MAX_DIMENSION = min(MAX_DIM, 8*sizeof(Py_ssize_t))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(True)
def fast_linbin_nd(np.ndarray[DOUBLE, ndim=2] X not None,
                    np.ndarray[DOUBLE] a not None,
                    np.ndarray[DOUBLE] b not None,
                    object grid,
                    np.ndarray[DOUBLE] weights not None,
                    str s_bin_types):
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
        double delta[MAX_DIM]
        double w
        int base_idx[MAX_DIM]
        int next_idx[MAX_DIM]
        int is_out
        Py_ssize_t nb_corner = 1 << D
        double wc
        Py_ssize_t pos
        void *data = np.PyArray_DATA(grid)
        np.npy_intp *strides = np.PyArray_STRIDES(grid)
        np.npy_intp *M = np.PyArray_DIMS(grid)
        int bin_types[MAX_DIM]
        int has_weight = weights.shape[0] > 0

    for d in range(D):
        try:
            bin_types[d] = bin_type_map[s_bin_types[d]]
        except KeyError as err:
            raise ValueError("Error, letter '{0}' is invalid. "
                    "bin_types letters must be one of 'U', 'C', 'R' or 'N'".format(s_bin_types[d]))

        delta[d] = (b[d] - a[d]) / M[d]
        if bin_types[d] == CYCLIC:
            shift[d] = -a[d]-delta[d]/2
            lower[d] = 0
            upper[d] = M[d]
        elif bin_types[d] == NON_CONTINUOUS:
            shift[d] = -a[d]
            lower[d] = 0
            upper[d] = M[d]-1
        else:
            shift[d] = -a[d]-delta[d]/2
            lower[d] = -0.5
            upper[d] = M[d]-0.5

    for i in range(nobs):
        is_out = 0
        for d in range(D):
            val[d] = (X[i,d] + shift[d]) / delta[d]

            if bin_types[d] == CYCLIC:
                if val[d] < lower[d]:
                    rem[d] = fmod(lower[d] - val[d], M[d])
                    val[d] = upper[d] - rem[d]
                if val[d] >= upper[d]:
                    rem[d] = fmod(val[d] - upper[d], M[d])
                    val[d] = lower[d] + rem[d]
            elif bin_types[d] == REFLECTED:
                if val[d] < lower[d]:
                    rem[d] = fmod(lower[d] - val[d], 2*M[d])
                    if rem[d] < M[d]:
                        val[d] = lower[d] + rem[d]
                    else:
                        val[d] = upper[d] - rem[d] + M[d]
                elif val[d] > upper[d]:
                    rem[d] = fmod(val[d] - upper[d], 2*M[d])
                    if rem[d] < M[d]:
                        val[d] = upper[d] - rem[d]
                    else:
                        val[d] = lower[d] + rem[d] - M[d]
            else: # UNBOUNDED or NON_CONTINUOUS
                if val[d] < lower[d] or val[d] > upper[d]:
                    is_out = 1
                    break

        if is_out: continue
        if has_weight:
            w = weights[i]
        else:
            w = 1.

        for d in range(D):
            base_idx[d] = <int> floor(val[d])
            if bin_types[d] == NON_CONTINUOUS:
                rem[d] = 0
            else:
                rem[d] = val[d] - base_idx[d]
            if bin_types[d] == CYCLIC:
                if base_idx[d] == M[d]-1:
                    next_idx[d] = 0
                else:
                    next_idx[d] = base_idx[d]+1
            else:
                if base_idx[d] < 0:
                    base_idx[d] = 0
                    next_idx[d] = 1
                    rem[d] = 0
                elif base_idx[d] >= M[d]-1:
                    rem[d] = 0
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
                if c & 1:
                    wc *= 1-rem[d]
                    pos += strides[d]*base_idx[d]
                else:
                    wc *= rem[d]
                    pos += strides[d]*next_idx[d]
                c >>= 1
            (<double*>(data+pos))[0] += wc

    mesh = [None]*D
    bounds = np.zeros((D,2), dtype=np.float)
    for d in range(D):
        if bin_types[d] == NON_CONTINUOUS:
            mesh[d] = np.linspace(a[d], b[d], M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]
        else: # UNBOUNDED or REFLECTED
            mesh[d] = np.linspace(a[d]+delta[d]/2, b[d]-delta[d]/2, M[d])
            bounds[d,0] = a[d]
            bounds[d,1] = b[d]

    return mesh, bounds


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def fast_bin_nd(np.ndarray[DOUBLE, ndim=2] X not None,
                np.ndarray[DOUBLE] a not None,
                np.ndarray[DOUBLE] b not None,
                object grid,
                np.ndarray[DOUBLE] weights not None,
                str s_bin_types):
    cdef:
        Py_ssize_t i, pos, d
        Py_ssize_t D = X.shape[1]
        int nobs = X.shape[0]
        object mesh
        double delta[MAX_DIM]
        double inv_delta[MAX_DIM]
        double shift[MAX_DIM]
        double val[MAX_DIM]
        double lower[MAX_DIM]
        double upper[MAX_DIM]
        double w
        int base_idx[MAX_DIM]
        int N, is_in
        int bin_types[MAX_DIM]
        int has_weight = weights.shape[0] > 0
        void *data = np.PyArray_DATA(grid)
        np.npy_intp *strides = np.PyArray_STRIDES(grid)
        np.npy_intp *M = np.PyArray_DIMS(grid)

    for d in range(D):
        try:
            bin_types[d] = bin_type_map[s_bin_types[d]]
        except KeyError as err:
            raise ValueError("Error, letter '{0}' is invalid. "
                    "bin_types letters must be one of 'U', 'C', 'R' or 'N'".format(s_bin_types[d]))

        delta[d] = (b[d] - a[d])/M[d]
        inv_delta[d] = 1 / delta[d]
        shift[d] = -a[d]
        lower[d] = 0
        upper[d] = M[d]

    for i in range(nobs):
        is_in = 1
        for d in range(D):
            val[d] = (X[i,d] + shift[d]) / delta[d]
            if bin_types[d] == CYCLIC:
                while val[d] < lower[d]:
                    val[d] += M[d]
                while val[d] > upper[d]:
                    val[d] -= M[d]
            elif bin_types[d] == REFLECTED:
                while val[d] < lower[d] or val[d] > upper[d]:
                    if val[d] < lower[d]:
                        val[d] =  2*lower[d] - val[d]
                    if val[d] > upper[d]:
                        val[d] =  2*upper[d] - val[d]
            elif val[d] < lower[d] or val[d] > upper[d]:
                is_in = 0
                break
        if is_in:
            if has_weight:
                w = weights[i]
            else:
                w = 1.
            pos = 0
            for d in range(D):
                base_idx[d] = <int> floor(val[d])
                if base_idx[d] == M[d]:
                    base_idx[d] -= 1
                pos += strides[d]*base_idx[d]
            (<double*>(data+pos))[0] += w

    mesh = [ np.linspace(a[i]+delta[i]/2, b[i]-delta[i]/2, M[i]) for i in range(D) ]
    return mesh, np.c_[a,b]

