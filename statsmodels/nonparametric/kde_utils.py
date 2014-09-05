"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module contained a variety of small useful functions.
"""

from __future__ import division, print_function, absolute_import
from ..compat.python import string_types
import numpy as np
import inspect
from .namedtuple import namedtuple

# Find the largest float available for this numpy
if hasattr(np, 'float128'):
    large_float = np.float128
elif hasattr(np, 'float96'):
    large_float = np.float96
else:
    large_float = np.float64

def finite(val):
    return val is not None and np.isfinite(val)


def atleast_2df(*arys):
    """
    Return at least a 2D array, fortran style (e.g. adding dimensions at the end)
    """
    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if ary.ndim == 0:
            ary = ary.reshape(1,1)
        elif ary.ndim == 1:
            ary = ary[:,np.newaxis]
        res.append(ary)
    if len(res) == 1:
        return res[0]
    return res

def make_ufunc(nin = None, nout=1):
    """
    Decorator used to create a ufunc using `np.frompyfunc`. Note that the 
    returns array will always be of dtype 'object'. You should use the `out` if 
    you know the wanted type for the output.

    :param int nin: Number of input. Default is found by using
        ``inspect.getargspec``
    :param int nout: Number of output. Default is 1.
    """
    def f(fct):
        if nin is None:
            Nin = len(inspect.getargspec(fct).args)
        else:
            Nin = nin
        return np.frompyfunc(fct, Nin, nout)
    return f

def _process_trans_args(z, out, input_dim, output_dim, in_dtype, out_dtype):
    """
    This function is the heart of the numpy_trans* functions.
    """
    z = np.asarray(z)
    if in_dtype is not None:
        z = z.astype(in_dtype)
    input_shape = z.shape
    need_transpose = False
    # Compute data shape (i.e. input without the dimension)
    z_empty = False
    if z.ndim == 0:
        z_empty = True
        data_shape = (1,)
        if input_dim == 0:
            z = z.reshape(1)
        else:
            z = z.reshape(1,1)
    elif input_dim == 0:
        data_shape = input_shape
    elif input_dim < 0:
        data_shape = input_shape[:-1]
        input_dim = input_shape[-1]
    else:
        if input_shape[-1] == input_dim:
            data_shape = input_shape[:-1]
        elif input_shape[0] == input_dim:
            data_shape = input_shape[1:]
            need_transpose = True
        else:
            raise ValueError("Error, the input array is of dimension {0} "
                             "(expected: {1})".format(input_shape[-1], input_dim))
    # Allocate the output
    if out is None:
        # Compute the output shape
        if output_dim > 1:
            if need_transpose:
                output_shape = (output_dim,) + data_shape
            else:
                output_shape = data_shape + (output_dim,)
        else:
            output_shape = data_shape
        if out_dtype is None:
            out_dtype = z.dtype
            if issubclass(out_dtype.type, np.integer):
                out_dtype = np.float64
        out = np.empty(output_shape, dtype=out_dtype)
        write_out = out.view()
        if z_empty and output_dim == 1:
            out.shape = ()
    else:
        write_out = out.view()
    # Transpose if needed
    if input_dim != 0:
        size_data = np.prod(data_shape)
        if output_dim > 1:
            if need_transpose:
                write_out.shape = (output_dim, size_data)
            else:
                write_out.shape = (size_data, output_dim)
        else:
            write_out.shape = (size_data,)
        z = z.view()
        if need_transpose:
            z.shape = (input_dim, size_data)
        else:
            z.shape = (size_data, input_dim)
    if need_transpose:
        write_out = write_out.T
        z = z.T
    return z, write_out, out

def numpy_trans(input_dim, output_dim, out_dtype=None, in_dtype=None):
    """
    Decorator to create a function taking a single array-like argument and return a numpy array with the same number of 
    points.

    The function will always get an input and output with the last index corresponding to the dimension of the problem.

    Parameters
    ----------
    input_dim: int
        Number of dimensions of the input. The behavior depends on the value:
            < 0 : The last index is the dimension, but it is of variable size.
            = 0 : The array is passed as-is to the calling method and is assumed to be 1D. The output will have either
                  the same shape, or the same shape with another index for the dimension is output_dim > 1.
            > 0 : There is a dimension, and its size is known. The dimension should be the first or last index. If it is
                  on the first, the arrays are transposed before being sent to the function.

    output_dim: int
        Dimension of the output. If more than 1, the last index of the output array is the dimension. It cannot be 0 or 
        less.

    out_dtype: dtype or None
        Expected types of the output array.
        If the output array is created by this function, dtype specifies its type. If dtype is None, the output array is 
        given the same as the input array, unless it is an integer, in which case the output will be a float64.

    in_dtype: dtype or None
        If not None, the input array will be converted to this type before being passed on.

    Notes
    -----
    If input_dim is not 0, the function will always receive a 2D array with the second index for the dimension.
    """
    if out_dtype is not None:
        out_dtype = np.dtype(out_dtype)
    if in_dtype is not None:
        in_dtype = np.dtype(in_dtype)
    if output_dim <= 0:
        raise ValueError("Error, the number of output dimension must be strictly more than 0.")
    def decorator(fct):
        def f(z, out=None):
            z, write_out, out = _process_trans_args(z, out, input_dim, output_dim,
                                                    in_dtype, out_dtype)
            fct(z, out=write_out)
            return out
        return f
    return decorator

def numpy_trans1d(out_dtype=None, in_dtype=None):
    """
    This decorator helps provide a uniform interface to 1D numpy transformation functions.

    The returned function takes any array-like argument and transform it as a 1D ndarray sent to the decorated function. 
    If the `out` argument is not provided, it will be allocated with the same size and shape as the first argument. And 
    as with the first argument, it will be reshaped as a 1D ndarray before being sent to the function.

    Examples
    --------

    The following example illustrate how a 2D array will be passed as 1D, and the output allocated as the input 
    argument:

    >>> @numpy_trans1d()
    ... def broadsum(z, out):
    ...   out[:] = np.sum(z, axis=0)
    >>> broadsum([[1,2],[3,4]])
    array([[ 10.,  10.], [ 10.,  10.]])

    """
    if out_dtype is not None:
        out_dtype = np.dtype(out_dtype)
    if in_dtype is not None:
        in_dtype = np.dtype(in_dtype)
    def decorator(fct):
        def f(z, out=None):
            z = np.asarray(z)
            if in_dtype is not None:
                z = z.astype(in_dtype)
            if out is None:
                if out_dtype is None:
                    dtype = z.dtype
                else:
                    dtype = out_dtype
                if issubclass(dtype.type, np.integer):
                    dtype = np.float64
                out = np.empty(z.shape, dtype=dtype)
            size_data = np.prod(z.shape)
            if size_data == 0:
                size_data = 1
            z = z.view()
            z.shape = (size_data,)
            write_out = out.view()
            write_out.shape = (size_data,)
            fct(z, write_out)
            return out
        return f
    return decorator


def numpy_trans_method(input_dim, output_dim, out_dtype=None, in_dtype=None):
    """
    Decorator to create a method taking a single array-like argument and return a numpy array with the same number of 
    points.

    The function will always get an input and output with the last index corresponding to the dimension of the problem.

    Parameters
    ----------
    input_dim: int or str
        Number of dimensions of the input. The behavior depends on the value:
            < 0 : The last index is the dimension, but it is of variable size.
            = 0 : The array is passed as-is to the calling method and is assumed to be 1D. The output will have either
                  the same shape, or the same shape with another index for the dimension is output_dim > 1.
            > 0 : There is a dimension, and its size is known. The dimension should be the first or last index. If it is
                  on the first, the arrays are transposed before being sent to the function.
        If a string, it should be the name of an attribute containing the input dimension.

    output_dim: int or str
        Dimension of the output. If more than 1, the last index of the output array is the dimension. If cannot be 0 or 
        less.
        If a string, it should be the name of an attribute containing the output dimension

    out_dtype: dtype or None
        Expected types of the output array.
        If the output array is created by this function, dtype specifies its type. If dtype is None, the output array is 
        given the same as the input array, unless it is an integer, in which case the output will be a float64.

    in_dtype: dtype or None
        If not None, the input array will be converted to this type before being passed on.

    Notes
    -----
    If input_dim is not 0, the function will always receive a 2D array with the second index for the dimension.
    """
    if output_dim <= 0:
        raise ValueError("Error, the number of output dimension must be strictly more than 0.")
    # Resolve how to get input dimension
    if isinstance(input_dim, string_types):
        def get_input_dim(self):
            return getattr(self, input_dim)
    else:
        def get_input_dim(self):
            return input_dim
    # Resolve how to get output dimension
    if isinstance(output_dim, string_types):
        def get_output_dim(self):
            return getattr(self, output_dim)
    else:
        def get_output_dim(self):
            return output_dim
    if out_dtype is not None:
        out_dtype = np.dtype(out_dtype)
    if in_dtype is not None:
        in_dtype = np.dtype(in_dtype)
    # Decorator itself
    def decorator(fct):
        def f(self, z, out=None):
            z, write_out, out = _process_trans_args(z, out, get_input_dim(self), get_output_dim(self),
                                                    in_dtype, out_dtype)
            fct(self, z, out=write_out)
            return out
        return f
    return decorator

def numpy_trans1d_method(out_dtype=None, in_dtype=None):
    '''
    This is the method equivalent to :py:func:`numpy_trans1d`
    '''
    if out_dtype is not None:
        out_dtype = np.dtype(out_dtype)
    if in_dtype is not None:
        in_dtype = np.dtype(in_dtype)
    def decorator(fct):
        def f(self, z, out=None):
            z = np.asarray(z)
            if in_dtype is not None:
                z = z.astype(in_dtype)
            if out is None:
                if out_dtype is None:
                    dtype = z.dtype
                else:
                    dtype = out_dtype
                if issubclass(dtype.type, np.integer):
                    dtype = np.float64
                out = np.empty(z.shape, dtype=dtype)
            size_data = np.prod(z.shape)
            if size_data == 0:
                size_data = 1
            z = z.view()
            z.shape = (size_data,)
            write_out = out.view()
            write_out.shape = (size_data,)
            fct(self, z, write_out)
            return out
        return f
    return decorator

class Grid(object):
    def __init__(self, grid):
        """
        Create a grid from a full or sparse grid as returned by meshgrid or a 1D grid.
        """
        self._ndim = None
        self._order = None
        self._shape = None
        self._dim_index = None
        try:
            grid = np.asarray(grid, dtype=grid[0].dtype)
            self._grid = grid
            ndim = grid.ndim-1
            self._ndim = ndim
            if ndim == 0:
                self._ndim = 1
                self._order = ''
                self._shape = grid.shape
            elif grid.shape[-1] == ndim:
                self._order = 'F'
                self._shape = grid.shape[:-1]
                self._dim_indx = -1
            elif grid.shape[0] == ndim:
                self._order = 'C'
                self._shape = grid.shape[1:]
                self._dim_index = 0
            else:
                raise ValueError("Error, this is not a valid grid")
        except (AttributeError, ValueError): # This might be a sparse grid
            ndim = len(grid)
            shape = [None]*ndim
            for i, m in enumerate(grid):
                if m.ndim != ndim:
                    raise ValueError("Error, this is not a valid (sparse) grid")
                s = m.shape[i]
                target_shape = (1,)*i + (s,) + (1,)*(ndim-i-1)
                if m.shape != target_shape:
                    raise ValueError("Error, this is not a valid (sparse) grid")
                shape[i] = s
            self._ndim = ndim
            self._order = 'S'
            self._shape = shape
            self._dim_index = 0
            self._grid = grid

    @property
    def dim_index(self):
        return self._dim_index

    @property
    def ndim(self):
        return self._ndim

    @property
    def order(self):
        return self._order

    @property
    def shape(self):
        return self._shape

    @property
    def grid(self):
        return self._grid

    def interval(self):
        """
        Compute the interval between dimensions of the grid.
        """
        dim_idx = self.dim_index
        n = self.ndim
        grid = self.sparse()
        upper = [0] * n
        lower = [0] * n
        result = np.empty((n,), dtype=grid[0].dtype)
        for i in range(n):
            upper[i] = 1
            result[i] = grid[i][tuple(upper)] - grid[i][tuple(lower)]
            upper[i] = 0
        return result

    def full(self, order=None):
        if self.order == 'S':
            m = np.meshgrid(*self._grid, indexing='ij')
            if order is 'C':
                return np.asarray(m)
            return np.concatenate([g[...,None] for g in m], axis=-1)
        if order is None or self.order == '' or order == self.order:
            return self._grid
        ndim = self.ndim
        if order == 'C':
            grid = self._grid
            return np.asarray([grid[...,i] for i in range(ndim)])
        return np.concatenate([grid[i,...,None] for i in range(ndim)])


    def sparse(self):
        order = self.order
        if order == 'C' or order == 'F':
            grid = self.grid
            res = [None] * self.ndim
            fst = np.s_[:1]
            col = np.s_[:]
            ndim = self.ndim
            for i in range(ndim):
                idx = (fst,)*i + (col,) + (fst,)*(ndim-i-1)
                if order == 'C':
                    idx = (i,) + idx
                else:
                    idx = idx + (i,)
                res[i] = grid[idx]
            return res
        return self._grid

    def __getitem__(self, idx):
        return self._grid[idx]

#
from scipy import sqrt
from numpy import finfo, asarray, asfarray, zeros

_epsilon = sqrt(finfo(float).eps)


def approx_jacobian(x, func, epsilon, *args):
    """
    Approximate the Jacobian matrix of callable function func

    :param ndarray x: The state vector at which the Jacobian matrix is desired
    :param callable func: A vector-valued function of the form f(x,*args)
    :param ndarray epsilon: The peturbation used to determine the partial derivatives
    :param tuple args: Additional arguments passed to func

    :returns: An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

    .. note::

         The approximation is done using forward differences

    """
    x0 = asarray(x)
    x0 = asfarray(x0, dtype=x0.dtype)
    epsilon = x0.dtype.type(epsilon)
    f0 = func(*((x0,) + args))
    jac = zeros([len(x0), len(f0)], dtype=x0.dtype)
    dx = zeros(len(x0), dtype=x0.dtype)
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (func(*((x0 + dx,) + args)) - f0) / epsilon
        dx[i] = 0.0
    return jac.transpose()
