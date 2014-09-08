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
        if need_transpose:
            z = z.reshape(input_dim, size_data)
        else:
            z = z.reshape(size_data, input_dim)
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
    """
    Object representing a grid.
    """
    def __init__(self, grid_axes, bounds=None, bin_types=None, edges=None, dtype=None):
        """
        Create a grid from a full or sparse grid as returned by meshgrid or a 1D grid.

        Parameters
        ----------
        grid_axes: list of ndarray
            Each ndarray can have at most 1 dimension with more than 1 element. This dimension contains the position of 
            each point on the axis
        bounds: ndarray
            This is a Dx2 array. For each dimension, the lower and upper bound of the axes (doesn't have to correspond 
            to the min and max of the axes.
        bin_types: str
            A string with as many letter as there are dimensions. For each dimension, gives the kind of axes. Can be one 
            of 'U', 'R', 'C' or 'N' (See :py:attr:`bin_types`). If a single letter is provided, this is the class for 
            all axes. If not specified, the default is 'U'.
        edges: list of ndarray
            If provided, should be a list with one array per dimension. Each array should have one more element than the 
            bin for that dimension. These represent the edges of the bins.
        """
        self._interval = None
        if isinstance(grid_axes, Grid):
            self._grid = grid_axes.grid
            self._ndim = grid_axes.ndim
            if bounds is None:
                self._bounds = grid_axes._bounds
            else:
                self._bounds = bounds
            if bin_types is None:
                self._bin_types = grid_axes._bin_types
            else:
                if len(bin_types) == 1:
                    bin_types = bin_types * self._ndim
                if len(bin_types) != self._ndim:
                    raise ValueError("Error, there must be as many bin types as bins")
                self._bin_types = bin_types
            if edges is None:
                self._edges = grid_axes._edges
            else:
                self._edges = edges
            if dtype is not None and dtype != grid_axes._dtype:
                self._dtype = dtype
                self._bounds = self._bounds.astype(dtype)
                self._grid = [ g.astype(dtype) for d in self._grid ]
                if self._edges is not None:
                    self._edges = [ e.astype(dtype) for e in self._edges ]
            return
        first_elemt = np.asarray(grid_axes[0])
        if first_elemt.ndim == 0:
            ndim = 1
            grid_axes = [ np.asarray(grid_axes) ]
        else:
            ndim = len(grid_axes)
            grid_axes = [ np.asarray(ax) for ax in grid_axes ]
        if dtype is None:
            dtype = np.find_common_type([ax.dtype for ax in grid_axes], [])
        for d in range(ndim):
            if grid_axes[d].ndim != 1:
                raise ValueError("Error, the axis of a grid must be 1D arrays or "
                                 "have exacltly one dimension with more than 1 element")
            grid_axes[d] = grid_axes[d].astype(dtype)
        self._grid = grid_axes
        self._ndim = ndim
        if bin_types is None:
            bin_types = 'U' * ndim
        if len(bin_types) == 1:
            bin_types = bin_types * ndim
        elif len(bin_types) != ndim:
            raise ValueError("Error, there must be as many bin_types as dimensions")
        self._bin_types = bin_types
        self._shape = tuple(len(ax) for ax in grid_axes)
        if bounds is None:
            bounds = np.empty((ndim, 2), dtype=dtype)
            for d in range(ndim):
                ax = grid_axes[d]
                if bin_types[d] == 'N':
                    bounds[d] = [ax[0], ax[-1]]
                elif bin_types[d] == 'C':
                    bounds[d] = [ax[0], 2*ax[-1]-ax[-2]]
                else:
                    bounds[d] = [(3*ax[0]-ax[1])/2, (3*ax[-1]-ax[-2])/2]
        else:
            bounds = np.asarray(bounds)
            if bounds.ndim == 1:
                bounds = bounds[None,:]
        self._bounds = bounds
        self._edges = edges

    def __repr__(self):
        dims = 'x'.join(str(s)+bt for s, bt in zip(self.shape, self.bin_types))
        lims = '[{}]'.format(" ; ".join('{0:g} - {1:g}'.format(b[0], b[1]) for b in self.bounds))
        return "<Grid {0}, {1}, dtype={2}>".format(dims, lims, self.dtype)

    @staticmethod
    def fromSparse(grid, *args, **kwords):
        return Grid([np.squeeze(g) for g in grid], *args, **kwords)

    @staticmethod
    def fromFull(grid, order='F', *args, **kwords):
        """
        Create a Grid from a full mesh represented as a single ndarray.
        """
        grid_shape = None
        if order == 'F':
            grid_shape = grid.shape[:-1]
            ndim = grid.shape[-1]
        else:
            grid_shape = grid.shape[1:]
            ndim = grid.shape[0]
        if len(grid_shape) != ndim:
            raise ValueError("This is not a valid grid")
        grid_axes = [None]*ndim
        selector = [0]*ndim
        for d in range(ndim):
            selector[d] = np.s_[:]
            if order == 'F':
                sel = tuple(selector) + (d,)
            else:
                sel = (d,) + tuple(selector)
            grid_axes[d] = grid[sel]
            selector[d] = 0
        return Grid(grid_axes, *args, **kwords)

    @staticmethod
    def fromArrays(grid, *args, **kwords):
        """
        Create a grid from a list of grids, a list of arrays or a full array C or Fortram-style.
        """
        try:
            grid = np.asarray(grid).squeeze()
            if not np.issubdtype(grid.dtype, np.number):
                raise ValueError('Is not full numeric grid')
            if grid.ndim == 2: # Cannot happen for full grid
                raise ValueError('Is not full numeric grid')
            ndim = grid.ndim-1
            if grid.shape[-1] == ndim:
                return Grid.fromFull(grid, 'F', *args, **kwords)
            elif grid.shape[0] == ndim:
                return Grid.fromFull(grid, 'C', *args, **kwords)
        except ValueError as ex:
            return Grid.fromSparse(grid, *args, **kwords)
        raise ValueError("Couldn't find what kind of grid this is.")

    @property
    def ndim(self):
        """
        Number of dimensions of the grid
        """
        return self._ndim

    @property
    def bin_types(self):
        """
        Types of the axes.

        The valid types are:
            - U: Unbounded
            - R: Reflective
            - C: Cyclic
            - N: Non-continuous
        """
        return self._bin_types

    @property
    def shape(self):
        """
        Shape of the grid (e.g. number of bin for each dimension)
        """
        return self._shape

    @property
    def edges(self):
        """
        list of ndarray
            Edges of the bins for each dimension
        """
        if self._edges is None:
            edges = [ np.empty((s+1,), dtype=dtype) for s in self._shape ]
            for d, (es, bnd, ax) in enumerate(zip(edges, self.bounds, self.grid_axes)):
                es[1:-1] = (ax[1:] + ax[:-1])/2
                if bin_types[d] == 'C':
                    es[0]
                else:
                    es[0] = bnd[0]
                    es[-1] = bnd[1]
            self._edges = edges
        return self._edges

    @property
    def grid(self):
        """
        list of ndarray
            Position of the bins for each dimensions
        """
        return self._grid

    @property
    def dtype(self):
        """
        Type of arrays for the bin positions
        """
        return self._grid[0].dtype

    @property
    def bounds(self):
        """
        ndarray
            Dx2 array with the bounds of each axes
        """
        return self._bounds

    @property
    def interval(self):
        """
        For each dimension, the distance between the two first bins.
        """
        if self._interval is None:
            ndim = self.ndim
            grid_axes = self.grid
            inter = np.empty((ndim,), dtype=self.dtype)
            for d in range(ndim):
                inter[d] = grid_axes[d][1] - grid_axes[d][0]
            self._interval = inter
        return self._interval

    def full(self, order='F'):
        """
        Return a full representation of the grid.

        If order is 'C', then the first index is the dimension, otherwise the last index is.
        """
        if self._ndim == 1:
            return self._grid[0]
        m = np.meshgrid(*self._grid, indexing='ij')
        if order is 'C':
            return np.asarray(m)
        return np.concatenate([g[...,None] for g in m], axis=-1)

    def linear(self):
        """
        Return a 2D array with all the points "in line"
        """
        if self._ndim == 1:
            return self._grid[0]
        m = np.meshgrid(*self._grid, indexing='ij')
        npts = np.prod(self.shape)
        ndim = self.ndim
        return np.concatenate([g.reshape(npts, 1) for g in m], axis=1)

    def sparse(self):
        """
        Return the sparse representation of the grid.
        """
        if self._ndim == 1:
            return self._grid[0]
        return np.meshgrid(*self._grid, indexing='ij', copy=False, sparse=True)

    def __len__(self):
        return len(self._grid)

    def __getitem__(self, idx):
        """
        Shortcut to access bin positions.

        Usage
        -----
        >>> grid = Grid([[1,2,3,4,5],[-1,0,1],[7,8,9,10]])
        >>> grid[0,2]
        3
        >>> grid[2,:2]
        array([7,8])
        >>> grid[1]
        array([-1,0,1])
        """
        try:
            dim, pos = idx
            return self._grid[dim][pos]
        except TypeError:
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
