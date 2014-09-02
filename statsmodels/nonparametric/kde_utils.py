"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module contained a variety of small useful functions.
"""

from __future__ import division, print_function, absolute_import
from ..compat.python import string_types
from collections import OrderedDict
from keyword import iskeyword as _iskeyword
from operator import itemgetter as _itemgetter
import sys
from ..compat.python import string_types
import numpy as np
import inspect

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

def _process_trans_args(z, out, input_dim, output_dim, dtype):
    """
    This function is the heart of the numpy_trans* functions.
    """
    z = np.asarray(z)
    if dtype is not None:
        z = z.astype(dtype)
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
        if dtype is None:
            dtype = z.dtype
            if issubclass(dtype.type, np.integer):
                dtype = np.float64
        out = np.empty(output_shape, dtype=dtype)
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

def numpy_trans(input_dim, output_dim, dtype=None):
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

    dtype: dtype or None
        Expected types of the arrays. The input array is converted to this type. If set to None, the input array is left 
        untouched.
        If the output array is created by this function, dtype specifies its type. If dtype is None, the output array is 
        given the same as the input array, unless it is an integer, in which case the output will be a float64.

    Notes
    -----
    If input_dim is not 0, the function will always receive a 2D array with the second index for the dimension.
    """
    if output_dim <= 0:
        raise ValueError("Error, the number of output dimension must be strictly more than 0.")
    def decorator(fct):
        def f(z, out=None):
            z, write_out, out = _process_trans_args(z, out, input_dim, output_dim, dtype)
            fct(z, out=write_out)
            return out
        return f
    return decorator

def numpy_trans1d(fct):
    """
    This decorator helps provide a uniform interface to 1D numpy transformation functions.

    The returned function takes any array-like argument and transform it as a 1D ndarray sent to the decorated function. 
    If the `out` argument is not provided, it will be allocated with the same size and shape as the first argument. And 
    as with the first argument, it will be reshaped as a 1D ndarray before being sent to the function.

    Examples
    --------

    The following example illustrate how a 2D array will be passed as 1D, and the output allocated as the input 
    argument:

    >>> @numpy_trans1d
    ... def broadsum(z, out):
    ...   out[:] = np.sum(z, axis=0)
    >>> broadsum([[1,2],[3,4]])
    array([[ 10.,  10.], [ 10.,  10.]])

    """
    def f(z, out=None):
        z = np.asarray(z)
        if out is None:
            dtype = z.dtype
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


def numpy_trans_method(input_dim, output_dim, dtype=None):
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

    dtype: dtype or None
        Expected types of the arrays. The input array is converted to this type. If set to None, the input array is left 
        untouched.
        If the output array is created by this function, dtype specifies its type. If dtype is None, the output array is 
        given the same as the input array, unless it is an integer, in which case the output will be a float64.

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
    # Decorator itself
    def decorator(fct):
        def f(self, z, out=None):
            z, write_out, out = _process_trans_args(z, out, get_input_dim(self), get_output_dim(self), dtype)
            fct(self, z, out=write_out)
            return out
        return f
    return decorator

def numpy_trans1d_method(fct):
    '''
    This is the method equivalent to :py:fun:`numpy_trans1d`
    '''
    def f(self, z, out=None):
        z = np.asarray(z)
        if out is None:
            dtype = z.dtype
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

def namedtuple(typename, field_names, verbose=False, rename=False):
    """Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', 'x y')
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessable by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

    """

    # Parse and validate the field names.  Validation serves two purposes,
    # generating informative error messages and preventing template injection attacks.
    if isinstance(field_names, string_types):
        # names separated by whitespace and/or commas
        field_names = field_names.replace(',', ' ').split()
    field_names = tuple(map(str, field_names))
    forbidden_fields = {'__init__', '__slots__', '__new__', '__repr__', '__getnewargs__'}
    if rename:
        names = list(field_names)
        seen = set()
        for i, name in enumerate(names):
            need_suffix = (not all(c.isalnum() or c == '_' for c in name) or _iskeyword(name)
                           or not name or name[0].isdigit() or name.startswith('_')
                           or name in seen)
            if need_suffix:
                names[i] = '_%d' % i
            seen.add(name)
        field_names = tuple(names)
    for name in (typename,) + field_names:
        if not all(c.isalnum() or c == '_' for c in name):
            raise ValueError('Type names and field names can only contain alphanumeric characters '
                             'and underscores: %r' % name)
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a keyword: %r' % name)
        if name[0].isdigit():
            raise ValueError('Type names and field names cannot start with a number: %r' % name)
    seen_names = set()
    for name in field_names:
        if name.startswith('__'):
            if name in forbidden_fields:
                raise ValueError('Field names cannot be on of %s' % ', '.join(forbidden_fields))
        elif name.startswith('_') and not rename:
            raise ValueError('Field names cannot start with an underscore: %r' % name)
        if name in seen_names:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen_names.add(name)

    # Create and fill-in the class template
    numfields = len(field_names)
    argtxt = repr(field_names).replace("'", "")[1:-1]   # tuple repr without parens or quotes
    reprtxt = ', '.join('%s=%%r' % name for name in field_names)
    template = '''class %(typename)s(tuple):
        '%(typename)s(%(argtxt)s)' \n
        __slots__ = () \n
        _fields = %(field_names)r \n
        def __new__(_cls, %(argtxt)s):
            'Create new instance of %(typename)s(%(argtxt)s)'
            return _tuple.__new__(_cls, (%(argtxt)s)) \n
        @classmethod
        def _make(cls, iterable, new=tuple.__new__, len=len):
            'Make a new %(typename)s object from a sequence or iterable'
            result = new(cls, iterable)
            if len(result) != %(numfields)d:
                raise TypeError('Expected %(numfields)d arguments, got %%d' %% len(result))
            return result \n
        def __repr__(self):
            'Return a nicely formatted representation string'
            return '%(typename)s(%(reprtxt)s)' %% self \n
        def _asdict(self):
            'Return a new OrderedDict which maps field names to their values'
            return OrderedDict(zip(self._fields, self)) \n
        def _replace(_self, **kwds):
            'Return a new %(typename)s object replacing specified fields with new values'
            result = _self._make(map(kwds.pop, %(field_names)r, _self))
            if kwds:
                raise ValueError('Got unexpected field names: %%r' %% kwds.keys())
            return result \n
        def __getnewargs__(self):
            'Return self as a plain tuple.  Used by copy and pickle.'
            return tuple(self) \n\n''' % dict(numfields=numfields, field_names=field_names,
                                              typename=typename, argtxt=argtxt, reprtxt=reprtxt)
    for i, name in enumerate(field_names):
        template += "        %s = _property(_itemgetter(%d), " \
                    "doc='Alias for field number %d')\n" % (name, i, i)
    if verbose:
        print(template)

    # Execute the template string in a temporary namespace and
    # support tracing utilities by setting a value for frame.f_globals['__name__']
    namespace = dict(_itemgetter=_itemgetter, __name__='namedtuple_%s' % typename,
                     OrderedDict=OrderedDict, _property=property, _tuple=tuple)
    try:
        exec(template, namespace)
    except SyntaxError as e:
        raise SyntaxError(e.message + ':\n' + template)
    result = namespace[typename]

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    try:
        result.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return result


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
