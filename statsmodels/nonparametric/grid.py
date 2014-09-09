from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.lib.stride_tricks import broadcast_arrays
from ..compat.python import range, zip

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
        self._edges = edges

        expected_bounds = np.empty((ndim, 2), dtype=dtype)
        for d in range(ndim):
            ax = grid_axes[d]
            if bin_types[d] == 'N':
                expected_bounds[d] = [ax[0], ax[-1]]
            else:
                expected_bounds[d] = [(3*ax[0]-ax[1])/2, (3*ax[-1]-ax[-2])/2]

        if bounds is None:
            bounds = expected_bounds
        else:
            bounds = np.asarray(bounds)
            if bounds.ndim == 1:
                bounds = bounds[None,:]
            diff_bounds = np.sqrt(np.sum((expected_bounds - bounds)**2)) / sum(expected_bounds[:,1] - expected_bounds[:,0])
            # If bounds are un-expected
            if edges is None and diff_bounds > 1e-5:
                self._bounds = bounds
                self.edges # pre-compute edges as they are probably not regular
        self._bounds = bounds

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
            edges = [ np.empty((s+1,), dtype=self.dtype) for s in self._shape ]
            bin_types = self.bin_types
            for d, (es, bnd, ax) in enumerate(zip(edges, self.bounds, self.grid)):
                es[1:-1] = (ax[1:] + ax[:-1])/2
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
    def start_interval(self):
        """
        For each dimension, the distance between the two first edges, or if there are no edges, the distance between the 
        two first bins (which will be the same thing ...).
        """
        if self._interval is None:
            ndim = self.ndim
            if self._edges is not None:
                axes = self._edges
            else:
                axes = self.grid
            inter = np.empty((ndim,), dtype=self.dtype)
            for d in range(ndim):
                inter[d] = axes[d][1] - axes[d][0]
            self._interval = inter
        return self._interval

    def bin_sizes(self):
        """
        Return the size of each bin, per dimension.

        Notes: this requires computed edges if they are not already present
        """
        edges = self.edges
        return [ es[1:] - es[:-1] for es in edges ]

    @property
    def start_volume(self):
        """
        Return the volume of the first bin, using :py:attr:`start_interval`
        """
        return np.prod(self.start_interval)

    def bin_volumes(self):
        """
        Return the volume of each bin
        """
        if self.ndim == 1:
            return self.bin_sizes()[0]
        bins = np.meshgrid(*self.bin_sizes(), indexing='ij', copy=False, sparse=True)
        return np.prod(bins)

    def full(self, order='F'):
        """
        Return a full representation of the grid.

        If order is 'C', then the first index is the dimension, otherwise the last index is.
        """
        if self._ndim == 1:
            return self._grid[0]
        m = broadcast_arrays(*np.meshgrid(*self._grid, indexing='ij', sparse='True', copy='False'))
        if order is 'C':
            return np.asarray(m)
        return np.dstack(m)

    def linear(self):
        """
        Return a 2D array with all the points "in line"
        """
        if self._ndim == 1:
            return self._grid[0]
        m = self.full()
        npts = np.prod(self.shape)
        m.shape = (npts, self.ndim)
        return m

    def sparse(self):
        """
        Return the sparse representation of the grid.
        """
        if self._ndim == 1:
            return self._grid[0]
        return np.meshgrid(*self._grid, indexing='ij', copy=False, sparse=True)

    def __iter__(self):
        return iter(self._grid)

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

    def transform(self, *fct):
        '''
        Return the grid, transformed by the function given as argument

        Parameters
        ----------
        *fct: fun or list of fun
            Either a single function, or a list with one function per dimension

        Returns
        -------
        Grid
            A new grid with edges and bin positions transformed with the function(s)
        '''
        if len(fct) == 1:
            fct = fct*self.ndim
        if len(fct) != self.ndim:
            raise ValueError('Error, you need to provide either a single function, or as many as there are dimensions')
        edges = [ f(es) for f, es in zip(fct, self.edges) ]
        bounds = [ f(bs) for f, bs in zip(fct, self.bounds) ]
        grid = [ f(gr) for f, gr in zip(fct, self.grid) ]
        return Grid(grid, bounds, self.bin_types, edges, self.dtype)

    def integrate(self, values = None):
        """
        Integrate values over the grid

        If values is None, the integration is of the function f(x) = 1
        """
        if values is None:
            return np.sum(self.bin_volumes())
        values = np.asarray(values)
        return np.sum(values*self.bin_volumes())

    def cum_integrate(self, values = None):
        """
        Integrate values over the grid and return the cumulative values

        If values is None, the integration is of the function f(x) = 1
        """
        if values is None:
            out = self.bin_volumes()
        else:
            values = np.asarray(values)
            out = values * self.bin_volumes()
        for d in range(self.ndim):
            out.cumsum(axis=d, out=out)
        return out

