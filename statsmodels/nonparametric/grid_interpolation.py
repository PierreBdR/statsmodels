import numpy as np
from . import _grid_interpolation
from .grid import Grid

class GridInterpolator(object):
    def __init__(self, grid, values):
        """
        Interpolation on a grid

        Parameters
        ----------
        grid: kde_util.Grid or list of ndarray or ndarray
            Should be a grid as generated by meshgrid, possibly concatenated
        values: ndarray
            Array with values for each position of the grid
        """
        if not isinstance(grid, Grid):
            grid = Grid.fromArrays(grid)
        self._grid = [ np.ascontiguousarray(grid[i], float) for i in range(grid.ndim) ]
        self._values = np.asarray(values, float)
        if self._values.shape != grid.shape:
            raise ValueError("The values must have the same shape as the grid")
        self._bin_types = grid.bin_types
        self._ndim = grid.ndim
        self._bounds = grid.bounds
        if self._ndim == 1:
            self._grid = self._grid[0]
            self._call = self.eval1d
        else:
            self._call = self.evalnd

    @property
    def ndim(self):
        return self._ndim

    def eval1d(self, pts, out=None):
        pts = np.asarray(pts).astype(float)
        if pts.ndim == 0:
            pts = np.array([pts], dtype=float)
        elif pts.ndim > 1:
            raise ValueError("Error, the input array must be a float or 1D")
        if out is None:
            out = np.zeros(pts.shape,dtype=float)
        _grid_interpolation.interp1d(pts, self._bounds[0,0], self._bounds[0,1],
                                     self._grid, self._values, self._bin_types,
                                     out)
        return out.squeeze()

    def evalnd(self, pts, out=None):
        pts = np.asarray(pts).astype(float)
        if pts.ndim == 1:
            pts = np.array([pts], dtype=float)
        elif pts.ndim != 2:
            raise ValueError("Error, the input array must be 1 or 2D array")
        if pts.shape[-1] != self.ndim:
            raise ValueError("Error, {0} dimensions expected, but pts has {1}".format(self.ndim, pts.shape[-1]))
        if out is None:
            out = np.zeros(pts.shape[:-1],dtype=float)
        _grid_interpolation.interpnd(pts, self._bounds[:,0], self._bounds[:,1],
                                     self._grid, self._values, self._bin_types,
                                     out)
        return out.squeeze()

    def __call__(self, pts, out=None):
        return self._call(pts, out)
