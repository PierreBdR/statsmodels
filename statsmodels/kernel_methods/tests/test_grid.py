from __future__ import division, absolute_import, print_function

from ..kde_utils import Grid, GridInterpolator
import numpy as np
from ...compat.numpy import np_meshgrid, NumpyVersion
from scipy.interpolate import interp2d
import scipy
from nose.plugins.attrib import attr
from nose.tools import (raises, eq_, set_trace, assert_almost_equal,
                        assert_less_equal, assert_greater_equal)
from numpy.testing import assert_equal, assert_allclose

# interp2d doesn't work on older versions of scipy
can_use_inter2p = NumpyVersion(scipy.__version__) > NumpyVersion('0.11.0')

@attr('kernel_methods')
class TestBasics(object):
    @classmethod
    def setUpClass(cls):
        cls.sparse_grid = np.ogrid[0:11, 1:100:100j, -5.5:5.5:12j, 0:124]
        cls.sparse_grid[-1] = (cls.sparse_grid[-1] + 0.5) * 2 * np.pi / 124
        cls.axes_def = [g.squeeze() for g in cls.sparse_grid]
        cls.full_grid_c = np.array(np_meshgrid(*cls.sparse_grid, indexing='ij'))
        cls.full_grid_f = np.concatenate([g[..., None] for g in np_meshgrid(*cls.sparse_grid, indexing='ij')], axis=-1)
        cls.bin_types = 'DBRC'
        cls.ndim = 4
        cls.shape = cls.full_grid_c.shape[1:]
        cls.bounds = np.asarray([[-.5, 10.5],
                                 [0.5, 100.5],
                                 [-6, 6],
                                 [0, 2 * np.pi]])
        cls.edges = [np.r_[-0.5:10.5:12j],
                     np.r_[0.5:100.5:101j],
                     np.r_[-6:6:13j],
                     np.r_[0:2 * np.pi:125j]]
        cls.start_interval = np.array([e[1] - e[0] for e in cls.edges])
        cls.reference = Grid(cls.axes_def, bounds=cls.bounds, bin_types=cls.bin_types,
                             edges=cls.edges)

    def test_interval(self):
        si = self.reference.start_interval
        assert_equal(si, self.start_interval)

    def test_bin_sizes(self):
        bs = self.reference.bin_sizes()
        es = self.edges
        for i, e in enumerate(es):
            assert_allclose(bs[i], e[1:] - e[:-1])

    def test_bin_volumes(self):
        vols = self.reference.bin_volumes()
        eq_(vols.shape, self.reference.shape)

    def test_linear(self):
        ln = self.reference.linear()
        eq_(ln.shape, (np.prod(self.reference.shape), 4))

    def test_getitem(self):
        it = self.reference[0, 0]
        eq_(it, self.axes_def[0][0])

    @raises(ValueError)
    def test_bad_array_type(self):
        p = [1, 2]
        row = [p, p]
        col = [row, row]
        Grid.fromArrays([col, col])

    @raises(ValueError)
    def test_bad_bounds1(self):
        Grid(self.reference, bounds=[[0, 1], [0, 1], [1, 0], [1, 0]])

    @raises(ValueError)
    def test_bad_bounds2(self):
        Grid(self.reference, bounds=[[0, 1], [0, 1], [0, 0], [0, 1]])

    @raises(ValueError)
    def test_bad_bounds3(self):
        Grid(self.axes_def[0], bounds=[[0, 1], [0, 1], [0, 1], [0, 1]])

    def checkIsSame(self, g):
        assert self.reference.almost_equal(g)

    def test_to_sparse(self):
        assert_equal(self.reference.sparse(), self.sparse_grid)

    def test_to_full_c(self):
        assert_equal(self.reference.full('C'), self.full_grid_c)

    def test_to_full_f(self):
        assert_equal(self.reference.full('F'), self.full_grid_f)

    def test_make_1d(self):
        Grid(self.axes_def[0], bounds=self.bounds[0])

    def test_from_axes(self):
        g = Grid(self.axes_def, bin_types=self.bin_types,
                 edges=self.edges, bounds=self.bounds)
        self.checkIsSame(g)

    def test_from_sparse(self):
        g = Grid.fromSparse(self.sparse_grid, bin_types=self.bin_types,
                            edges=self.edges, bounds=self.bounds)
        self.checkIsSame(g)

    def test_from_full_C(self):
        g = Grid.fromFull(self.full_grid_c, order='C', bin_types=self.bin_types,
                          edges=self.edges, bounds=self.bounds)
        self.checkIsSame(g)

    def test_from_full_F(self):
        g = Grid.fromFull(self.full_grid_f, order='F', bin_types=self.bin_types,
                          edges=self.edges, bounds=self.bounds)
        self.checkIsSame(g)

    def test_make_edges(self):
        g = Grid(self.axes_def, bounds=self.bounds, bin_types=self.bin_types)
        es = g.edges
        for i in range(len(es)):
            assert_less_equal(es[i][0], self.bounds[i, 0])
            assert_greater_equal(es[i][-1], self.bounds[i, 1])

    def test_copy(self):
        g2 = self.reference.copy()
        eq_(self.reference.shape, g2.shape)
        eq_(self.reference.bin_types, g2.bin_types)
        assert self.reference.grid[0] is not g2.grid[0]

    def test_build_from_grid(self):
        g2 = Grid(self.reference)
        eq_(self.reference.shape, g2.shape)
        eq_(self.reference.bin_types, g2.bin_types)
        assert self.reference.grid[0] is not g2.grid[0]

    def test_build_from_grid_dtype(self):
        e1 = self.reference.edges
        eq_(e1[0].dtype, np.dtype(float))
        dt = np.dtype(np.float32)
        g2 = Grid(self.reference, dtype=dt)
        eq_(g2.dtype, dt)
        eq_(g2.edges[0].dtype, dt)

    def test_build_from_grid_edges(self):
        edges = self.reference.edges
        edges = [edges[0] + 1, edges[1], edges[2]]
        g2 = Grid(self.reference, edges=edges)
        assert_equal(g2.edges, edges)

    def test_build_from_grid_bin_types(self):
        g2 = Grid(self.reference, bin_types='CCRR')
        eq_(g2.bin_types, 'CCRR')

    def test_build_from_grid_bin_types_1(self):
        g2 = Grid(self.reference, bin_types='C')
        eq_(g2.bin_types, 'CCCC')

    @raises(ValueError)
    def test_build_from_grid_bin_types_err1(self):
        Grid(self.reference, bin_types='CR')

    @raises(ValueError)
    def test_build_from_grid_bin_types_err2(self):
        Grid(self.reference, bin_types='CCRX')

    def test_build_from_grid_bounds(self):
        bnds = [[-1, 1], [-2, 2], [-3, 3], [-4, 4]]
        g2 = Grid(self.reference, bounds=bnds)
        assert_equal(g2.bounds, bnds)

    def test_transform_all(self):
        g2 = self.reference.copy()
        g2.transform(lambda x: x+1)
        for i in range(g2.ndim):
            assert_equal(g2.edges[i], self.reference.edges[i]+1)
            assert_equal(g2.grid[i], self.reference.grid[i]+1)
        assert_equal(g2.bounds, self.reference.bounds+1)

    def test_transform_one(self):
        g2 = self.reference.copy()
        g2.transform({1: lambda x: x+1})
        for i in range(g2.ndim):
            if i == 1:
                assert_equal(g2.edges[i], self.reference.edges[i]+1)
                assert_equal(g2.grid[i], self.reference.grid[i]+1)
                assert_equal(g2.bounds[i], self.reference.bounds[i]+1)
            else:
                assert_equal(g2.edges[i], self.reference.edges[i])
                assert_equal(g2.grid[i], self.reference.grid[i])
                assert_equal(g2.bounds[i], self.reference.bounds[i])

    def test_unequal_bounds(self):
        g1 = Grid(self.axes_def, bounds=self.bounds + 1, bin_types=self.bin_types,
                  edges=self.edges)
        assert g1 != self.reference

    def test_unequal_bintypes(self):
        g1 = Grid(self.axes_def, bounds=self.bounds + 1, bin_types='CCCC',
                  edges=self.edges)
        assert g1 != self.reference

    def test_equal_badtype(self):
        assert self.reference != [1]

    @raises(ValueError)
    def test_bad_grid(self):
        Grid([[[1, 2], [2, 3]]])

    def test_integrate1(self):
        g = Grid(self.axes_def[:2])
        val = g.integrate()
        assert_almost_equal(val, g.bin_volumes().sum())

    def test_integrate2(self):
        g = Grid(self.axes_def[:2])
        val = g.integrate(0.5*np.ones(g.shape))
        assert_almost_equal(val, 0.5*g.bin_volumes().sum())

    def test_cum_integrate1(self):
        g = Grid(self.axes_def[:2])
        val = g.cum_integrate()
        assert_almost_equal(val[-1, -1], g.integrate())

    def test_cum_integrate2(self):
        g = Grid(self.axes_def[:2])
        values = 0.5 * np.ones(g.shape)
        val = g.cum_integrate(values)
        assert_almost_equal(val[-1, -1], g.integrate(values))

@attr('kernel_methods')
class TestInterpolation(object):
    @classmethod
    def setUpClass(cls):
        ax1 = np.r_[0:2 * np.pi:124j]
        ax1 = (ax1 + (ax1[1] - ax1[0]) / 2)[:-1]
        cls.grid1 = Grid([ax1])
        cls.val1 = np.cos(ax1)
        ax2 = np.r_[-90:90:257j]
        ax2 = (ax2 + (ax2[1] - ax2[0]) / 2)[:-1]
        cls.grid2 = Grid([ax1, ax2])
        sg = cls.grid2.sparse()
        cls.val2 = np.cos(sg[0]) + np.sin(sg[1] * np.pi / 180)
        ax3 = np.r_[0:20]
        cls.grid3 = Grid([ax1, ax3], bin_types='BD')
        sg = cls.grid3.sparse()
        cls.val3 = np.cos(sg[0]) + sg[1]

    @raises(ValueError)
    def test_bad_value_shape(self):
        GridInterpolator(self.grid1, self.val1[:-5])

    @raises(ValueError)
    def test_bad_pts_1d(self):
        self.grid1.bin_types = 'B'
        interp = GridInterpolator(self.grid1.sparse(), self.val1)
        test_values = np.array([[1.]])
        interp_test = interp(test_values)
        interp_comp = np.interp(test_values, self.grid1.full(),
                                self.val1, self.val1[0], self.val1[-1])
        np.testing.assert_allclose(interp_test, interp_comp)

    @raises(ValueError)
    def test_bad_pts_2d_1(self):
        self.grid3.bin_types = 'BB'
        interp = GridInterpolator(self.grid3.sparse(), self.val3)
        test_values = np.array([[[1., 1., 1.]]])
        interp(test_values)

    @raises(ValueError)
    def test_bad_pts_2d_2(self):
        self.grid3.bin_types = 'BB'
        interp = GridInterpolator(self.grid3.sparse(), self.val3)
        test_values = np.array([[1., 1., 1., 1., 1.]])
        interp(test_values)

    def test_0d_pts(self):
        self.grid1.bin_types = 'B'
        interp = GridInterpolator(self.grid1.sparse(), self.val1)
        test_values = np.array(1.)
        interp(test_values)

    def test_1d_pts(self):
        grid = self.grid3
        grid.bin_types = 'BB'
        interp = GridInterpolator(grid.sparse(), self.val3)
        test_values = np.array([1., 1.])
        interp(test_values)

    def test_1d_bounded_from_array(self):
        self.grid1.bin_types = 'B'
        interp = GridInterpolator(self.grid1.sparse(), self.val1)
        test_values = np.random.rand(256) * 3 * np.pi - np.pi / 2
        interp_test = interp(test_values)
        interp_comp = np.interp(test_values, self.grid1.full(),
                                self.val1, self.val1[0], self.val1[-1])
        np.testing.assert_allclose(interp_test, interp_comp)

    def test_1d_bounded(self):
        self.grid1.bin_types = 'B'
        interp = GridInterpolator(self.grid1, self.val1)
        test_values = np.random.rand(256) * 3 * np.pi - np.pi / 2
        interp_test = interp(test_values)
        interp_comp = np.interp(test_values, self.grid1.full(),
                                self.val1, self.val1[0], self.val1[-1])
        np.testing.assert_allclose(interp_test, interp_comp)

    def test_1d_cyclic(self):
        self.grid1.bin_types = 'C'
        interp = GridInterpolator(self.grid1, self.val1)
        test_values = np.random.rand(256) * 3 * np.pi - np.pi / 2
        # Compute equivalent values
        real_values = test_values % (2 * np.pi)
        interp_test = interp(test_values)
        interp_comp = np.interp(real_values, self.grid1.full(),
                                self.val1, self.val1[0], self.val1[-1])
        np.testing.assert_allclose(interp_test, interp_comp)

    def test_1d_reflection(self):
        self.grid1.bin_types = 'R'
        interp = GridInterpolator(self.grid1, self.val1)
        test_values = np.random.rand(256) * 3 * np.pi - np.pi / 2
        # Compute equivalent values
        real_values = test_values % (4 * np.pi)
        real_values[real_values > 2 * np.pi] = 4 * np.pi - real_values[real_values > 2 * np.pi]

        interp_test = interp(test_values)
        interp_comp = np.interp(real_values, self.grid1.full(),
                                self.val1, self.val1[0], self.val1[-1])
        np.testing.assert_allclose(interp_test, interp_comp)

    @staticmethod
    def np_interpolate_2d(ax1, ax2, values, test_values):
        interp = interp2d(ax1, ax2, values.T)
        res = np.empty_like(test_values[:, 0])
        for i in range(test_values.shape[0]):
            res[i] = interp(test_values[i, 0], test_values[i, 1])
        return res

    def test_2d_bounded(self):
        if not can_use_inter2p:
            return
        grid = self.grid2
        grid.bin_types = 'B'
        interp = GridInterpolator(grid, self.val2)
        N = 1024
        test_values = np.c_[np.random.rand(N) * 3 * np.pi - np.pi / 2,
                            np.random.rand(N) * 200 - 100]
        real_values = test_values.copy()
        min_val0 = grid.grid[0][0]
        max_val0 = grid.grid[0][-1]
        min_val1 = grid.grid[1][0]
        max_val1 = grid.grid[1][-1]
        real_values[real_values[:, 0] < min_val0, 0] = min_val0
        real_values[real_values[:, 0] > max_val0, 0] = max_val0
        real_values[real_values[:, 1] < min_val1, 1] = min_val1
        real_values[real_values[:, 1] > max_val1, 1] = max_val1

        interp_test = interp(test_values)
        interp_comp = self.np_interpolate_2d(grid.grid[0], grid.grid[1], self.val2, real_values)
        np.testing.assert_allclose(interp_test, interp_comp)

    def test_2d_cyclic(self):
        if not can_use_inter2p:
            return
        grid = self.grid2
        grid.bin_types = 'C'
        interp = GridInterpolator(grid, self.val2)
        N = 1024
        test_values = np.c_[np.random.rand(N) * 3 * np.pi - np.pi / 2,
                            np.random.rand(N) * 200 - 100]
        min_val0 = grid.grid[0][0]
        max_val0 = grid.grid[0][-1] + grid.start_interval[0]
        delta_val0 = max_val0 - min_val0
        min_val1 = grid.grid[1][0]
        max_val1 = grid.grid[1][-1] + grid.start_interval[1]
        delta_val1 = max_val1 - min_val1

        real_values = test_values.copy()
        real_values[:, 0] = (real_values[:, 0] - min_val0) % delta_val0 + min_val0
        real_values[:, 1] = (real_values[:, 1] - min_val1) % delta_val1 + min_val1

        ax1 = np.concatenate([grid.grid[0], [grid.grid[0][-1] + grid.start_interval[0]]])
        ax2 = np.concatenate([grid.grid[1], [grid.grid[1][-1] + grid.start_interval[1]]])
        val2 = self.val2
        val2 = np.concatenate([val2, val2[:1, :]], axis=0)
        val2 = np.concatenate([val2, val2[:, :1]], axis=1)
        interp_test = interp(test_values)
        interp_comp = self.np_interpolate_2d(ax1, ax2, val2, real_values)
        np.testing.assert_allclose(interp_test, interp_comp)


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
