import numpy as np
from ..compat.python import range, zip
from scipy import optimize, integrate, interpolate

class LeaveOut(object):
    """
    Object that returns the arrays, leaving parts out.

    The object can be iterated on and return a tuples whose first element is the index(es) of the element(s) left out, 
    and the second element is a tuple of same size of `data` with what is to be used for the arrays.

    If no parameter is specified beside the data, then an exhaustive leave-one-out is performed.

    Parameters
    ----------
    exog: ndarray
        Data used to fit
    *data: tuple
        Other arrays or values to select for. If the value doesn't have the same length as exog, then it will be sent 
        as-is all the times. Otherwise, it will be selected like exog.

    Optional Parameters
    -------------------
    sampling: int
        Instead of an exhaustive leave-one-out, a random sub-sample is iterated over
    folding: int
        The exogeneous dataset is split into k groups of same length. For each iteration, (k-1) groups are used for 
        fitting and the last one is used for testing.
    """
    def __init__(self, exog, *data, **kwords):
        sampling = kwords.get('sampling', None)
        data = (exog,) + data
        self.data = data
        self.is_sel = [d.shape[0] == exog.shape[0] for d in data]
        self.n = exog.shape[0]
        n = self.n
        if sampling is not None and sampling > n:
            sampling = None
        self.sampling = sampling
        if sampling is not None:
            self.ks = np.random.randint(0, n, size=sampling)

    def __len__(self):
        if self.sampling is None:
            return self.n
        return self.sampling

    def __iter__(self):
        data = self.data
        is_sel = self.is_sel
        sel = np.ones((self.n,), dtype=bool)
        n = self.n
        if self.sampling is None:
            for i in range(n):
                sel[i] = False
                yield (i,) + tuple(d[sel] if is_sel[i] else d for i, d in enumerate(data))
                sel[i] = True
        else:
            for k in self.ks:
                sel[k] = False
                yield (k, tuple(d[sel] if is_sel[i] else d for i, d in enumerate(data)))
                sel[k] = True

def integrate_grid(values, grid=None, dv=None):
    if grid.ndim == 1:
        return integrate.trapz(values, grid)
    if grid is not None:
        n = grid.shape[-1]
        size_dp = tuple(d-1 for d in grid.shape[:n])
        dp = np.ones(size_dp, dtype=float)
        for i in range(n):
            left = (np.s_[:-1],) * i
            right = (np.s_[:-1],) * (n-i-1) + (i,)
            upper = (np.s_[1:],)
            lower = (np.s_[:-1],)
            dp *= (grid[left + upper + right] - grid[left + lower + right])
        n1 = n-1
        var = [[0,1]] * n
        starts = np.array(np.meshgrid(*var)).T.reshape(2**n, n)
        S = 0
        for start in starts:
            sel = tuple(np.s_[s:s+sd] for s, sd in zip(start, size_dp))
            S += np.sum(values[sel]*dp)
        S /= 2**n
        return S
    if dv is None:
        dv = 1
    return np.sum(values) * dv

class ContinuousIMSE(object):
    def __init__(self, model, initial_method = None, max_sampling = 3000, grid_size = None):
        from . import bandwidths
        test_model = model.copy()
        if initial_method is None:
            test_model.bandwidth = bandwidths.scotts_bandwidth
        else:
            test_model.bandwidth = initial_method
        test_est = test_model.fit()
        #test_est.kernel = test_est.kernel.convolution

        LOO_model = model.copy()
        LOO_model.exog = np.array(model.exog[1:])
        LOO_model.bandwidth = test_est.bandwidth
        if model.weights.ndim > 0:
            LOO_model.weights = np.array(model.weights[1:])
        if model.adjust.ndim > 0:
            LOO_model.adjust = np.array(model.adjust[1:])
        LOO_est = LOO_model.fit()
        min_bw = np.min(test_est.bandwidth)*1e-9

        self.LOO = LeaveOut(test_est.exog, test_est.weights, test_est.adjust, sampling=max_sampling)
        self.bw_min = test_est.bandwidth * 1e-3
        self.test_est = test_est
        self.LOO_est = LOO_est
        self.grid_size = grid_size

    @property
    def init_bandwidth(self):
        return self.test_est.bandwidth

    def __call__(self, bw):
        if np.any(bw <= self.bw_min):
            return np.inf
        LOO_est = self.LOO_est
        test_est = self.test_est

        LOO_est.bandwidth = test_est.bandwidth = bw
        exog = test_est.exog
        npts = test_est.npts
        Fx, Fy = test_est.grid(N=self.grid_size)
        F = integrate_grid(Fy**2, grid=Fx)
        L = 0
        for i, Xi, Wi, Li in self.LOO:
            LOO_est.exog = Xi
            LOO_est.weights = Wi
            LOO_est.adjust = Li
            L += LOO_est.pdf(exog[i])
        return F - 2*L / len(self.LOO)

class leastsquare_cv_bandwidth(object):
    """
    Implement the Cross-Validation Least Square bandwidth estimation method.

    Notes
    -----
    For more details see pp. 16, 27 in Ref. [1] (see module docstring).

    Returns the value of the bandwidth that maximizes the integrated mean
    square error between the estimated and actual distribution.  The
    integrated mean square error (IMSE) is given by:

    .. math:: \int\left[\hat{f}(x)-f(x)\right]^{2}dx

    This is the general formula for the IMSE.

    Attributes
    ----------
    initial_method: fun
        Method used to get the initial estimate for the bandwidth

    max_sampling: int
        Maximum number of samples taken for the leave-one-out IMSE estimator. If None, all the points will be evaluated, 
        always.
    """

    def __init__(self, imse = None, imse_args = {}):
        if imse is None:
            self.imse = ContinuousIMSELOO
        else:
            self.imse = imse
        self.imse_args = imse_args
        self.max_sampling = 3000

    def __call__(self, model):
        imse = self.imse(model, **self.imse_args)
        bw = optimize.fmin(imse, x0=imse.init_bandwidth, maxiter=1e3, maxfun=1e3, disp=0, xtol=1e-3)
        return bw

