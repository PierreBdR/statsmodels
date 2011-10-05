"""
Test VAR Model
"""
from __future__ import with_statement

# pylint: disable=W0612,W0231

from cStringIO import StringIO
from nose.tools import assert_raises
import nose
import os
import sys

import numpy as np

import scikits.statsmodels.api as sm
import scikits.statsmodels.tsa.vector_ar.var_model as model
import scikits.statsmodels.tsa.vector_ar.util as util
import scikits.statsmodels.tools.data as data_util
#reload(model)
from scikits.statsmodels.tsa.vector_ar.var_model import VAR

from numpy.testing import assert_almost_equal, assert_equal

DECIMAL_12 = 12
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2

class CheckVAR(object):
    # just so pylint won't complain
    res1 = None
    res2 = None

    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_3)

    def test_neqs(self):
        assert_equal(self.res1.neqs, self.res2.neqs)

    def test_nobs(self):
        assert_equal(self.res1.avobs, self.res2.nobs)

    def test_df_eq(self):
        assert_equal(self.res1.df_eq, self.res2.df_eq)

    def test_rmse(self):
        results = self.res1.results
        for i in range(len(results)):
            assert_almost_equal(results[i].mse_resid**.5,
                    eval('self.res2.rmse_'+str(i+1)), DECIMAL_6)

    def test_rsquared(self):
        results = self.res1.results
        for i in range(len(results)):
            assert_almost_equal(results[i].rsquared,
                    eval('self.res2.rsquared_'+str(i+1)), DECIMAL_3)

    def test_llf(self):
        results = self.res1.results
        assert_almost_equal(self.res1.llf, self.res2.llf, DECIMAL_2)
        for i in range(len(results)):
            assert_almost_equal(results[i].llf,
                    eval('self.res2.llf_'+str(i+1)), DECIMAL_2)

    def test_aic(self):
        assert_almost_equal(self.res1.aic, self.res2.aic)

    def test_bic(self):
        assert_almost_equal(self.res1.bic, self.res2.bic)

    def test_hqic(self):
        assert_almost_equal(self.res1.hqic, self.res2.hqic)

    def test_fpe(self):
        assert_almost_equal(self.res1.fpe, self.res2.fpe)

    def test_detsig(self):
        assert_almost_equal(self.res1.detomega, self.res2.detsig)

    def test_bse(self):
        assert_almost_equal(self.res1.bse, self.res2.bse, DECIMAL_4)

def get_macrodata():
    data = sm.datasets.macrodata.load().data[['realgdp','realcons','realinv']]
    names = data.dtype.names
    nd = data.view((float,3))
    nd = np.diff(np.log(nd), axis=0)
    return nd.ravel().view(data.dtype)

def generate_var():
    from rpy2.robjects import r
    import pandas.rpy.common as prp
    r.source('tests/var.R')
    return prp.convert_robj(r['result'], use_pandas=False)

def write_generate_var():
    result = generate_var()
    np.savez('tests/results/vars_results.npz', **result)

class RResults(object):
    """
    Simple interface with results generated by "vars" package in R.
    """

    def __init__(self):
        #data = np.load(resultspath + 'vars_results.npz')
        from results.results_var_data import var_results
        data = var_results.__dict__

        self.names = data['coefs'].dtype.names
        self.params = data['coefs'].view((float, len(self.names)))
        self.stderr = data['stderr'].view((float, len(self.names)))

        self.irf = data['irf'].item()
        self.orth_irf = data['orthirf'].item()

        self.nirfs = int(data['nirfs'][0])
        self.nobs = int(data['obs'][0])
        self.totobs = int(data['totobs'][0])

        crit = data['crit'].item()
        self.aic = crit['aic'][0]
        self.sic = self.bic = crit['sic'][0]
        self.hqic = crit['hqic'][0]
        self.fpe = crit['fpe'][0]

        self.detomega = data['detomega'][0]
        self.loglike = data['loglike'][0]

        self.nahead = int(data['nahead'][0])
        self.ma_rep = data['phis']

        self.causality = data['causality']

def close_plots():
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass

_orig_stdout = None

def setup_module():
    global _orig_stdout
    _orig_stdout = sys.stdout
    sys.stdout = StringIO()

def teardown_module():
    sys.stdout = _orig_stdout
    close_plots()

def have_matplotlib():
    try:
        import matplotlib
        if matplotlib.__version__ < '1':
            raise
        return True
    except:
        return False

class CheckIRF(object):

    ref = None; res = None; irf = None
    k = None

    #---------------------------------------------------------------------------
    # IRF tests

    def test_irf_coefs(self):
        self._check_irfs(self.irf.irfs, self.ref.irf)
        self._check_irfs(self.irf.orth_irfs, self.ref.orth_irf)

    def _check_irfs(self, py_irfs, r_irfs):
        for i, name in enumerate(self.res.names):
            ref_irfs = r_irfs[name].view((float, self.k))
            res_irfs = py_irfs[:, :, i]
            assert_almost_equal(ref_irfs, res_irfs)

    def test_plot_irf(self):
        if not have_matplotlib():
            raise nose.SkipTest

        self.irf.plot()
        self.irf.plot(plot_stderr=False)

        self.irf.plot(impulse=0, response=1)
        self.irf.plot(impulse=0)
        self.irf.plot(response=0)

        self.irf.plot(orth=True)
        self.irf.plot(impulse=0, response=1, orth=True)
        close_plots()

    def test_plot_cum_effects(self):
        if not have_matplotlib():
            raise nose.SkipTest

        self.irf.plot_cum_effects()
        self.irf.plot_cum_effects(plot_stderr=False)
        self.irf.plot_cum_effects(impulse=0, response=1)

        self.irf.plot_cum_effects(orth=True)
        self.irf.plot_cum_effects(impulse=0, response=1, orth=True)
        close_plots()

class CheckFEVD(object):

    fevd = None

    #---------------------------------------------------------------------------
    # FEVD tests

    def test_fevd_plot(self):
        if not have_matplotlib():
            raise nose.SkipTest

        self.fevd.plot()
        close_plots()

    def test_fevd_repr(self):
        print self.fevd

    def test_fevd_summary(self):
        self.fevd.summary()

    def test_fevd_cov(self):
        # test does not crash
        # not implemented
        # covs = self.fevd.cov()

        pass

class TestVARResults(CheckIRF, CheckFEVD):

    @classmethod
    def setupClass(cls):
        cls.p = 2

        cls.data = get_macrodata()
        cls.model = VAR(cls.data)
        cls.names = cls.model.names

        cls.ref = RResults()
        cls.k = len(cls.ref.names)
        cls.res = cls.model.fit(maxlags=cls.p)

        cls.irf = cls.res.irf(cls.ref.nirfs)
        cls.nahead = cls.ref.nahead

        cls.fevd = cls.res.fevd()

    def test_constructor(self):
        # make sure this works with no names
        ndarr = self.data.view((float, 3))
        model = VAR(ndarr)
        res = model.fit(self.p)

    def test_names(self):
        assert_equal(self.model.names, self.ref.names)

        model2 = VAR(self.data, names=self.names)
        assert_equal(model2.names, self.ref.names)

    def test_get_eq_index(self):
        assert(isinstance(self.res.names, list))

        for i, name in enumerate(self.names):
            idx = self.res.get_eq_index(i)
            idx2 = self.res.get_eq_index(name)

            assert_equal(idx, i)
            assert_equal(idx, idx2)

        assert_raises(Exception, self.res.get_eq_index, 'foo')

    def test_repr(self):
        # just want this to work
        foo = str(self.res)
        bar = repr(self.res)

    def test_params(self):
        assert_almost_equal(self.res.params, self.ref.params, DECIMAL_3)

    def test_cov_params(self):
        # do nothing for now
        self.res.cov_params

    def test_cov_ybar(self):
        self.res.cov_ybar()

    def test_tstat(self):
        self.res.tvalues

    def test_pvalues(self):
        self.res.pvalues

    def test_summary(self):
        summ = self.res.summary()
        print summ

    def test_detsig(self):
        assert_almost_equal(self.res.detomega, self.ref.detomega)

    def test_aic(self):
        assert_almost_equal(self.res.aic, self.ref.aic)

    def test_bic(self):
        assert_almost_equal(self.res.bic, self.ref.bic)

    def test_hqic(self):
        assert_almost_equal(self.res.hqic, self.ref.hqic)

    def test_fpe(self):
        assert_almost_equal(self.res.fpe, self.ref.fpe)

    def test_lagorder_select(self):
        ics = ['aic', 'fpe', 'hqic', 'bic']

        for ic in ics:
            res = self.model.fit(maxlags=10, ic=ic, verbose=True)

        assert_raises(Exception, self.model.fit, ic='foo')

    def test_nobs(self):
        assert_equal(self.res.nobs, self.ref.nobs)

    def test_stderr(self):
        assert_almost_equal(self.res.stderr, self.ref.stderr, DECIMAL_4)

    def test_loglike(self):
        assert_almost_equal(self.res.llf, self.ref.loglike)

    def test_ma_rep(self):
        ma_rep = self.res.ma_rep(self.nahead)
        assert_almost_equal(ma_rep, self.ref.ma_rep)

    #--------------------------------------------------
    # Lots of tests to make sure stuff works...need to check correctness

    def test_causality(self):
        causedby = self.ref.causality['causedby']

        for i, name in enumerate(self.names):
            variables = self.names[:i] + self.names[i + 1:]
            result = self.res.test_causality(name, variables, kind='f')
            assert_almost_equal(result['pvalue'], causedby[i], DECIMAL_4)

            rng = range(self.k)
            rng.remove(i)
            result2 = self.res.test_causality(i, rng, kind='f')
            assert_almost_equal(result['pvalue'], result2['pvalue'], DECIMAL_12)

            # make sure works
            result = self.res.test_causality(name, variables, kind='wald')

        # corner cases
        _ = self.res.test_causality(self.names[0], self.names[1])
        _ = self.res.test_causality(0, 1)

        assert_raises(Exception,self.res.test_causality, 0, 1, kind='foo')

    def test_select_order(self):
        result = self.model.fit(10, ic='aic', verbose=True)
        result = self.model.fit(10, ic='fpe', verbose=True)

        # bug
        model = VAR(self.model.endog)
        model.select_order()

    def test_is_stable(self):
        # may not necessarily be true for other datasets
        assert(self.res.is_stable(verbose=True))

    def test_acf(self):
        # test that it works...for now
        acfs = self.res.acf(10)

        # defaults to nlags=lag_order
        acfs = self.res.acf()
        assert(len(acfs) == self.p + 1)

    def test_acorr(self):
        acorrs = self.res.acorr(10)

    def test_forecast(self):
        point = self.res.forecast(self.res.y[-5:], 5)

    def test_forecast_interval(self):
        y = self.res.y[:-self.p:]
        point, lower, upper = self.res.forecast_interval(y, 5)

    def test_plot_sim(self):
        if not have_matplotlib():
            raise nose.SkipTest

        self.res.plotsim(steps=100)
        close_plots()

    def test_plot(self):
        if not have_matplotlib():
            raise nose.SkipTest

        self.res.plot()
        close_plots()

    def test_plot_acorr(self):
        if not have_matplotlib():
            raise nose.SkipTest

        self.res.plot_acorr()
        close_plots()

    def test_plot_forecast(self):
        if not have_matplotlib():
            raise nose.SkipTest

        self.res.plot_forecast(5)
        close_plots()

    def test_reorder(self):
        #manually reorder
        data = self.data.view((float,3))
        names = self.names
        data2 = np.append(np.append(data[:,2,None], data[:,0,None], axis=1), data[:,1,None], axis=1)
        names2 = []
        names2.append(names[2])
        names2.append(names[0])
        names2.append(names[1])
        res2 = VAR(data2,names=names2).fit(maxlags=self.p)

        #use reorder function
        res3 = self.res.reorder(['realinv','realgdp', 'realcons'])

        #check if the main results match
        assert_almost_equal(res2.params, res3.params)
        assert_almost_equal(res2.sigma_u, res3.sigma_u)
        assert_almost_equal(res2.bic, res3.bic)
        assert_almost_equal(res2.stderr, res3.stderr)

class E1_Results(object):
    """
    Results from Lutkepohl (2005) using E2 dataset
    """

    def __init__(self):
        # Lutkepohl p. 120 results

        # I asked the author about these results and there is probably rounding
        # error in the book, so I adjusted these test results to match what is
        # coming out of the Python (double-checked) calculations
        self.irf_stderr = np.array([[[.125, 0.546, 0.664 ],
                                     [0.032, 0.139, 0.169],
                                     [0.026, 0.112, 0.136]],

                                    [[0.129, 0.547, 0.663],
                                     [0.032, 0.134, 0.163],
                                     [0.026, 0.108, 0.131]],

                                    [[0.084, .385, .479],
                                     [.016, .079, .095],
                                     [.016, .078, .103]]])

        self.cum_irf_stderr = np.array([[[.125, 0.546, 0.664 ],
                                         [0.032, 0.139, 0.169],
                                         [0.026, 0.112, 0.136]],

                                        [[0.149, 0.631, 0.764],
                                         [0.044, 0.185, 0.224],
                                         [0.033, 0.140, 0.169]],

                                        [[0.099, .468, .555],
                                         [.038, .170, .205],
                                         [.033, .150, .185]]])

        self.lr_stderr = np.array([[.134, .645, .808],
                                   [.048, .230, .288],
                                   [.043, .208, .260]])

basepath = os.path.split(sm.__file__)[0]
resultspath = basepath + '/tsa/vector_ar/tests/results/'

def get_lutkepohl_data(name='e2'):
    lut_data = basepath + '/tsa/vector_ar/data/'
    path = lut_data + '%s.dat' % name

    return util.parse_lutkepohl_data(path)

def have_pandas():
    try:
        import pandas as _
        return True
    except ImportError:
        return False

def test_lutkepohl_parse():
    files = ['e%d' % i for i in range(1, 7)]

    if not have_pandas():
        raise nose.SkipTest

    for f in files:
        get_lutkepohl_data(f)

class TestVARResultsLutkepohl(object):
    """
    Verify calculations using results from Lutkepohl's book
    """

    def __init__(self):
        self.p = 2

        if not have_pandas():
            return

        sdata, dates = get_lutkepohl_data('e1')

        names = sdata.dtype.names
        data = data_util.struct_to_ndarray(sdata)
        adj_data = np.diff(np.log(data), axis=0)
        # est = VAR(adj_data, p=2, dates=dates[1:], names=names)

        self.model = VAR(adj_data[:-16], dates=dates[1:-16], names=names,
                freq='Q')
        self.res = self.model.fit(maxlags=self.p)
        self.irf = self.res.irf(10)
        self.lut = E1_Results()

    def test_approx_mse(self):
        if not have_pandas():
            raise nose.SkipTest

        # 3.5.18, p. 99
        mse2 = np.array([[25.12, .580, 1.300],
                         [.580, 1.581, .586],
                         [1.300, .586, 1.009]]) * 1e-4

        assert_almost_equal(mse2, self.res.forecast_cov(3)[1],
                            DECIMAL_3)

    def test_irf_stderr(self):
        if not have_pandas():
            raise nose.SkipTest

        irf_stderr = self.irf.stderr(orth=False)
        for i in range(1, 1 + len(self.lut.irf_stderr)):
            assert_almost_equal(np.round(irf_stderr[i], 3),
                                self.lut.irf_stderr[i-1])

    def test_cum_irf_stderr(self):
        if not have_pandas():
            raise nose.SkipTest

        stderr = self.irf.cum_effect_stderr(orth=False)
        for i in range(1, 1 + len(self.lut.cum_irf_stderr)):
            assert_almost_equal(np.round(stderr[i], 3),
                                self.lut.cum_irf_stderr[i-1])

    def test_lr_effect_stderr(self):
        if not have_pandas():
            raise nose.SkipTest

        stderr = self.irf.lr_effect_stderr(orth=False)
        orth_stderr = self.irf.lr_effect_stderr(orth=True)
        assert_almost_equal(np.round(stderr, 3), self.lut.lr_stderr)

def test_get_trendorder():
    results = {
        'c' : 1,
        'nc' : 0,
        'ct' : 2,
        'ctt' : 3
    }

    for t, trendorder in results.iteritems():
        assert(util.get_trendorder(t) == trendorder)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)
