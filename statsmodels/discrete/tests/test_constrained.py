# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:22:29 2014

Author: Josef Perktold
License: BSD-3

"""

import StringIO

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import patsy

from statsmodels.discrete.discrete_model import Poisson
import statsmodels.base._constraints as monkey

from .results import results_poisson_constrained as results

ss='''\
agecat	smokes	deaths	pyears
1	1	32	52407
2	1	104	43248
3	1	206	28612
4	1	186	12663
5	1	102	5317
1	0	2	18790
2	0	12	10673
3	0	28	5710
4	0	28	2585
5	0	31	1462'''

data = pd.read_csv(StringIO.StringIO(ss), delimiter='\t')
data['logpyears'] = np.log(data['pyears'])


class CheckPoissonConstrainedMixin(object):

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1[0], res2.params[self.idx], rtol=1e-6)
        assert_allclose(res1[1], res2.bse[self.idx], rtol=1e-6)


class TestPoissonConstrained1a(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):

        cls.res2 = results.results_noexposure_constraint
        cls.idx = [7, 3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data)
        #res1a = mod1a.fit()
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint('C(agecat)[T.4] = C(agecat)[T.5]')
        cls.res1 = mod.fit_constrained(lc.coefs, lc.constants,
                                       fit_kwds={'method':'bfgs'})
        # TODO: Newton fails

class TestPoissonConstrained1b(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):

        cls.res2 = results.results_exposure_constraint
        #cls.idx = [3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical
        cls.idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data,
                                   exposure=data['pyears'].values)
                                   #offset=np.log(data['pyears'].values))
        #res1a = mod1a.fit()
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint('C(agecat)[T.4] = C(agecat)[T.5]')
        cls.res1 = mod.fit_constrained(lc.coefs, lc.constants,
                                       fit_kwds={'method':'newton'})
        cls.constraints = lc
        # TODO: bfgs fails

class TestPoissonConstrained1c(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):

        cls.res2 = results.results_exposure_constraint
        #cls.idx = [3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical
        cls.idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data,
                                   offset=np.log(data['pyears'].values))
        #res1a = mod1a.fit()
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint('C(agecat)[T.4] = C(agecat)[T.5]')
        cls.res1 = mod.fit_constrained(lc.coefs, lc.constants,
                                       fit_kwds={'method':'newton'})
        cls.constraints = lc
        # TODO: bfgs fails


class TestPoissonNoConstrained(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):

        cls.res2 = results.results_exposure_noconstraint
        cls.idx = [6, 2, 3, 4, 5, 0] # 1 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data,
                                   #exposure=data['pyears'].values)
                                   offset=np.log(data['pyears'].values))
        res1a = mod.fit()._results
        cls.res1 = (res1a.params, res1a.bse)


class TestPoissonConstrained2a(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):

        cls.res2 = results.results_noexposure_constraint2
        cls.idx = [7, 3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data)

        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = mod.fit_constrained(lc.coefs, lc.constants,
                                       fit_kwds={'method':'bfgs'})
        # TODO: Newton fails

class TestPoissonConstrained2b(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):

        cls.res2 = results.results_exposure_constraint2
        #cls.idx = [3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical
        cls.idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data,
                                   exposure=data['pyears'].values)
                                   #offset=np.log(data['pyears'].values))
        #res1a = mod1a.fit()
        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = mod.fit_constrained(lc.coefs, lc.constants,
                                       fit_kwds={'method':'newton'})
        cls.constraints = lc
        # TODO: bfgs fails

class TestPoissonConstrained2c(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):

        cls.res2 = results.results_exposure_constraint2
        #cls.idx = [3, 4, 5, 6, 0, 1]  # 2 is dropped baseline for categorical
        cls.idx = [6, 2, 3, 4, 5, 0]  # 2 is dropped baseline for categorical

        # example without offset
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data,
                                   offset=np.log(data['pyears'].values))

        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = mod.fit_constrained(lc.coefs, lc.constants,
                                       fit_kwds={'method':'newton'})
        cls.constraints = lc
        # TODO: bfgs fails


def junk():
    # Singular Matrix in mod1a.fit()

    formula1 = 'deaths ~ smokes + C(agecat)'

    formula2 = 'deaths ~ C(agecat) + C(smokes) : C(agecat)'  # same as Stata default

    mod = Poisson.from_formula(formula2, data=data, exposure=data['pyears'].values)

    res0 = mod.fit()

    constraints = 'C(smokes)[T.1]:C(agecat)[3] = C(smokes)[T.1]:C(agecat)[4]'

    import patsy
    lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constraints)
    R, q = lc.coefs, lc.constants

    resc = mod.fit_constrained(R,q, fit_kwds={'method':'bfgs'})

    # example without offset
    formula1a = 'deaths ~ logpyears + smokes + C(agecat)'
    mod1a = Poisson.from_formula(formula1a, data=data)
    print mod1a.exog.shape

    res1a = mod1a.fit()
    lc_1a = patsy.DesignInfo(mod1a.exog_names).linear_constraint('C(agecat)[T.4] = C(agecat)[T.5]')
    resc1a = mod1a.fit_constrained(lc_1a.coefs, lc_1a.constants, fit_kwds={'method':'newton'})
    print resc1a[0]
    print resc1a[1]
