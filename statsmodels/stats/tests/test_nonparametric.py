# -*- coding: utf-8 -*-
"""

Created on Fri Jul 05 14:05:24 2013

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from statsmodels.sandbox.stats.runs import (mcnemar, cochrans_q, Runs,
                                            symmetry_bowker,
                                            runstest_1samp, runstest_2samp)

def _expand_table(table):
    '''expand a 2 by 2 contingency table to observations
    '''
    return np.repeat([[1, 1], [1, 0], [0, 1], [0, 0]], table.ravel(), axis=0)


def test_mcnemar_exact():
    f_obs1 = np.array([[101, 121], [59, 33]])
    f_obs2 = np.array([[101,  70], [59, 33]])
    f_obs3 = np.array([[101,  80], [59, 33]])
    f_obs4 = np.array([[101,  30], [60, 33]])
    f_obs5 = np.array([[101,  10], [30, 33]])
    f_obs6 = np.array([[101,  10], [10, 33]])

    #vassar college online computation
    res1 = 0.000004
    res2 = 0.378688
    res3 = 0.089452
    res4 = 0.00206
    res5 = 0.002221
    res6 = 1.

    assert_almost_equal(mcnemar(f_obs1, exact=True), [59, res1], decimal=6)
    assert_almost_equal(mcnemar(f_obs2, exact=True), [59, res2], decimal=6)
    assert_almost_equal(mcnemar(f_obs3, exact=True), [59, res3], decimal=6)
    assert_almost_equal(mcnemar(f_obs4, exact=True), [30, res4], decimal=6)
    assert_almost_equal(mcnemar(f_obs5, exact=True), [10, res5], decimal=6)
    assert_almost_equal(mcnemar(f_obs6, exact=True), [10, res6], decimal=6)

    x, y = _expand_table(f_obs2).T  # tuple unpack
    assert_allclose(mcnemar(f_obs2, exact=True),
                    mcnemar(x, y, exact=True), rtol=1e-13)


def test_mcnemar_chisquare():
    f_obs1 = np.array([[101, 121], [59, 33]])
    f_obs2 = np.array([[101,  70], [59, 33]])
    f_obs3 = np.array([[101,  80], [59, 33]])

    #> mcn = mcnemar.test(matrix(c(101, 121,  59,  33),nrow=2))
    res1 = [2.067222e01, 5.450095e-06]
    res2 = [0.7751938,    0.3786151]
    res3 = [2.87769784,   0.08981434]

    assert_allclose(mcnemar(f_obs1, exact=False), res1, rtol=1e-6)
    assert_allclose(mcnemar(f_obs2, exact=False), res2, rtol=1e-6)
    assert_allclose(mcnemar(f_obs3, exact=False), res3, rtol=1e-6)

    # compare table versus observations
    x, y = _expand_table(f_obs2).T  # tuple unpack
    assert_allclose(mcnemar(f_obs2, exact=False),
                    mcnemar(x, y, exact=False), rtol=1e-13)

    # test correction = False
    res1 = [2.135556e01, 3.815136e-06]
    res2 = [0.9379845,   0.3327967]
    res3 = [3.17266187,  0.07488031]

    res = mcnemar(f_obs1, exact=False, correction=False)
    assert_allclose(res, res1, rtol=1e-6)
    res = mcnemar(f_obs2, exact=False, correction=False)
    assert_allclose(res, res2, rtol=1e-6)
    res = mcnemar(f_obs3, exact=False, correction=False)
    assert_allclose(res, res3, rtol=1e-6)


def test_mcnemar_vectorized():
    ttk = np.random.randint(5,15, size=(2,2,3))
    mcnemar(ttk)
    res = mcnemar(ttk, exact=False)
    res1 = zip(*[mcnemar(ttk[:,:,i], exact=False) for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)

    res = mcnemar(ttk, exact=False, correction=False)
    res1 = zip(*[mcnemar(ttk[:,:,i], exact=False, correction=False)
                                                          for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)

    res = mcnemar(ttk, exact=True)
    res1 = zip(*[mcnemar(ttk[:,:,i], exact=True) for i in range(3)])
    assert_allclose(res, res1, rtol=1e-13)


def test_symmetry_bowker():
    table = np.array([0, 3, 4, 4, 2, 4, 1, 2, 4, 3, 5, 3, 0, 0, 2, 2, 3, 0, 0,
                      1, 5, 5, 5, 5, 5]).reshape(5, 5)

    res = symmetry_bowker(table)
    mcnemar5_1 = dict(statistic=7.001587, pvalue=0.7252951, parameters=(10,),
                      distr='chi2')
    assert_allclose(res[:2], [mcnemar5_1['statistic'], mcnemar5_1['pvalue']],
                    rtol=1e-7)

    res = symmetry_bowker(1 + table)
    mcnemar5_1b = dict(statistic=5.355988, pvalue=0.8661652, parameters=(10,),
                       distr='chi2')
    assert_allclose(res[:2], [mcnemar5_1b['statistic'], mcnemar5_1b['pvalue']],
                    rtol=1e-7)


    table = np.array([2, 2, 3, 6, 2, 3, 4, 3, 6, 6, 6, 7, 1, 9, 6, 7, 1, 1, 9,
                      8, 0, 1, 8, 9, 4]).reshape(5, 5)

    res = symmetry_bowker(table)
    mcnemar5_2 = dict(statistic=18.76432, pvalue=0.04336035, parameters=(10,),
                      distr='chi2')
    assert_allclose(res[:2], [mcnemar5_2['statistic'], mcnemar5_2['pvalue']],
                    rtol=1.5e-7)

    res = symmetry_bowker(1 + table)
    mcnemar5_2b = dict(statistic=14.55256, pvalue=0.1492461, parameters=(10,),
                       distr='chi2')
    assert_allclose(res[:2], [mcnemar5_2b['statistic'], mcnemar5_2b['pvalue']],
                    rtol=1e-7)


def test_cochransq():
    #example from dataplot docs, Conovover p. 253
    #http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/cochran.htm
    x = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [0, 1, 0],
                   [1, 1, 0],
                   [0, 0, 0],
                   [1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [1, 1, 1],
                   [1, 1, 1]])
    res_qstat = 2.8
    res_pvalue = 0.246597
    assert_almost_equal(cochrans_q(x), [res_qstat, res_pvalue])

    #equivalence of mcnemar and cochranq for 2 samples
    a,b = x[:,:2].T
    assert_almost_equal(mcnemar(a,b, exact=False, correction=False),
                        cochrans_q(x[:,:2]))


def test_cochransq2():
    # from an example found on web, verifies 13.286
    data = np.array('''
        0 0 0 1
        0 0 0 1
        0 0 0 1
        1 1 1 1
        1 0 0 1
        0 1 0 1
        1 0 0 1
        0 0 0 1
        0 1 0 0
        0 0 0 0
        1 0 0 1
        0 0 1 1'''.split(), int).reshape(-1, 4)

    res = cochrans_q(data)
    assert_allclose(res, [13.2857143, 0.00405776], rtol=1e-6)


def test_cochransq3():
    # another example compared to SAS
    # in frequency weight format
    dt = [('A', 'S1'), ('B', 'S1'), ('C', 'S1'), ('count', int)]
    dta = np.array([('F', 'F', 'F', 6),
                    ('U', 'F', 'F', 2),
                    ('F', 'F', 'U', 16),
                    ('U', 'F', 'U', 4),
                    ('F', 'U', 'F', 2),
                    ('U', 'U', 'F', 6),
                    ('F', 'U', 'U', 4),
                    ('U', 'U', 'U', 6)], dt)

    cases = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 0, 1],
                      [1, 0, 1],
                      [0, 1, 0],
                      [1, 1, 0],
                      [0, 1, 1],
                      [1, 1, 1]])
    count = np.array([ 6,  2, 16,  4,  2,  6,  4,  6])
    data = np.repeat(cases, count, 0)

    res = cochrans_q(data)
    assert_allclose(res, [8.4706, 0.0145], atol=5e-5)

def test_runstest():
    #comparison numbers from R, tseries, runs.test
    #currently only 2-sided used
    x = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1])

    z_twosided = 1.386750
    pvalue_twosided = 0.1655179

    z_greater = 1.386750
    pvalue_greater = 0.08275893

    z_less = 1.386750
    pvalue_less = 0.917241

    #print Runs(x).runs_test(correction=False)
    assert_almost_equal(np.array(Runs(x).runs_test(correction=False)),
                        [z_twosided, pvalue_twosided], decimal=6)

def test_runstest_2sample():
    # regression test, checked with MonteCarlo and looks reasonable

    x = [31.8, 32.8, 39.2, 36, 30, 34.5, 37.4]
    y = [35.5, 27.6, 21.3, 24.8, 36.7, 30]
    y[-1] += 1e-6  #avoid tie that creates warning
    groups = np.concatenate((np.zeros(len(x)), np.ones(len(y))))

    res = runstest_2samp(x, y)
    res1 = (0.022428065200812752, 0.98210649318649212)
    assert_allclose(res, res1, rtol=1e-6)

    # check as stacked array
    res2 = runstest_2samp(x, y)
    assert_allclose(res2, res, rtol=1e-6)

    xy = np.concatenate((x, y))
    res_1s = runstest_1samp(xy)
    assert_allclose(res_1s, res1, rtol=1e-6)
    # check cutoff
    res2_1s = runstest_1samp(xy, xy.mean())
    assert_allclose(res2_1s, res_1s, rtol=1e-6)
