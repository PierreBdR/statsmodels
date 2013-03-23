# -*- coding: utf-8 -*-
"""

Created on Sat Mar 23 13:34:19 2013

Author: Josef Perktold
"""

    from numpy.testing import assert_allclose, assert_equal, assert_raises

    cases = [
        (0, {}),
        (50, {}),
        (-50, {}),
        (500000, dict(low=10000)),
        (-50000, dict(upp=-1000)),
        (500000, dict(low=300000, upp=700000)),
        (-50000, dict(low= -70000, upp=-1000))
        ]

    funcs = [(func, None),
             (func, True),
             (funcn, None),
             (funcn, False)]

    for f, inc in funcs:
        for a, kwds in cases:
            kw = {'increasing':inc}
            kw.update(kwds)
            res = brentq_expanding(f, args=(a,), **kwds)
            print '%10d'%a, ['dec', 'inc'][f is func], res - a
            assert_allclose(res, a, rtol=1e-5)

    # wrong sign for start bounds
    # doesn't raise yet during development TODO: activate this
    # it kind of works in some cases, but not correctly or in a useful way
    #assert_raises(ValueError, brentq_expanding, func, args=(-500,), start_upp=-1000)
    #assert_raises(ValueError, brentq_expanding, func, args=(500,), start_low=1000)

    # low upp given, but doesn't bound root, leave brentq exception
    # ValueError: f(a) and f(b) must have different signs
    assert_raises(ValueError, brentq_expanding, funcn, args=(-50000,), low= -40000, upp=-10000)

    # max_it too low to find root bounds
    # ValueError: f(a) and f(b) must have different signs
    assert_raises(ValueError, brentq_expanding, func, args=(-50000,), max_it=2)

    # maxiter_bq too low
    # RuntimeError: Failed to converge after 3 iterations.
    assert_raises(RuntimeError, brentq_expanding, func, args=(-50000,), maxiter_bq=3)

    # cannot determin whether increasing, all 4 low trial points return nan
    assert_raises(ValueError, brentq_expanding, func_nan, args=(-20, 0.6))

    # test for full_output
    a = 500
    val, info = brentq_expanding(func, args=(a,), full_output=True)
    assert_allclose(val, a, rtol=1e-5)
    info1 = {'iterations': 63, 'start_bounds': (-1, 1),
             'brentq_bounds': (100, 1000), 'flag': 'converged',
             'function_calls': 64, 'iterations_expand': 3, 'converged': True}
    for k in info1:
        assert_equal(info1[k], info.__dict__[k])

    assert_allclose(info.root, a, rtol=1e-5)
