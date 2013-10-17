"""
Variance functions for use with the link functions in statsmodels.family.links
"""

__docformat__ = 'restructuredtext'

import numpy as np
FLOAT_EPS = np.finfo(float).eps

class VarianceFunction(object):
    """
    Relates the variance of a random variable to its mean. Defaults to 1.

    Methods
    -------
    call
        Returns an array of ones that is the same shape as `mu`

    Notes
    -----
    After a variance function is initialized, its call method can be used.

    Alias for VarianceFunction:
    constant = VarianceFunction()

    See also
    --------
    statsmodels.family.family
    """

    def __call__(self, mu):
        """
        Default variance function

        Parameters
        -----------
        mu : array-like
            mean parameters

        Returns
        -------
        v : array
            ones(mu.shape)
        """
        mu = np.asarray(mu)
        return np.ones(mu.shape, np.float64)

constant = VarianceFunction()
constant.__doc__ = """
The call method of constnat returns a constant variance, ie., a vector of ones.

constant is an alias of VarianceFunction()
"""

class Power(object):
    """
    Power variance function

    Parameters
    ----------
    power : float
        exponent used in power variance function

    Methods
    -------
    call
        Returns the power variance

    Formulas
    --------
    V(mu) = numpy.fabs(mu)**power

    Notes
    -----
    Aliases for Power:
    mu = Power()
    mu_squared = Power(power=2)
    mu_cubed = Power(power=3)
    """

    def __init__(self, power=1.):
        self.power = power

    def __call__(self, mu):
        """
        Power variance function

        Parameters
        ----------
        mu : array-like
            mean parameters

        Returns
        -------
        variance : array
            numpy.fabs(mu)**self.power
        """
        return np.power(np.fabs(mu), self.power)

mu = Power()
mu.__doc__ = """
Returns np.fabs(mu)

Notes
-----
This is an alias of Power()
"""
mu_squared = Power(power=2)
mu_squared.__doc__ = """
Returns np.fabs(mu)**2

Notes
-----
This is an alias of statsmodels.family.links.Power(power=2)
"""
mu_cubed = Power(power=3)
mu_cubed.__doc__ = """
Returns np.fabs(mu)**3

Notes
-----
This is an alias of statsmodels.family.links.Power(power=3)
"""

class Binomial(object):
    """
    Binomial variance function

    Parameters
    ----------
    n : int, optional
        The number of trials for a binomial variable.  The default is 1 for
        p in (0,1)

    Methods
    -------
    call
        Returns the binomial variance

    Formulas
    --------
    V(mu) = p * (1 - p) * n

    where p = mu / n

    Notes
    -----
    Alias for Binomial:
    binary = Binomial()

    A private method _clean trims the data by machine epsilon so that p is
    in (0,1)
    """

    def __init__(self, n=1):
        self.n = n

    def _clean(self, p):
        return np.clip(p, FLOAT_EPS, 1 - FLOAT_EPS)

    def __call__(self, mu):
        """
        Binomial variance function

        Parameters
        -----------
        mu : array-like
            mean parameters

        Returns
        -------
        variance : array
           variance = mu/n * (1 - mu/n) * self.n
        """
        p = self._clean(mu / self.n)
        return p * (1 - p) * self.n

binary = Binomial()
binary.__doc__ = """
The binomial variance function for n = 1

Notes
-----
This is an alias of Binomial(n=1)
"""

class NegativeBinomial(object):
    '''
    Negative binomial variance function

    Parameters
    ----------
    alpha : float
        The ancillary parameter for the negative binomial variance function.
        `alpha` is assumed to be nonstochastic.  The default is 1.

    Methods
    -------
    call
        Returns the negative binomial variance

    Formulas
    --------
    V(mu) = mu + alpha*mu**2

    Notes
    -----
    Alias for NegativeBinomial:
    nbinom = NegativeBinomial()

    A private method _clean trims the data by machine epsilon so that p is
    in (0,inf)
    '''

    def __init__(self, alpha=1.):
        self.alpha = alpha

    def _clean(self, p):
        return np.clip(p, FLOAT_EPS, np.inf)

    def __call__(self, mu):
        """
        Negative binomial variance function

        Parameters
        ----------
        mu : array-like
            mean parameters

        Returns
        -------
        variance : array
            variance = mu + alpha*mu**2
        """
        p = self._clean(mu)
        return mu + self.alpha*mu**2

nbinom = NegativeBinomial()
nbinom.__doc__ = """
Negative Binomial variance function.

Notes
-----
This is an alias of NegativeBinomial(alpha=1.)
"""
