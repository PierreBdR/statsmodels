"""
Linear mixed effects models for Statsmodels

The data are partitioned into disjoint groups.  The probability model
for group i is:

Y = X*beta + Z*gamma + epsilon

where

* Y is a n_i dimensional response vector
* X is a n_i x k_fe dimensional matrix of fixed effects
  coefficients
* beta is a k_fe-dimensional vector of fixed effects slopes
* Z is a n_i x k_re dimensional matrix of random effects
  coefficients
* gamma is a k_re-dimensional random vector with mean 0
  and covariance matrix Psi; note that each group
  gets its own independent realization of gamma.
* epsilon is a n_i dimensional vector of iid normal
  errors with mean 0 and variance sigma^2; the epsilon
  values are independent both within and between groups

Y, X and Z must be entirely observed.  beta, Psi, and sigma^2 are
estimated using ML or REML estimation, and gamma and epsilon are
random so define the probability model.

The mean structure is E[Y|X,Z] = X*beta.  If only the mean structure
is of interest, GEE is a good alternative to mixed models.

The primary reference for the implementation details is:

MJ Lindstrom, DM Bates (1988).  "Newton Raphson and EM algorithms for
linear mixed effects models for repeated measures data".  Journal of
the American Statistical Association. Volume 83, Issue 404, pages
1014-1022.

All the likelihood, gradient, and Hessian calculations closely follow
Lindstrom and Bates.

The following two documents are written more from the perspective of
users:

http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

Notation:

* `cov_re` is the random effects covariance matrix (referred to above
  as Psi) and `sig2` is the (scalar) error variance.  For a single
  group, the marginal covariance matrix of endog given exog is sig2*I
  + Z * cov_re * Z', where Z is the design matrix for the random
  effects.

Notes:

1. Three different parameterizations are used here in different
places.  The regression slopes (usually called `fe_params`) are
identical in all three parameterizations, but the variance parameters
differ.  The parameterizations are:

* The "natural parameterization" in which cov(endog) = sig2*I + Z *
  cov_re * Z', as described above.  This is the main parameterization
  visible to the user.

* The "profile parameterization" in which cov(endog) = I +
  Z * cov_re1 * Z'.  This is the parameterization of the profile
  likelihood that is maximized to produce parameter estimates.
  (see Lindstrom and Bates for details).  The "natural" cov_re is
  equal to the "profile" cov_re1 times sig2.

* The "square root parameterization" in which we work with the
  Cholesky factor of cov_re1 instead of cov_re1 directly.

All three parameterizations can be "packed" by concatenating fe_params
together with the lower triangle of the dependence structure.  Note
that when unpacking, it is important to either square or reflect the
dependence structure depending on which parameterization is being
used.

2. The situation where the random effects covariance matrix is
singular is numerically challenging.  Small changes in the covariance
parameters may lead to large changes in the likelihood and
derivatives.

3. The optimization strategy is to optionally perform a few EM steps,
followed by optionally performing a few steepest descent steps,
followed by conjugate gradient descent using one of the scipy gradient
optimizers.  The EM and steepest descent steps are used to get
adequate starting values for the conjugate gradient optimization,
which is much faster.
"""

import numpy as np
import statsmodels.base.model as base
from scipy.optimize import fmin_ncg, fmin_cg, fmin_bfgs, fmin
from scipy.stats.distributions import norm
import pandas as pd
import patsy
#import collections  # OrderedDict requires python >= 2.7
from statsmodels.compatnp.collections import OrderedDict
import warnings
from statsmodels.tools.sm_exceptions import \
     ConvergenceWarning

# This is a global switch to use direct linear algebra calculations
# for solving factor-structured linear systems and calculating
# factor-structured determinants.  If False, use the
# Sherman-Morrison-Woodbury update which is more efficient for
# factor-structured matrices.  Should be False except when testing.
_no_smw = False

def _smw_solve(s, A, B, BI, rhs):
    """
    Solves the system (s*I + A*B*A') * x = rhs for x and returns x.

    Parameters:
    -----------
    s : scalar
        See above for usage
    A : square symmetric ndarray
        See above for usage
    B : square symmetric ndarray
        See above for usage
    BI : square symmetric ndarray
        The inverse of `B`
    rhs : ndarray
        See above for usage

    Returns:
    --------
    x : ndarray
        See above

    If the global variable `_no_smw` is True, this routine uses direct
    linear algebra calculations.  Otherwise it uses the
    Sherman-Morrison-Woodbury identity to speed up the calculation.
    """

    # Direct calculation
    if _no_smw or BI is None:
        mat = np.dot(A, np.dot(B, A.T))
        mat += s * np.eye(A.shape[0])
        return np.linalg.solve(mat, rhs)

    # Use SMW identity
    qmat = BI + np.dot(A.T, A) / s
    u = np.dot(A.T, rhs)
    qmat = np.linalg.solve(qmat, u)
    qmat = np.dot(A, qmat)
    rslt = rhs / s - qmat / s**2
    return rslt


def _smw_logdet(s, A, B, BI, B_logdet):
    """
    Use the matrix determinant lemma to accelerate the calculation of
    the log determinant of s*I + A*B*A'.

    Parameters:
    -----------
    s : scalar
        See above for usage
    A : square symmetric ndarray
        See above for usage
    B : square symmetric ndarray
        See above for usage
    BI : square symmetric ndarray
        The inverse of `B`.
    B_logdet : real
        The log determinant of B
    """

    if _no_smw or BI is None:
        mat = np.dot(A, np.dot(B, A.T))
        mat += s * np.eye(A.shape[0])
        _, ld = np.linalg.slogdet(mat)
        return ld

    p = A.shape[0]
    ld = p * np.log(s)

    qmat = BI + np.dot(A.T, A) / s
    _, ld1 = np.linalg.slogdet(qmat)

    return B_logdet + ld + ld1


class MixedLM(base.Model):
    """
    An object specifying a linear mixed effects model.  Use the `fit`
    method to fit the model and obtain a results object.

    Arguments:
    ----------
    endog : 1d array-like
        The dependent variable
    exog : 2d array-like
        A matrix of independent variables used to determine the
        mean structure
    groups : 1d array-like
        A vector of labels determining the groups -- data from
        different groups are independent
    exog_re : 2d array-like
        A matrix of independent variables used to determine the
        variance structure.  If None, defaults to a random
        intercept for each of the groups.  May also be set from
        a formula using a call to `set_random`.
    missing : string
        The approach to missing data handling

    Notes:
    ------
    The covariates in `exog` and `exog_re` may (but need not)
    partially or wholly overlap.
    """

    def __init__(self, endog, exog, groups, exog_re=None,
                 missing='none'):

        groups = np.asarray(groups)

        # If there is one covariate, it may be passed in as a column
        # vector, convert these to 2d arrays.
        if exog is not None and exog.ndim == 1:
            exog = exog[:,None]
        if exog_re is not None and exog_re.ndim == 1:
            exog_re = exog_re[:,None]

        self.exog_re_names = None
        if exog_re is None:
            # Default random effects structure (random intercepts).
            exog_re = np.ones((len(endog), 1), dtype=np.float64)
        else:
            try:
                self.exog_re_names = exog_re.columns
            except:
                pass
            self.exog_re_orig = exog_re
            self.exog_re = np.asarray(exog_re)

        # Calling super creates self.endog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(MixedLM, self).__init__(endog, exog, groups=groups,
                                  exog_re=exog_re, missing=missing)

        # Model dimensions
        self.k_fe = exog.shape[1] # Number of fixed effects parameters
        if exog_re is not None:

            # Number of random effect covariates
            self.k_re = exog_re.shape[1]

            # Number of covariance parameters
            self.k_re2 = self.k_re * (self.k_re + 1) // 2

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the groups.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        for i,g in enumerate(groups):
            row_indices[g].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.n_groups = len(self.group_labels)

        # Split the data by groups
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)

        # The total number of observations, summed over all groups
        self.n_totobs = sum([len(y) for y in self.endog_li])


    def set_random(self, re_formula, data):
        """
        Set the random effects structure using a formula.  This is an
        alternative to providing `exog_re` in the MixedLM constructor.

        Arguments:
        ----------
        re_formula : string
            A string defining the variance structure of the model
            as a formula.
        data : array-like
            The data referenced in re_formula.  Currently must be a
            Pandas DataFrame.

        Notes
        -----
        If the random effects structure is not set either by providing
        `exog_re` to the MixedLM constructor, or by calling
        `set_random`, then the default is a structure with random
        intercepts for the groups.

        This does not automatically drop missing values, so if
        `missing` is set to "drop" in the model construction, the
        missing values must be dropped from the data frame before
        calling this function.
        """

        # TODO: need a way to process this for missing data
        self.exog_re = patsy.dmatrix(re_formula, data)
        self.exog_re_names = self.exog_re.design_info.column_names
        self.exog_re = np.asarray(self.exog_re)
        self.exog_re_li = self.group_list(self.exog_re)
        self.k_re = self.exog_re.shape[1]
        self.k_re2 = self.k_re * (self.k_re + 1) // 2

    def group_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """

        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :])
                    for k in self.group_labels]


    def fit_regularized(self, method='l1', alpha=0, ceps=1e-4,
                        ptol=1e-6, maxit=200, **fit_args):
        """
        Minimize the L1-norm penalized log-likelihood with respect to
        the fixed effects parameters.  The dependence parameters are
        held fixed at their estimated values in the full model.

        Parameters:
        -----------
        method : string
            Method for regularization.  Currently must be 'l1',
            'l1_shooting', or 'l2'.
        alpha : array-like
            Scalar or vector of penalty weights.  If a scalar, the
            same weight is applied to all coefficients  If a
            vector, it contains a weight for each coefficient.
        ceps : positive real scalar
            Fixed effects parameters smaller than this value
            in magnitude are treaded as being zero.
        ptol : positive real scalar
            Convergence occurs when the sup norm difference
            between successive values of `fe_params` is less than
            `ptol`.
        maxit : integer
            The maximum number of iterations.

        Returns:
        --------
        A MixedLMResults instance containing the results.

        Notes:
        ------
        The covariance structure is not updated as the fixed effects
        parameters are varied.

        The algorithm used here is a "shooting" or cyclic coordinate
        descent algorithm.

        References:
        -----------
        Friedman, J. H., Hastie, T. and Tibshirani, R. Regularized
        Paths for Generalized Linear Models via Coordinate
        Descent. Journal of Statistical Software, 33(1) (2008)
        http://www.jstatsoft.org/v33/i01/paper

        http://statweb.stanford.edu/~tibs/stat315a/Supplements/fuse.pdf
        """

        method = method.lower()
        if method not in ('l1', 'l1_shooting', 'l2'):
            raise ValueError("Invalid regularization method")

        # For l2 we just call fit.
        if method == 'l2':
            return self.fit(fe_l2_pen=alpha, **fit_args)

        # No benefit in using the reml criterion here since the
        # dependence structure is not updated.
        cov_pen = fit_args["cov_pen"] if "cov_pen" in fit_args else 0.
        fe_l2_pen = fit_args["fe_l2_pen"] if "fe_l2_pen" in fit_args\
                    else None
        cov_pen = fit_args["cov_pen"] if "cov_pen" in fit_args\
                    else None
        like = lambda x: self.loglike_L(x, reml=False,
                                        fe_l2_pen=fe_l2_pen,
                                        cov_pen=cov_pen)

        # Fit the unconstrained model to get the dependence structure.
        fit_args["use_L"] = True
        mdf = self.fit(**fit_args)
        fe_params = mdf.fe_params
        cov_re = mdf.cov_re
        sig2 = mdf.sig2
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        if np.isscalar(alpha):
            alpha = alpha * np.ones(len(fe_params), dtype=np.float64)

        for itr in range(maxit):

            fe_params_s = fe_params.copy()
            for j in range(self.k_fe):

                if abs(fe_params[j]) < ceps:
                    continue

                # The residuals
                fe_params[j] = 0.
                expval = np.dot(self.exog, fe_params)
                resid_all = self.endog - expval

                # The loss function has the form
                # a*x^2 + b*x + pwt*|x|
                a, b = 0., 0.
                for k, lab in enumerate(self.group_labels):

                    exog = self.exog_li[k]
                    ex_r = self.exog_re_li[k]
                    resid = resid_all[self.row_indices[lab]]

                    x = exog[:,j]
                    u = _smw_solve(sig2, ex_r, cov_re, cov_re_inv, x)
                    a += np.dot(u, x)
                    b -= 2*np.dot(u, resid)

                pwt1 = alpha[j]
                if b > pwt1:
                    fe_params[j] = -(b - pwt1) / (2*a)
                elif b < -pwt1:
                    fe_params[j] = -(b + pwt1) / (2*a)

            if np.abs(fe_params_s - fe_params).max() < ptol:
                break

        # Replace the fixed effects estimates with their penalized
        # values, leave the dependence parameters in their unpenalized
        # state.
        params_prof = mdf.params.copy()
        params_prof[0:self.k_fe] = fe_params

        # Get the Hessian including only the nonzero fixed effects,
        # then blow back up to the full size after inverting.
        hess = self.hessian(params_prof)
        pcov = np.nan * np.ones_like(hess)
        ii = np.abs(params_prof) > ceps
        ii[self.k_fe:] = True
        ii = np.flatnonzero(ii)
        hess1 = hess[ii,:][:,ii]
        pcov[np.ix_(ii,ii)] = np.linalg.inv(-hess1)

        results = MixedLMResults(self, params_prof, pcov)
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.sig2 = sig2
        results.cov_re_unscaled = mdf.cov_re_unscaled
        results.method = mdf.method
        results.converged = True
        results.cov_pen = mdf.cov_pen
        results.likeval = like(params_prof)
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2

        return results


    def _unpack(self, params, sym=True):
        """
        Takes as input the packed parameter vector and returns a
        vector containing the regression slopes and a matrix defining
        the dependence structure.

        Arguments:
        ----------
        params : array-like
            The packed parameters
        sym : bool
          If true, the variance parameters are returned as a symmetric
          matrix; if False, the variance parameters are returned as a
          lower triangular matrix.

        Returns:
        --------
        params : 1d ndarray
            The fixed effects coefficients
        cov_re : 2d ndarray
            The random effects covariance matrix
        """

        fe_params = params[0:self.k_fe]
        re_params = params[self.k_fe:]

        # Unpack the covariance matrix of the random effects
        cov_re = np.zeros((self.k_re, self.k_re), dtype=np.float64)
        ix = np.tril_indices(self.k_re)
        cov_re[ix] = re_params

        if sym:
            cov_re = (cov_re + cov_re.T) - np.diag(np.diag(cov_re))

        return fe_params, cov_re

    def _pack(self, vec, mat):
        """
        Packs the model parameters into a single vector.

        Arguments
        ---------
        vec : 1d ndarray
            A vector
        mat : 2d ndarray
            An (assumed) symmetric matrix

        Returns
        -------
        params : 1d ndarray
            The vector and the lower triangle of the matrix,
            concatenated.
        """

        ix = np.tril_indices(mat.shape[0])
        return np.concatenate((vec, mat[ix]))

    def loglike(self, params, reml=True, fe_l2_pen=None,
                cov_pen=None):
        """
        Evaluate the profile log-likelihood of the linear mixed
        effects model.  Specifically, this is the profile likelihood
        in which the scale parameter sig2 has been profiled out.

        Arguments
        ---------
        params : 1d ndarray
            The parameter values, packed into a single vector.  See
            below for details.
        reml : bool
            If true, return REML log likelihood, else return ML
            log likelihood
        fe_l2_pen : float or array-like
            Penalty weight for L2 (ridge) penalization of fixed
            effects.  If a scalar, all coefficients are penalized by
            the same amount; if a vector, each element gives the
            penalty weight for one coefficient.
        cov_pen : non-negative float
            Weight for the penalty for non-positive semidefinite
            covariance matrices; if None there is no penalization.

        Returns
        -------
        likeval : scalar
            The log-likelihood value at `params`.

        Notes
        -----
        The first p elements of the packed vector are the regression
        slopes, and the remaining q*(q+1)/2 elements are the lower
        triangle of the random effects covariance matrix Psi, packed
        row-wise.  The matrix Psi is used to form the covariance
        matrix V = I + Z * Psi * Z', where Z is the design matrix for
        the random effects structure.  To convert this to the full
        likelihood (not profiled) parameterization, calculate the
        error variance sig2, and divide Psi by sig2.
        """

        fe_params, cov_re = self._unpack(params)
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None
        _, cov_re_logdet = np.linalg.slogdet(cov_re)

        # The residuals
        expval = np.dot(self.exog, fe_params)
        resid_all = self.endog - expval

        likeval = 0.

        # Handle the covariance penalty
        if cov_pen is not None:
            cy = np.linalg.cholesky(cov_re)
            likeval += cov_pen * 2 * np.sum(np.log(np.diag(cy)))

        # Handle the fixed effects penalty
        if fe_l2_pen is not None:
            if np.isscalar(fe_l2_pen):
                likeval += fe_l2_pen * np.sum(fe_params**2)
            else:
                likeval += np.sum(fe_l2_pen * fe_params**2)

        xvx, qf = 0., 0.
        for k, lab in enumerate(self.group_labels):

            exog = self.exog_li[k]
            ex_r = self.exog_re_li[k]
            resid = resid_all[self.row_indices[lab]]

            # Part 1 of the log likelihood (for both ML and REML)
            ld = _smw_logdet(1., ex_r, cov_re, cov_re_inv,
                             cov_re_logdet)
            likeval -= ld / 2.

            # Part 2 of the log likelihood (for both ML and REML)
            u = _smw_solve(1., ex_r, cov_re, cov_re_inv, resid)
            qf += np.dot(resid, u)

            # Adjustment for REML
            if reml:
                mat = _smw_solve(1., ex_r, cov_re, cov_re_inv, exog)
                xvx += np.dot(exog.T, mat)

        if reml:
            likeval -= (self.n_totobs - self.k_fe) * np.log(qf) / 2.
            _,ld = np.linalg.slogdet(xvx)
            likeval -= ld / 2.
            likeval -= (self.n_totobs - self.k_fe) * np.log(2 * np.pi) / 2.
            likeval += ((self.n_totobs - self.k_fe) *
                        np.log(self.n_totobs - self.k_fe) / 2.)
            likeval -= (self.n_totobs - self.k_fe) / 2.
        else:
            likeval -= self.n_totobs * np.log(qf) / 2.
            likeval -= self.n_totobs * np.log(2 * np.pi) / 2.
            likeval += self.n_totobs * np.log(self.n_totobs) / 2.
            likeval -= self.n_totobs / 2.

        return likeval


    def _gen_dV_dPsi(self, ex_r, max_ix=None):
        """
        A generator that yields the derivative of the covariance
        matrix V (=I + Z*Psi*Z') with respect to the free elements of
        Psi.  Each call to the generator yields the index of Psi with
        respect to which the derivative is taken, and the derivative
        matrix with respect to that element of Psi.  Psi is a
        symmetric matrix, so the free elements are the lower triangle.
        If max_ix is not None, the iterations terminate after max_ix
        values are yielded.
        """

        jj = 0
        for j1 in range(self.k_re):
            for j2 in range(j1 + 1):
                if max_ix is not None and jj > max_ix:
                    return
                mat = np.outer(ex_r[:,j1], ex_r[:,j2])
                if j1 != j2:
                    mat += mat.T
                yield jj,mat
                jj += 1


    def score(self, params, reml=True, fe_l2_pen=None,
              cov_pen=None):
        """
        Calculates the score vector for the mixed effects model.
        Specifically, this is the score for the profile likelihood in
        which the scale parameter sig2 has been profiled out.

        Parameters
        ----------
        params : 1d ndarray
            All model parameters in packed form
        reml : bool
            If true, return REML score, else return ML score
        fe_l2_pen : float or array-like
            Penalty weight for L2 (ridge) penalization of fixed
            effects.  If a scalar, all coefficients are penalized by
            the same amount; if a vector, each element gives the
            penalty weight for one coefficient.
        cov_pen : non-negative float
            Weight for the penalty for non positive semidefinite
            covariance matrices.  If None, there is no penalization.

        Returns
        -------
        scorevec : 1d ndarray
            The score vector, calculated at `params`.
        """

        fe_params, cov_re = self._unpack(params)
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        score_fe = np.zeros(self.k_fe, dtype=np.float64)
        score_re = np.zeros(self.k_re2, dtype=np.float64)

        # Handle the covariance penalty.
        if cov_pen is not None:
            cy = cov_re_inv.copy()
            cy = 2*cy - np.diag(np.diag(cy))
            i,j = np.tril_indices(self.k_re)
            score_re += cov_pen * cy[i,j]

        # Handle the fixed effects penalty.
        if fe_l2_pen is not None:
            score_fe += 2 * fe_l2_pen * fe_params

        # resid' V^{-1} resid, summed over the groups (a scalar)
        rvir = 0.

        # exog' V^{-1} resid, summed over the groups (a k_fe
        # dimensional vector)
        xtvir = 0.

        # exog' V^{_1} exog, summed over the groups (a k_fe x k_fe
        # matrix)
        xtvix = 0.

        # V^{-1} exog' dV/dQ_jj exog V^{-1}, where Q_jj is the jj^th
        # covariance parameter.
        xtax = [0.,] * self.k_re2

        # Temporary related to the gradient of log |V|
        dlv = np.zeros(self.k_re2, dtype=np.float64)

        # resid' V^{-1} dV/dQ_jj V^{-1} resid (a scalar)
        rvavr = np.zeros(self.k_re2, dtype=np.float64)

        for k in range(self.n_groups):

            exog = self.exog_li[k]
            ex_r = self.exog_re_li[k]

            # The residuals
            expval = np.dot(exog, fe_params)
            resid = self.endog_li[k] - expval

            if reml:
                viexog = _smw_solve(1., ex_r, cov_re, cov_re_inv, exog)
                xtvix += np.dot(exog.T, viexog)

            # Contributions to the covariance parameter gradient
            jj = 0
            vex = _smw_solve(1., ex_r, cov_re, cov_re_inv, ex_r)
            vir = _smw_solve(1., ex_r, cov_re, cov_re_inv, resid)
            for jj,mat in self._gen_dV_dPsi(ex_r):
                dlv[jj] = np.trace(_smw_solve(1., ex_r, cov_re,
                                     cov_re_inv, mat))
                rvavr[jj] += np.dot(vir, np.dot(mat, vir))
                if reml:
                    xtax[jj] += np.dot(viexog.T, np.dot(mat, viexog))

            # Contribution of log|V| to the covariance parameter
            # gradient.
            score_re -= 0.5 * dlv

            # Nededed for the fixed effects params gradient
            rvir += np.dot(resid, vir)
            xtvir += np.dot(exog.T, vir)

        fac = self.n_totobs
        if reml:
            fac -= self.exog.shape[1]

        score_fe = fac * xtvir / rvir
        score_re += 0.5 * fac * rvavr / rvir

        if reml:
            for j in range(self.k_re2):
                score_re[j] += 0.5 * np.trace(np.linalg.solve(
                    xtvix, xtax[j]))

        return np.concatenate((score_fe, score_re))


    def loglike_L(self, params, reml=True, fe_l2_pen=None,
                  cov_pen=None):
        """
        Returns the log likelihood evaluated at a given point.  The
        random effects covariance is passed as a square root.

        Arguments:
        ----------
        params : array-like
            The model parameters (for the profile likelihood) in
            packed form.  The first p elements are the regression
            slopes, and the remaining elements are the lower triangle
            of a lower triangular matrix L such that Psi = LL'
        reml : bool
            If true, returns the REML log likelihood, else returns
            the standard log likeihood.
        cov_pen : non-negative float
            See `loglike`.
        fe_l2_pen : float or array
            See `loglike`.

        Returns:
        --------
        The value of the log-likelihood or REML criterion.
        """

        fe_params, L = self._unpack(params, sym=False)
        cov_re = np.dot(L, L.T)

        params_r = self._pack(fe_params, cov_re)

        likeval = self.loglike(params_r, reml=reml,
                               cov_pen=cov_pen,
                               fe_l2_pen=fe_l2_pen)
        return likeval



    def score_L(self, params, reml=True, fe_l2_pen=None,
                cov_pen=None):
        """
        Returns the score vector valuated at a given point.  The
        random effects covariance matrix is passed as a square root.

        Arguments:
        ----------
        params : array-like
            The model parameters (for the profile likelihood) in
            packed form.  The first p elements are the regression
            slopes, and the remaining elements are the lower triangle
            of a lower triangular matrix L such that Psi = LL'
        reml : bool
            If true, returns the REML log likelihood, else returns
            the standard log likeihood.
        cov_pen : non-negative float
            See `score`.
        fe_l2_pen : float or array
            See `score`.

        Returns:
        --------
        The score vector for the log-likelihood or REML criterion.
        """

        fe_params, L = self._unpack(params, sym=False)
        cov_re = np.dot(L, L.T)

        params_f = self._pack(fe_params, cov_re)
        svec = self.score(params_f, reml=reml, cov_pen=cov_pen,
                          fe_l2_pen=fe_l2_pen)
        s_fe, s_re = self._unpack(svec, sym=False)

        # Use the chain rule to get d/dL from d/dPsi
        s_l = np.zeros(self.k_re2, dtype=np.float64)
        jj = 0
        for i in range(self.k_re):
            for j in range(i+1):
                s_l[jj] += np.dot(s_re[:,i], L[:,j])
                s_l[jj] += np.dot(s_re[i,:], L[:,j])
                jj += 1

        gr = np.concatenate((s_fe, s_l))

        return gr

    def hessian(self, params, reml=True):
        """
        Calculates the Hessian matrix for the mixed effects model.
        Specifically, this is the Hessian matrix for the profile
        likelihood in which the scale parameter sig2 has been profiled
        out.  The parameters are passed in packed form, with only the
        lower triangle of the covariance (not its square root) passed.

        Parameters
        ----------
        params : 1d ndarray
            All model parameters in packed form
        reml : bool
            If true, return REML Hessian, else return ML Hessian

        Returns
        -------
        hess : 2d ndarray
            The Hessian matrix, evaluated at `params`.
        """

        fe_params, cov_re = self._unpack(params)
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        # Blocks for the fixed and random effects parameters.
        hess_fe = 0.
        hess_re = np.zeros((self.k_re2, self.k_re2), dtype=np.float64)
        hess_fere = np.zeros((self.k_re2, self.k_fe),
                             dtype=np.float64)

        fac = self.n_totobs
        if reml:
            fac -= self.exog.shape[1]

        rvir = 0.
        xtvix = 0.
        xtax = [0.,] * self.k_re2
        B = np.zeros(self.k_re2, dtype=np.float64)
        D = np.zeros((self.k_re2, self.k_re2), dtype=np.float64)
        F = [[0.,]*self.k_re2 for k in range(self.k_re2)]
        for k in range(self.n_groups):

            exog = self.exog_li[k]
            ex_r = self.exog_re_li[k]

            # The residuals
            expval = np.dot(exog, fe_params)
            resid = self.endog_li[k] - expval

            viexog = _smw_solve(1., ex_r, cov_re, cov_re_inv, exog)
            xtvix += np.dot(exog.T, viexog)
            vir = _smw_solve(1., ex_r, cov_re, cov_re_inv, resid)
            rvir += np.dot(resid, vir)

            for jj1,mat1 in self._gen_dV_dPsi(ex_r):

                hess_fere[jj1,:] += np.dot(viexog.T,
                                           np.dot(mat1, vir))
                if reml:
                    xtax[jj1] += np.dot(viexog.T, np.dot(mat1, viexog))

                B[jj1] += np.dot(vir, np.dot(mat1, vir))
                E = _smw_solve(1., ex_r, cov_re, cov_re_inv, mat1)

                for jj2,mat2 in self._gen_dV_dPsi(ex_r, jj1):
                    Q = np.dot(mat2, E)
                    Q1 = Q + Q.T
                    vt = np.dot(vir, np.dot(Q1, vir))
                    D[jj1, jj2] += vt
                    if jj1 != jj2:
                        D[jj2, jj1] += vt
                    R = _smw_solve(1., ex_r, cov_re, cov_re_inv, Q)
                    rt = np.trace(R) / 2
                    hess_re[jj1, jj2] += rt
                    if jj1 != jj2:
                        hess_re[jj2, jj1] += rt
                    if reml:
                        F[jj1][jj2] += np.dot(viexog.T,
                                              np.dot(Q, viexog))

        hess_fe -= fac * xtvix / rvir

        hess_re -= 0.5 * fac * (D / rvir - np.outer(B, B) / rvir**2)

        hess_fere = -fac * hess_fere / rvir

        if reml:
            for j1 in range(self.k_re2):
                Q1 = np.linalg.solve(xtvix, xtax[j1])
                for j2 in range(j1 + 1):
                    Q2 = np.linalg.solve(xtvix, xtax[j2])
                    a = np.trace(np.dot(Q1, Q2))
                    a -= np.trace(np.linalg.solve(xtvix, F[j1][j2]))
                    a *= 0.5
                    hess_re[j1, j2] += a
                    if j1 > j2:
                        hess_re[j2, j1] += a

        # Put the blocks together to get the Hessian.
        m = self.k_fe + self.k_re2
        hess = np.zeros((m, m), dtype=np.float64)
        hess[0:self.k_fe, 0:self.k_fe] = hess_fe
        hess[0:self.k_fe, self.k_fe:] = hess_fere.T
        hess[self.k_fe:, 0:self.k_fe] = hess_fere
        hess[self.k_fe:, self.k_fe:] = hess_re

        return hess

    def Estep(self, fe_params, cov_re, sig2):
        """
        The E-step of the EM algorithm.  This is for ML (not REML),
        but it seems to be good enough to use for REML starting
        values.

        Parameters
        ----------
        fe_params : 1d ndarray
            The current value of the fixed effect coefficients
        cov_re : 2d ndarray
            The current value of the covariance matrix of random
            effects
        sig2 : positive scalar
            The current value of the error variance

        Returns
        -------
        m1x : 1d ndarray
            sum_groups X'*Z*E[gamma | Y], where X and Z are the fixed
            and random effects covariates, gamma is the random
            effects, and Y is the observed data
        m1y : scalar
            sum_groups Y'*E[gamma | Y]
        m2 : 2d ndarray
            sum_groups E[gamma * gamma' | Y]
        m2xx : 2d ndarray
            sum_groups Z'*Z * E[gamma * gamma' | Y]
        """

        m1x, m1y, m2, m2xx = 0., 0., 0., 0.
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        for k in range(self.n_groups):

            # Get the residuals
            expval = np.dot(self.exog_li[k], fe_params)
            resid = self.endog_li[k] - expval

            # Contruct the marginal covariance matrix for this group
            ex_r = self.exog_re_li[k]

            vr1 = _smw_solve(sig2, ex_r, cov_re, cov_re_inv, resid)
            vr1 = np.dot(ex_r.T, vr1)
            vr1 = np.dot(cov_re, vr1)

            vr2 = _smw_solve(sig2, ex_r, cov_re, cov_re_inv,
                            self.exog_re_li[k])
            vr2 = np.dot(vr2, cov_re)
            vr2 = np.dot(ex_r.T, vr2)
            vr2 = np.dot(cov_re, vr2)

            rg = np.dot(ex_r, vr1)
            m1x += np.dot(self.exog_li[k].T, rg)
            m1y += np.dot(self.endog_li[k].T, rg)
            egg = cov_re - vr2 + np.outer(vr1, vr1)
            m2 += egg
            m2xx += np.dot(np.dot(ex_r.T, ex_r), egg)

        return m1x, m1y, m2, m2xx


    def EM(self, fe_params, cov_re, sig2, num_em=10,
           hist=None):
        """
        Run the EM algorithm from a given starting point.  This is for
        ML (not REML), but it seems to be good enough to use for REML
        starting values.

        Returns
        -------
        fe_params : 1d ndarray
            The final value of the fixed effects coefficients
        cov_re : 2d ndarray
            The final value of the random effects covariance
            matrix
        sig2 : float
            The final value of the error variance

        Notes
        -----
        This uses the parameterization of the likelihood sig2*I +
        Z'*V*Z, note that this differs from the profile likelihood
        used in the gradient calculations.
        """

        xxtot = 0.
        for x in self.exog_li:
            xxtot += np.dot(x.T, x)

        xytot = 0.
        for x,y in zip(self.exog_li, self.endog_li):
            xytot += np.dot(x.T, y)

        pp = []
        for itr in range(num_em):

            m1x, m1y, m2, m2xx = self.Estep(fe_params, cov_re, sig2)

            fe_params = np.linalg.solve(xxtot, xytot - m1x)
            cov_re = m2 / self.n_groups

            sig2 = 0.
            for x,y in zip(self.exog_li, self.endog_li):
                sig2 += np.sum((y - np.dot(x, fe_params))**2)
            sig2 -= 2*m1y
            sig2 += 2*np.dot(fe_params, m1x)
            sig2 += np.trace(m2xx)
            sig2 /= self.n_totobs

            if hist is not None:
                hist.append(["EM", fe_params, cov_re, sig2])

        return fe_params, cov_re, sig2


    def get_sig2(self, fe_params, cov_re, reml):
        """
        Returns the estimated error variance based on given estimates
        of the slopes and random effects covariance matrix.

        Arguments:
        ----------
        fe_params : array-like
            The regression slope estimates
        cov_re : 2d array
            Estimate of the random effects covariance matrix (Psi).
        reml : bool
            If true, use the REML esimate, otherwise use the MLE.

        Returns:
        --------
        sig2 : float
            The estimated error variance.
        """

        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        qf = 0.
        for k in range(self.n_groups):

            exog = self.exog_li[k]
            ex_r = self.exog_re_li[k]

            # The residuals
            expval = np.dot(exog, fe_params)
            resid = self.endog_li[k] - expval

            mat = _smw_solve(1., ex_r, cov_re, cov_re_inv, resid)
            qf += np.dot(resid, mat)

        if reml:
            qf /= (self.n_totobs - self.k_fe)
        else:
            qf /= self.n_totobs

        return qf


    def _steepest_descent(self, func, params, score, gtol=1e-4,
                          max_iter=50):
        """
        Uses the steepest descent algorithm to minimize a function.

        Arguments:
        ----------
        func : function
            The real-valued function to minimize.
        params : array-like
            A point in the domain of `func`, used as the starting
            point for the iterative minimization.
        score : function
            A function implementing the score vector (gradient) of
            `func`.
        gtol : non-negative float
            Return if the sup norm of the score vector is less than
            this value.
        max_iter: non-negative integer
            Return once this number of iterations have occured.

        Returns:
        --------
        params_out : array-like
            The final value of the iterations
        success : bool
            True if the final score vector has sup-norm no larger
            than `gtol`.
        """

        fval = func(params)

        for itr in range(max_iter):

            gro = score(params)
            gr = gro / np.max(np.abs(gro))

            sl = 1.
            success = False
            while sl > 1e-20:
                params1 = params - sl * gr
                fval1 = func(params1)

                if fval1 < fval:
                    params = params1
                    fval = fval1
                    success = True
                    break

                sl /= 2

            if not success:
                break

        return params, np.max(np.abs(gro)) < gtol

    def fit(self, start=None, reml=True, num_sd=1,
            num_em=0, do_cg=True, fe_l2_pen=None, cov_pen=None,
            gtol=1e-4, use_L=True, free=None, full_output=False):
        """
        Fit a linear mixed model to the data.

        Parameters
        ----------
        start: dict
            If provided, this is a dict containing starting values.
            `start["fe"]` contains starting values for the fixed
            effects regression slopes.  `start["cov_re"]` contains
            the covariance matrix of random effects as found
            in the `cov_re` component of MixedLMResults.  If
            `start["cov_re"]` is provided, then `start["sig2"]` must
            also be provided (this is the error variance).
            Alternatively, the random effects may be specified as
            `start["cov_re_L_unscaled"]`, which is the packed lower
            triangle of the covariance matrix in the
            profile parameterization (in this case sig2 is not used).
        reml : bool
            If true, fit according to the REML likelihood, else
            fit the standard likelihood using ML.
        num_sd : integer
            The number of steepest descent iterations
        num_em : non-negative integer
            The number of EM steps.  The EM steps always
            preceed steepest descent and conjugate gradient
            optimization.  The EM algorithm implemented here
            is for ML estimation.
        do_cg : bool
            If True, a conjugate gradient algorithm is
            used for optimization (following any steepest
            descent or EM steps).
        cov_pen : non-negative float
            Coefficient of a logarithmic barrier function
            to prevent the random effect covariance matrix from
            becoming singular.  If None, there is no penalization.
        fe_l2_pen : float or array
            Penalty weights for L2 (ridge) penalization of the
            fixed effects coefficients.  If a scalar, the same
            penalty is applied to all coefficients; if a vector,
            it contains a weight for each coefficient; if None,
            there is no penalization.
        gtol : non-negtive float
            Algorithm is considered converged if the sup-norm of
            the gradient is smaller than this value.
        use_L : bool
            If True, optimization is carried out using the lower
            triangle of the square root of the random effects
            covariance matrix, otherwise it is carried out using the
            lower triangle of the random effects covariance matrix.
        free : tuple of ndarrays
            If not `None`, this is a tuple of length 2 containing 2
            0/1 indicator arrays.  The first element of `free`
            corresponds to the regression slopes and the second
            element of `free` corresponds to the random effects
            covariance matrix (if `use_L` is False) or it square root
            (if `use_L` is True).  A 1 in either array indicates that
            the corresponding parameter is estimated, a 0 indicates
            that it is fixed at its starting value.  One use case if
            to set free[1] to the identity matrix to estimate a model
            with independent random effects.
        full_output : bool
            If true, attach iteration history to results

        Returns
        -------
        A MixedLMResults instance.
        """

        if use_L:
            like = lambda x: -self.loglike_L(x, reml=reml,
                                             cov_pen=cov_pen,
                                             fe_l2_pen=fe_l2_pen)
        else:
            like = lambda x: -self.loglike(x, reml=reml,
                                           cov_pen=cov_pen,
                                           fe_l2_pen=fe_l2_pen)

        if free is not None:
            pat_slopes = free[0]
            ix = np.tril_indices(self.k_re)
            pat_cov_re = free[1][ix]
            pat = np.concatenate((pat_slopes, pat_cov_re))
            if use_L:
                score = lambda x: -pat*self.score_L(x, reml=reml,
                                     cov_pen=cov_pen,
                                     fe_l2_pen=fe_l2_pen)
            else:
                score = lambda x: -pat*self.score(x, reml=reml,
                                       cov_pen=cov_pen,
                                       fe_l2_pen=fe_l2_pen)
        else:
            if use_L:
                score = lambda x: -self.score_L(x, reml=reml,
                                           cov_pen=cov_pen,
                                           fe_l2_pen=fe_l2_pen)
            else:
                score = lambda x: -self.score(x, reml=reml,
                                         cov_pen=cov_pen,
                                         fe_l2_pen=fe_l2_pen)

        if full_output:
            hist = []
        else:
            hist = None

        # Starting values
        ix = np.tril_indices(self.k_re)
        if start is None:
            start = {}
        if "fe" in start:
            fe_params = start["fe"]
        else:
            fe_params = np.zeros(self.exog.shape[1], dtype=np.float64)
        if "cov_re_L_unscaled" in start:
            if use_L:
                re_params = start["cov_re_L_unscaled"]
            else:
                vec = start["cov_re_L_unscaled"]
                mat = np.zeros((self.k_re, self.k_re), dtype=np.float64)
                mat[ix] = vec
                mat = np.dot(mat, mat.T)
                re_params = mat[ix]
        elif "cov_re" in start:
            cov_re_unscaled = start["cov_re"] / start["sig2"]
            if use_L:
                cov_re_L_unscaled = np.linalg.cholesky(cov_re_unscaled)
                re_params = cov_re_L_unscaled[ix]
            else:
                re_params = cov_re_unscaled[ix]
        else:
            re_params = np.eye(self.k_re)[ix]
        params_prof = np.concatenate((fe_params, re_params))

        success = False

        # EM iterations
        if num_em > 0:
            sig2 = 1.
            fe_params, cov_re, sig2 = self.EM(fe_params, cov_re, sig2,
                                             num_em, hist)

            # Gradient algorithms use a different parameterization
            # that profiles out sigma^2.
            if use_L:
                params_prof = self._pack(fe_params, cov_re / sig2)
            else:
                cov_re_rt = np.linalg.cholesky(cov_re / sig2)
                params_prof = self._pack(fe_params, cov_re_rt)
            if np.max(np.abs(score(params_prof))) < gtol:
                success = True

        for cycle in range(10):

            # Steepest descent iterations
            params_prof, success = self._steepest_descent(like,
                                  params_prof, score,
                                  gtol=gtol, max_iter=num_sd)
            if success:
                break

            # Gradient iterations
            try:
                rslt = fmin_bfgs(like, params_prof, score,
                                 full_output=True,
                                 disp=False,
                                 retall=hist is not None)
            # scipy.optimize routines have trouble staying in the
            # feasible region
            except np.linalg.LinAlgError:
                rslt = None
            if rslt is not None:
                if hist is not None:
                    hist.append(["Gradient", rslt[7]])
                params_prof = rslt[0]
                # TODO: do we need to recompute the score?
                if np.max(np.abs(score(params_prof))) < gtol:
                    success = True
                    break

        # Convert to the final parameterization (i.e. undo the square
        # root transform of the covariance matrix, and the profiling
        # over the error variance).
        fe_params, cov_re_ltri = self._unpack(params_prof, sym=False)
        if use_L:
            cov_re_unscaled = np.dot(cov_re_ltri, cov_re_ltri.T)
        else:
            cov_re_unscaled = cov_re_ltri
        sig2 = self.get_sig2(fe_params, cov_re_unscaled, reml)
        cov_re = sig2 * cov_re_unscaled

        if not success:
            if rslt is None:
                msg = "Gradient optimization failed, try increasing num_sd or cov_pen."
            else:
                msg = "Gradient sup norm=%.3f, try increasing num_sd or cov_pen." %\
                      np.max(np.abs(rslt[2]))
            warnings.warn(msg, ConvergenceWarning)

        if np.min(np.abs(np.diag(cov_re))) < 0.01:
            msg = "The MLE may be on the boundary of the parameter space."
            warnings.warn(msg, ConvergenceWarning)

        # Compute the Hessian at the MLE.  Note that the hessian
        # function expects the random effects covariance matrix (not
        # its square root).
        params_hess = self._pack(fe_params, cov_re_unscaled)
        hess = self.hessian(params_hess)
        if free is not None:
            ii = np.flatnonzero(pat)
            hess1 = hess[ii,:][:,ii]
            pcov = np.zeros_like(hess)
            pcov[np.ix_(ii,ii)] = np.linalg.inv(-hess1)
        else:
            pcov = np.linalg.inv(-hess)

        # Prepare a results class instance
        results = MixedLMResults(self, params_prof, pcov)
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.sig2 = sig2
        results.cov_re_unscaled = cov_re_unscaled
        results.method = "REML" if reml else "ML"
        results.converged = success
        results.hist = hist
        results.reml = reml
        results.cov_pen = cov_pen
        results.likeval = -like(params_prof)
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2

        return results




class MixedLMResults(base.LikelihoodModelResults):
    '''
    Class to contain results of fitting a linear mixed effects model.

    MixedLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Returns
    -------
    **Attributes**

    model : class instance
        Pointer to PHreg model instance that called fit.
    normalized_cov_params : array
        The sampling covariance matrix of the estimates
    fe_params : array
        The fitted fixed-effects coefficients
    re_params : array
        The fitted random-effects covariance matrix
    bse_fe : array
        The standard errors of the fitted fixed effects coefficients
    bse_re : array
        The standard errors of the fitted random effects covariance
        matrix

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''


    def __init__(self, model, params, cov_params):

        super(MixedLMResults, self).__init__(model, params,
           normalized_cov_params=cov_params)

    def bse_fe(self):
        """
        Returns the standard errors of the fixed effect regression
        coefficients.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(np.diag(self.cov_params())[0:p])

    def bse_re(self):
        """
        Returns the standard errors of the variance parameters.  Note
        that the sampling distribution of variance parameters is
        strongly skewed unless the sample size is large, so these
        standard errors may not give meaningful confidence intervals
        of p-values if used in the usual way.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(self.sig2 * np.diag(self.cov_params())[p:])


    def ranef(self):
        """
        Returns posterior means of all random effects.

        Returns:
        --------
        ranef_dict : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional means of the random effects
            given the data.
        """

        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        ranef_dict = {}
        for k in range(self.model.n_groups):

            endog = self.model.endog_li[k]
            exog = self.model.exog_li[k]
            ex_r = self.model.exog_re_li[k]
            label = self.model.group_labels[k]

            # Get the residuals
            expval = np.dot(exog, self.fe_params)
            resid = endog - expval

            vresid = _smw_solve(self.sig2, ex_r, self.cov_re,
                                cov_re_inv, resid)

            ranef_dict[label] = np.dot(self.cov_re,
                                       np.dot(ex_r.T, vresid))

        return ranef_dict


    def ranef_cov(self):
        """
        Returns the conditional covariance matrix of the random
        effects for each group.

        Returns:
        --------
        ranef_dict : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional covariance matrix of the
            random effects given the data.
        """

        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        ranef_dict = {}
        for k in range(self.model.n_groups):

            endog = self.model.endog_li[k]
            exog = self.model.exog_li[k]
            ex_r = self.model.exog_re_li[k]
            label = self.model.group_labels[k]

            mat1 = np.dot(ex_r, self.cov_re)
            mat2 = _smw_solve(self.sig2, ex_r, self.cov_re, cov_re_inv,
                             mat1)
            mat2 = np.dot(mat1.T, mat2)

            ranef_dict[label] = self.cov_re - mat2

        return ranef_dict


    def summary(self, yname=None, xname_fe=None, xname_re=None,
                title=None, alpha=.05):
        """Summarize the Regression Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname_fe : list of strings, optional
            Fixed effects covariate names
        xname_re : list of strings, optional
            Random effects covariate names
        title : string, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        from statsmodels.iolib import summary2
        smry = summary2.Summary()

        info = OrderedDict()
        info["Model:"] = "MixedLM"
        if yname is None:
            yname = self.model.endog_names
        info["Dependent Variable:"] = yname
        info["No. Groups:"] = str(self.model.n_groups)
        info["No. Observations:"] = str(self.model.n_totobs)
        info["Method:"] = self.method
        info["Res. Var.:"] = self.sig2
        info["Likelihood:"] = self.likeval
        info["Converged:"] = "Yes" if self.converged else "No"
        smry.add_dict(info)

        if xname_fe is not None:
            xname_fe = xname_fe
        elif self.model.exog_names is not None:
            xname_fe = self.model.exog_names
        else:
            xname_fe = []

        if xname_re is not None:
            xname_re = xname_re
        elif self.model.exog_re_names is not None:
            xname_re = self.model.exog_re_names
        else:
            xname_re = []

        while len(xname_fe) < self.k_fe:
            xname_fe.append("FE%d" % (len(xname_fe) + 1))

        while len(xname_re) < self.k_re:
            xname_re.append("RE%d" % (len(xname_re) + 1))

        float_fmt = "%.3f"

        names = xname_fe
        sdf = np.nan * np.ones((self.k_fe + self.k_re2, 6),
                               dtype=np.float64)
        sdf[0:self.k_fe, 0] = self.fe_params
        sdf[0:self.k_fe, 1] =\
                      np.sqrt(np.diag(self.cov_params()[0:self.k_fe]))
        sdf[0:self.k_fe, 2] = sdf[0:self.k_fe, 0] / sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 3] = 2 * norm.cdf(-np.abs(sdf[0:self.k_fe, 2]))
        qm = -norm.ppf(alpha / 2)
        sdf[0:self.k_fe, 4] = sdf[0:self.k_fe, 0] - qm * sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 5] = sdf[0:self.k_fe, 0] + qm * sdf[0:self.k_fe, 1]
        jj = self.k_fe
        for i in range(self.k_re):
            for j in range(i + 1):
                if i == j:
                    names.append(xname_re[i])
                else:
                    names.append(xname_re[i] + " x " + xname_re[j])
                sdf[jj, 0] = self.cov_re[i, j]
                sdf[jj, 1] = np.sqrt(self.sig2) * self.bse[jj]
                jj += 1

        sdf = pd.DataFrame(index=names, data=sdf)
        sdf.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                          '[' + str(alpha/2), str(1-alpha/2) + ']']
        for col in sdf.columns:
            sdf[col] = [float_fmt % x if np.isfinite(x) else ""
                        for x in sdf[col]]

        smry.add_df(sdf, align='l')

        return smry


    def profile_re(self, re_ix, num_low=5, dist_low=1., num_high=5,
                   dist_high=1.):
        """
        Calculate a series of values along a 1-dimensional profile
        likelihood.

        Arguments:
        ----------
        re_ix : integer
            The index of the variance parameter for which to construct
            a profile likelihood.
        num_low : integer
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : integer
            The number of points at which to calculate the likelihood
            abov the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.

        Result
        ------
        A matrix with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.
        """

        model = self.model
        p = model.exog.shape[1]
        pr = model.exog_re.shape[1]

        # Need to permute the variables so that the profiled variable
        # is first.
        exog_re_li_save = [x.copy() for x in model.exog_re_li]
        ix = range(pr)
        ix[0] = re_ix
        ix[re_ix] = 0
        for k in range(len(model.exog_re_li)):
           model.exog_re_li[k] = model.exog_re_li[k][:,ix]

        # Permute the covariance structure to match the permuted data.
        ru = self.params[p:]
        ik = np.tril_indices(pr)
        mat = np.zeros((pr ,pr), dtype=np.float64)
        mat[ik] = ru
        mat = np.dot(mat, mat.T)
        mat = mat[ix,:][:,ix]
        ix = np.tril_indices(pr)
        re_params = np.linalg.cholesky(mat)[ix]

        # Define the values to which the parameter of interest will be
        # constrained.
        ru0 = re_params[0]
        left = np.linspace(ru0 - dist_low, ru0, num_low + 1)
        right = np.linspace(ru0, ru0 + dist_high, num_high+1)[1:]
        rvalues = np.concatenate((left, right))

        # Indicators of which parameters are free and fixed.
        free_slopes = np.ones(p, dtype=np.float64)
        free_cov_re = np.ones((pr, pr), dtype=np.float64)
        free_cov_re[0] = 0

        start = {"fe": self.fe_params}

        likev = []
        for x in rvalues:
            re_params[0] = x
            start["cov_re_L_unscaled"] = re_params
            md1 = model.fit(start=start,
                            free=(free_slopes, free_cov_re),
                            reml=self.reml, cov_pen=self.cov_pen)
            likev.append([md1.cov_re[0,0], md1.likeval])
        likev = np.asarray(likev)

        model.exog_re = exog_re_li_save

        return likev
