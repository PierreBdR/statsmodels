import statsmodels.api as sm
import scipy as sp
from scipy import linalg
import statsmodels.discrete.l1 as l1
import pdb
# pdb.set_trace()



def main():
    """
    Demonstrates l1 regularization for MNLogit model.

    How to use this demo
    --------------------
    Adjust the "Commonly adjusted params" at the top, then run with
        $ python demo.py

    The Story
    ---------
    The maximum likelihood (ML) solution works well when the number of data
    points is large and the noise is small.  When the ML solution starts
    "breaking", the regularized solution should do better.

    The l1 Solvers
    --------------
    The solvers are slower than standard Newton, and sometimes have
        convergence issues Nonetheless, the final solution makes sense and
        is often better than the ML solution.
    The standard l1 solver is fmin_slsqp and is included with scipy.  It
        sometimes has trouble verifying convergence when the data size is
        large.
    The l1_cvxopt_cp solver is part of CVXOPT and this package needs to be
        installed separately.  It works well even for larger data sizes.
    """
    ##########################################################################
    #### Commonly adjusted params
    ##########################################################################
    N = 500 # Number of data points
    num_targets = 3  # Targets are the dependent variables
    num_nonconst_covariates = 10 # For every target
    num_zero_params = 8 # For every target
    print_summaries = False
    get_l1_slsqp_results = True
    get_l1_cvxopt_results = True
    save_results = False
    load_old_results = False
    # The regularization parameter
    # Here we scale it with N for simplicity.  In practice, you should
    # use cross validation to pick alpha
    alpha = 0.01 * N * sp.ones((num_nonconst_covariates+1, num_targets-1))
    alpha[0,:] = 0  # Don't regularize the intercept
    # Correlation length for the independent variables
    # Higher makes the problem more ill-posed, and easier to screw
    # up with noise.
    # BEWARE:  With long correlation lengths, you often get a singular KKT
    # matrix (during the l1_cvxopt_cp fit)
    cor_length = 2
    noise_level = 0.2  # As a fraction of the "signal"
    ##########################################################################

    #### Make the arrays
    exog = get_exog(N, num_nonconst_covariates, cor_length)
    exog = sm.add_constant(exog, prepend=True)
    true_params = sp.rand(num_nonconst_covariates+1, num_targets-1)
    if num_zero_params:
        true_params[-num_zero_params:, :] = 0
    endog = get_multinomial_endog(num_targets, true_params, exog, noise_level)
    if save_results:
        sp.save('endog.npy', endog)
        sp.save('exog.npy', exog)
        sp.save('true_params.npy', true_params)
    if load_old_results:
        endog = sp.load('endog.npy')
        exog = sp.load('exog.npy')
        true_params = sp.load('true_params.npy')

    #### Train the models
    print_str = '\n\n=========== Error Summary ============'
    model = sm.MNLogit(endog, exog)
    # Get ML results
    results_ML = model.fit(method='newton')
    RMSE_ML = get_RMSE(results_ML, true_params)
    print_str += '\n\n The maximum likelihood fit RMS error = %.4f'%RMSE_ML
    # Get l1 results
    start_params = results_ML.params.ravel(order='F')
    if get_l1_slsqp_results:
        results_l1_slsqp = model.fit(method='l1', alpha=alpha,
                maxiter=70, start_params=start_params, trim_params=True,
                retall=True)
        RMSE_l1_slsqp = get_RMSE(results_l1_slsqp, true_params)
        print_str += '\n The l1_slsqp fit RMS error = %.4f'%RMSE_l1_slsqp
    if get_l1_cvxopt_results:
        results_l1_cvxopt_cp = model.fit(method='l1_cvxopt_cp', alpha=alpha,
                maxiter=50, start_params=start_params, trim_params=True,
                retall=True, feastol=1e-5)
        RMSE_l1_cvxopt_cp = get_RMSE(results_l1_cvxopt_cp, true_params)
        print_str += '\n The l1_cvxopt_cp fit RMS error = %.4f'%RMSE_l1_cvxopt_cp

    #### Print results
    print_str += '\n\n\n============== Parameters ================='
    print_str += "\n\nTrue parameters: \n%s"%true_params
    if print_summaries:
        print_str += '\n' + results_ML.summary().as_text()
        if get_l1_slsqp_results:
            print_str += '\n' + results_l1_slsqp.summary().as_text()
        if get_l1_cvxopt_results:
            print_str += '\n' + results_l1_cvxopt_cp.summary().as_text()
    else:
        print_str += '\n\nThe maximum likelihood params are \n%s'%results_ML.params
        if get_l1_slsqp_results:
            print_str += '\n\nThe l1_slsqp params are \n%s'%results_l1_slsqp.params
        if get_l1_cvxopt_results:
            print_str += '\n\nThe l1_cvxopt_cp params are \n%s'%results_l1_cvxopt_cp.params
    print print_str


def get_RMSE(results, true_params):
    """
    Gets the (normalized) root mean square error.
    """
    raw_RMSE = sp.sqrt(((results.params - true_params)**2).sum())
    param_norm = sp.sqrt((true_params**2).sum())
    return raw_RMSE / param_norm

def get_multinomial_endog(true_params, exog, noise_level):
    """
    Gets an endogenous response that is consistent with the true_params,
        perturbed by noise at noise_level.
    """
    N = exog.shape[0]
    ### Create the probability of entering the different classes,
    ### given exog and true_params
    Xdotparams = sp.dot(exog, true_params)
    noise = noise_level * sp.randn(*Xdotparams.shape)
    eXB = sp.column_stack((sp.ones(len(Xdotparams)), sp.exp(Xdotparams)))
    class_probabilities = eXB / eXB.sum(1)[:, None]

    ### Create the endog
    cdf = class_probabilities.cumsum(axis=1)
    endog = sp.zeros(N)
    for n in xrange(N):
        endog[n] = sp.searchsorted(cdf[n, :], sp.rand())

    return endog

def get_exog(N, num_nonconst_covariates, cor_length):
    """
    Returns an exog array with correlations determined by cor_length.
    The covariance matrix of exog will have (asymptotically, as
    :math:'N\\to\\inf')
    .. math:: Cov[i,j] = \\exp(-|i-j| / cor_length)
    """
    ## Create the noiseless exog
    uncorrelated_exog = sp.randn(N, num_nonconst_covariates)
    if cor_length == 0:
        exog = uncorrelated_exog
    else:
        cov_matrix = sp.zeros((num_nonconst_covariates, num_nonconst_covariates))
        j = sp.arange(num_nonconst_covariates)
        for i in xrange(num_nonconst_covariates):
            cov_matrix[i,:] = sp.exp(-sp.fabs(i-j) / cor_length)
        chol = linalg.cholesky(cov_matrix)  # cov_matrix = sp.dot(chol.T, chol)
        exog = sp.dot(uncorrelated_exog, chol)
    ## Return
    return exog



if __name__ == '__main__':
    main()
