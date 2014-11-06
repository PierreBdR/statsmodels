.. currentmodule:: statsmodels.nonparametric

.. _nonparametric:

Nonparametric Methods :mod:`nonparametric`
==========================================

This section collects various methods in nonparametric statistics. This
includes kernel density estimation for univariate and multivariate data,
kernel regression and locally weighted scatterplot smoothing (lowess).

sandbox.nonparametric contains additional functions that are work in progress
or don't have unit tests yet. We are planning to include here nonparametric
density estimators, especially based on kernel or orthogonal polynomials,
smoothers, and tools for nonparametric models and methods in other parts of
statsmodels.


Kernel density estimation
-------------------------

The kernel density estimation (KDE) is split among a variety of classes
implementing various boundery conditions or estimation methods. All these
classes are accessed through the :py:class:`kde.KDE` class. Through this class
you can perform univariate, multi-dimensional or multi-variate density
estimation.

Depending on the type of boundary conditions and variables (e.g. continuous or
discrete), the estimation can make use of FFT or DCT transformations for fast
estimation of the density on a regular grid. The various methods let the user
choose the kernel, as well as the bandwidth estimation method.

The main interface estimation is described in the :py:mod:`kde` module, the
bandwidth estimation methods in the :py:mod:`bandwidths` module, and the kernels
in the :py:mod:`kernels` modules.

.. Multivariate estimation (as provided by `KDEMultivariate`) uses product
   kernels. It supports least squares and maximum likelihood cross-validation
   for bandwidth estimation, as well as estimating mixed continuous, ordered and
   unordered data. The default kernels (Gaussian, Wang-Ryzin and
   Aitchison-Aitken) cannot be altered at the moment however.  Direct estimation
   of the conditional density (:math:`P(X | Y) = P(X, Y) / P(Y)`) is supported
   by `KDEMultivariateConditional`.

.. `KDEMultivariate` can do univariate estimation as well, but is up to two
   orders of magnitude slower than `KDEUnivariate`.


Kernel regression
-----------------

Kernel regression (as provided by `KernelReg`) is based on the same product
kernel approach as `KDEMultivariate`, and therefore has the same set of
features (mixed data, cross-validated bandwidth estimation, kernels) as
described above for `KDEMultivariate`.  Censored regression is provided by
`KernelCensoredReg`.

Note that code for semi-parametric partial linear models and single index
models, based on `KernelReg`, can be found in the sandbox.


References
----------

* B.W. Silverman, "Density Estimation for Statistics and Data Analysis"
* J.S. Racine, "Nonparametric Econometrics: A Primer," Foundation and
  Trends in Econometrics, Vol. 3, No. 1, pp. 1-88, 2008.
* Q. Li and J.S. Racine, "Nonparametric econometrics: theory and practice",
  Princeton University Press, 2006.
* Hastie, Tibshirani and Friedman, "The Elements of Statistical Learning:
  Data Mining, Inference, and Prediction", Springer, 2009.
* Racine, J., Li, Q. "Nonparametric Estimation of Distributions
  with Categorical and Continuous Data." Working Paper. (2000)
* Racine, J. Li, Q. "Kernel Estimation of Multivariate Conditional
  Distributions Annals of Economics and Finance 5, 211-235 (2004)
* Liu, R., Yang, L. "Kernel estimation of multivariate
  cumulative distribution function." Journal of Nonparametric Statistics 
  (2008)
* Li, R., Ju, G. "Nonparametric Estimation of Multivariate CDF
  with Categorical and Continuous Data." Working Paper
* Li, Q., Racine, J. "Cross-validated local linear nonparametric
  regression" Statistica Sinica 14(2004), pp. 485-512
* Racine, J.: "Consistent Significance Testing for Nonparametric
  Regression" Journal of Business & Economics Statistics
* Racine, J., Hart, J., Li, Q., "Testing the Significance of
  Categorical Predictor Variables in Nonparametric Regression
  Models", 2006, Econometric Reviews 25, 523-544


Sub-Modules
----------------

The public functions and classes are

.. toctree::
   :maxdepth: 2

   mod_kde
   mod_kde1d_methods
   mod_kdend_methods
   mod_kdenc_methods
   mod_kde_multivariate

..   kde.KDE
..   kde_utils.AxesType
..   grid.Grid
..   kde1d_methods

helper functions to manage the KDE objects

.. autosummary::
   :toctree: generated/

   grid.Grid
   kde_utils.AxesTypes
   grid_interpolation.GridInterpolator
   fast_linbin.fast_linbin
   fast_linbin.fast_linbin_nd

bandwidth methods

.. autosummary::
   :toctree: generated/

   bandwidths.silverman_bandwidth
   bandwidths.scotts_bandwidth
   bandwidths.silverman_bandwidth_full
   bandwidths.scotts_bandwidth_full
   bandwidths.botev_bandwidth
   bandwidths.leastsquare_cv_bandwidth
   bandwidths.MultivariateBandwidth


There are some examples for nonlinear functions in
:mod:`statsmodels.nonparametric.dgp_examples`

The sandbox.nonparametric contains additional insufficiently tested classes
for testing functional form and for semi-linear and single index models.
