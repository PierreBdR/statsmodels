.. currentmodule:: statsmodels.nonparametric.kde

.. _nonparametric_kde:

Kernel Density Estimation :mod:`nonparametric.kde`
==================================================

Module implementing kernel-based estimation of density of probability.

Given a kernel :math:`K`, the density function is estimated from a sampling
:math:`X = \{X_i \in \mathbb{R}^n\}_{i\in\{1,\ldots,m\}}` as:

.. math::

    f(\mathbf{z}) \triangleq \frac{1}{hW} \sum_{i=1}^m \frac{w_i}{\lambda_i}
    K\left(\frac{X_i-\mathbf{z}}{h\lambda_i}\right)

    W = \sum_{i=1}^m w_i

where :math:`h` is the bandwidth of the kernel, :math:`w_i` are the weights of
the data points and :math:`\lambda_i` are the adaptation factor of the kernel
width.

The kernel is a function of :math:`\mathbb{R}^n` such that:

.. math::

    \begin{array}{rclcl}
       \idotsint_{\mathbb{R}^n} f(\mathbf{z}) d\mathbf{z}
       & = & 1 & \Longleftrightarrow & \text{$f$ is a probability}\\
       \idotsint_{\mathbb{R}^n} \mathbf{z}f(\mathbf{z}) d\mathbf{z} &=&
       \mathbf{0} & \Longleftrightarrow & \text{$f$ is
       centered}\\
       \forall \mathbf{u}\in\mathbb{R}^n, \|\mathbf{u}\|
       = 1\qquad\int_{\mathbb{R}} t^2f(t \mathbf{u}) dt &\approx&
       1 & \Longleftrightarrow & \text{The co-variance matrix of $f$ is close
       to be the identity.}
    \end{array}

The constraint on the covariance is only required to provide a uniform meaning
for the bandwidth of the kernel.

If the domain of the density estimation is bounded to the interval
:math:`[L,U]`, the density is then estimated with:

.. math::

    f(x) \triangleq \frac{1}{hW} \sum_{i=1}^n \frac{w_i}{\lambda_i}
    \hat{K}(x;X,\lambda_i h,L,U)

where :math:`\hat{K}` is a modified kernel that depends on the exact method
used. Currently, only 1D KDE supports bounded domains.

Basic Usage
-----------

The univariate case
```````````````````

Let's start with a simple example. We will generate some random data, create the
model and start an estimation::

    >>> import statsmodels.api as sm
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> data = stats.norm(0,1).rvs(3000)
    >>> model = sm.nonparametric.KDE(data)
    >>> est = model.fit()
    >>> est([-0.5, 0, 0.5])
    array([ 0.3482476 ,  0.3914852 ,  0.35589638])

A bit more interesting, we can generate
a :py:class:`statsmodels.nonparametric.grid.Grid` object and the
values associated to it::

    >>> grid, freqs = est.grid()
    >>> plt.hist(data, bins=data.ptp()/est.bandwidth, normed=True)
    >>> plt.plot(grid.full(), freqs, label='KDE estimate')
    >>> plt.legend(loc='best')

.. plot:: plots/nonparametric_kde_example.py

The multi-variate case
``````````````````````

What if we have a multi-variate dataset?

Estimation of the bandwidth
---------------------------

Using cross-validation to compute the bandwidth
```````````````````````````````````````````````

Let's now assume a distribution a bit more complex::

  >>> from scipy import stats
  >>> import statsmodels.api as sm
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> d1 = stat.norm(-2, 1)
  >>> d2 = stat.norm(1, 0.2)
  >>> data = np.r_[d1.rvs(3000), d2.rvs(3000)]

The default method used for estimation of the bandwidth assumes the data is
following a normal distribution. So if we plot the above distribution we
obtain::

  >>> model = sm.nonparametric.KDE(data)
  >>> est = model.fit()
  >>> grid, freqs = est.grid()
  >>> xs = grid.full()
  >>> plt.plot(xs, freqs, label='Estimate')
  >>> real = (d1.pdf(xs) + d2.pdf(xs))/2
  >>> plt.plot(xs, real, '--', label='Read distribution')
  >>> plt.legend(loc='best')

.. plot:: plots/nonparametric_kde_cv0.py

As we can see, the bandwidth is too large, and the right mode is wider and lower
than it should. A solution for these issues is to use the cross-validation
estimation method. As the method is very slow, we will use a heuristic
known as 'folding'::

  >>> model.bandwidth = sm.nonparametric.bandwidths.leastsquare_cv_bandwidth(imse_args=dict(folding=20, use_grid=True))
  >>> est_bw = model.fit()  # This can be quite long
  >>> grid, freqs = est.grid()
  >>> xs = grid.full()
  >>> plt.plot(xs, freqs, label='Estimate')
  >>> real = (d1.pdf(xs) + d2.pdf(xs))/2
  >>> plt.plot(xs, real, '--', label='Read distribution')
  >>> plt.legend(loc='best')

.. plot:: plots/nonparametric_kde_cv1.py

We can see that, although the noise increase for the larger, smaller mode, the
smaller mode is now well represented.

References
----------
Wasserman, L. All of Nonparametric Statistics Springer, 2005

http://en.wikipedia.org/wiki/Kernel_%28statistics%29

Silverman, B.W.  Density Estimation for Statistics and Data Analysis.

.. autosummary::
   :toctree: generated/

   KDE
