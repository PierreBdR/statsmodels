.. currentmodule:: statsmodels.nonparametric.kde1d_methods

.. _nonparametric_kde1d_methods:

1D Methods for Kernel Density Estimation :mod:`kde1d_methods`
=============================================================

This module contains a set of methods to compute univariate KDEs.

:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

These methods provide various variations on :math:`\hat{K}(x;X,h,L,U)`, the
modified kernel evaluated on the point :math:`x` based on the estimation points
:math:`X`, a bandwidth :math:`h` and on the domain :math:`[L,U]`.

The definitions of the methods rely on the following definitions:

.. math::

   \begin{array}{rcl}
     a_0(l,u) &=& \int_l^u K(z) dz\\
     a_1(l,u) &=& \int_l^u zK(z) dz\\
     a_2(l,u) &=& \int_l^u z^2K(z) dz
   \end{array}

These definitions correspond to:

- :math:`a_0(l,u)` -- The partial cumulative distribution function
- :math:`a_1(l,u)` -- The partial first moment of the distribution. In
  particular, :math:`a_1(-\infty, \infty)` is the mean of the kernel (i.e. and
  should be 0).
- :math:`a_2(l,u)` -- The partial second moment of the distribution. In
  particular, :math:`a_2(-\infty, \infty)` is the variance of the kernel (i.e.
  which should be close to 1, unless using higher order kernel).

Estimation Methods
------------------

Default: :py:class:`KDE1DMethod`
````````````````````````````````

The default method can only estimate the KDE for unbounded domains. The
probability density function is estimated by an explicit convolution performed
on the evaluation points.

:py:class:`Cyclic`
``````````````````

This method can be used for unbounded or closed domains. But it cannot be used
if the domain is bounded on only one side (otherwise, what would cyclic mean?).

If there is no per-point adjustment, the convolutions are done using the FFT
transform for efficiency. If the domain is un-bounded, then the function is
still considered cyclic, but with a domain large enough that it shouldn't influence
the result. The `cut` argument to the grid methods can be used to change what
"large enough" means.

:py:class:`Reflection`
``````````````````````

This method can be used on all domains. If one side of the domain is closed, it
will consider the data is reflected on it.

If there is no per-point adjustment, the convolutions are done using the CDT
transform for efficiency. If a side of the domain is un-bounded, then the
function is still considered reflective, but choosing a boundary far enough from
the data so it shouldn't influence the result. Like for the cyclic method, the
`cut` argument to the grid methods can be used to change what "large enough"
means.

:py:class:`Renormalization`
```````````````````````````

The renormalization method is described in [1] and is way to correct for
boundary errors. In short, it adjust the re-normalize the values obtained,
depending on the distance from the point to the domain boundary.

When possible, the grid methods will perform the convolutions using the FFT
transformation on a domain slightly larger than required.

:py:class:`TransformKDE`
````````````````````````

This method takes a fully different approach to boundary correction. The idea is
to transform the data so it is on an un-bounded domain, or at least on a domain
on which we know how to correct for the boundary problems.

:py:class:`LinearCombination`
`````````````````````````````

While the renormalization method was found by estimating the bias of the normal
KDE at the boundaries, this one was found using a first order approximation of
the same bias. The result is a method that converges as fast on the boundaries
as far from them. However, unlike most other methods, this one can produce
functions with values slightly negatives or whose sum is slightly above 1.


Module Classes
--------------

.. autosummary::
   :toctree: generated/

   KDE1DMethod
   Cyclic
   Reflection
   Renormalization
   LinearCombination
   TransformKDE
   LogTransform
   ExpTransform

References:
-----------
.. [1] Jones, M. C. 1993. Simple boundary correction for kernel density
    estimation. Statistics and Computing 3: 135--146.

