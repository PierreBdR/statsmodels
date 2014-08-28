r"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

Module providing a set of kernels for use with either the :py:mod:`pyqt_fit.kde` or the :py:mod:`kernel_smoothing` 
module.

Kernels should be created following this template:

"""
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.special import erf
from scipy import fftpack, integrate
from .kde_utils import make_ufunc, numpy_method_idx
from . import _kernels
from copy import copy as shallowcopy

S2PI = np.sqrt(2 * np.pi)


S2 = np.sqrt(2)

class Kernel1D(object):
    r"""
    A 1D kernel :math:`K(z)` is a function with the following properties:

    .. math::

        \begin{array}{rcl}
        \int_\mathbb{R} K(z) &=& 1 \\
        \int_\mathbb{R} zK(z)dz &=& 0 \\
        \int_\mathbb{R} z^2K(z) dz &<& \infty \quad (\approx 1)
        \end{array}

    Which translates into the function should have:

    - a sum of 1 (i.e. a valid density of probability);
    - an average of 0 (i.e. centered);
    - a finite variance. It is even recommanded that the variance is close to 1 to give a uniform meaning to the 
      bandwidth.

    .. py:attribute:: cut

        :type: float

        Cutting point after which there is a negligeable part of the probability. More formally, if :math:`c` is the 
        cutting point:

        .. math::

            \int_{-c}^c p(x) dx \approx 1

    .. py:attribute:: lower

        :type: float

        Lower bound of the support of the PDF. Formally, if :math:`l` is the lower bound:

        .. math::

            \int_{-\infty}^l p(x)dx = 0

    .. py:attribute:: upper

        :type: float

        Upper bound of the support of the PDF. Formally, if :math:`u` is the upper bound:

        .. math::

            \int_u^\infty p(x)dx = 0

    """
    cut = 3.
    lower = -np.inf
    upper = np.inf

    def for_ndim(self, ndim):
        """
        Create the same kernel but for a different number of dimensions
        """
        assert ndim == 1, "Error, this kernel only works in 1D"
        return self

    def pdf(self, z, out=None):
        r"""
        Returns the density of the kernel on the points `z`. This is the funtion :math:`K(z)` itself.

        :param ndarray z: Array of points to evaluate the function on. The method should accept any shape of array.
        :param ndarray out: If provided, it will be of the same shape as `z` and the result should be stored in it.
            Ideally, it should be used for as many intermediate computation as possible.
        """
        raise NotImplementedError()

    def __call__(self, z, out=None):
        """
        Alias for :py:meth:`Kernel1D.pdf`
        """
        return self.pdf(z, out=out)

    def cdf(self, z, out=None):
        r"""
        Returns the cumulative density function on the points `z`, i.e.:

        .. math::

            K_0(z) = \int_{-\infty}^z K(t) dt
        """
        z = np.asfarray(z)
        try:
            comp_cdf = self.__comp_cdf
        except AttributeError:
            lower = self.lower
            upper = self.upper
            @make_ufunc()
            def comp_cdf(x):
                if x <= lower:
                    return 0
                if x >= upper:
                    x = upper
                return integrate.quad(self.pdf, lower, x)[0]
            self.__comp_cdf = comp_cdf
        if out is None:
            out = np.empty(z.shape, dtype=float)
        return comp_cdf(z, out=out)

    def pm1(self, z, out=None):
        r"""
        Returns the first moment of the density function, i.e.:

        .. math::

            K_1(z) = \int_{-\infty}^z z K(t) dt
        """
        z = np.asfarray(z)
        try:
            comp_pm1 = self.__comp_pm1
        except AttributeError:
            lower = self.lower
            upper = self.upper
            def pm1(x):
                return x * self.pdf(x)
            @make_ufunc()
            def comp_pm1(x):
                if x <= lower:
                    return 0
                if x > upper:
                    x = upper
                return integrate.quad(pm1, lower, x)[0]
            self.__comp_pm1 = comp_pm1
        if out is None:
            out = np.empty(z.shape, dtype=float)
        return comp_pm1(z, out=out)

    def pm2(self, z, out=None):
        r"""
        Returns the second moment of the density function, i.e.:

        .. math::

            K_2(z) = \int_{-\infty}^z z^2 K(t) dt
        """
        z = np.asfarray(z)
        try:
            comp_pm2 = self.__comp_pm2
        except AttributeError:
            lower = self.lower
            upper = self.upper
            def pm2(x):
                return x * x * self.pdf(x)
            @make_ufunc()
            def comp_pm2(x):
                if x <= lower:
                    return 0
                if x > upper:
                    x = upper
                return integrate.quad(pm2, lower, x)[0]
            self.__comp_pm2 = comp_pm2
        if out is None:
            out = np.empty(z.shape, dtype=float)
        return comp_pm2(z, out=out)

    def fft(self, z, out=None):
        """
        FFT of the kernel on the points of ``z``. The points will always be 
        provided as a regular grid spanning the frequencies wanted.
        """
        period = np.pi / (z[1] - z[0])
        dz, step = np.linspace(-period, period, 2*len(z)-1, endpoint=False, retstep=True)
        dz = np.roll(dz, len(z))
        pdf = self.pdf(dz)
        pdf *= step
        if out is None:
            out = np.empty(z.shape, dtype=complex)
        out[:] = np.fft.rfft(pdf)
        return out

    def fft_xfx(self, z, out=None):
        """
        FFT of the function :math:`x k(x)`. The points are given as for the fft function.
        """
        period = np.pi / (z[1] - z[0])
        dz, step = np.linspace(-period, period, 2*len(z)-1, endpoint=False, retstep=True)
        dz = np.roll(dz, len(z))
        pdf = self.pdf(dz)
        pdf *= dz
        pdf *= step
        if out is None:
            out = np.empty(z.shape, dtype=complex)
        out[:] = np.fft.rfft(pdf)
        return out

    def dct(self, z, out=None):
        r"""
        DCT of the kernel on the points of ``z``. The points will always be provided as a grid with :math:`2^n` points, 
        representing the whole frequency range to be explored.
        """
        z = np.asfarray(z)
        a1 = z[1] - z[0]
        gp = (z / a1 + 0.5) * np.pi / (len(z) * a1)
        out = self.pdf(gp, out=out)
        out *= gp[1] - gp[0]
        out[:] = fftpack.dct(out, overwrite_x = True)
        return out

class normal_kernel1d(Kernel1D):
    """
    1D normal density kernel with extra integrals for 1D bounded kernel estimation.
    """

    def for_ndim(self, ndim):
        if ndim == 1:
            return self
        return normal_kernel(ndim)

    def pdf(self, z, out=None):
        r"""
        Return the probability density of the function. The formula used is:

        .. math::

            \phi(z) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}

        :param ndarray xs: Array of any shape
        :returns: an array of shape identical to ``xs``
        """
        return _kernels.norm1d_pdf(z, out)

    def _pdf(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.pdf`
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape, dtype=z.dtype)
        np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        out /= S2PI
        return out

    __call__ = pdf

    def fft(self, z, out=None):
        """
        Returns the FFT of the normal distribution
        """
        z = np.asfarray(z)
        out = np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        return out

    def fft_xfx(self, z, out=None):
        r"""
        The FFT of :math:`x\mathcal{N}(x)` which is:

        .. math::

            \text{FFT}(x \mathcal{N}(x)) = -e^{-\frac{\omega^2}{2}}\omega i
        """
        z = np.asfarray(z)
        if out is None:
            out = np.empty(z.shape, dtype=complex)
        np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        out *= -z
        out *= 1j
        return out

    def dct(self, z, out=None):
        """
        Returns the DCT of the normal distribution
        """
        z = np.asfarray(z)
        out = np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        return out

    def cdf(self, z, out=None):
        r"""
        Cumulative density of probability. The formula used is:

        .. math::

            \text{cdf}(z) \triangleq \int_{-\infty}^z \phi(z)
                dz = \frac{1}{2}\text{erf}\left(\frac{z}{\sqrt{2}}\right) + \frac{1}{2}
        """
        return _kernels.norm1d_cdf(z, out)

    def _cdf(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.cdf`
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape, dtype=z.dtype)
        np.divide(z, S2, out)
        erf(out, out)
        out *= 0.5
        out += 0.5
        return out

    def pm1(self, z, out=None):
        r"""
        Partial moment of order 1:

        .. math::

            \text{pm1}(z) \triangleq \int_{-\infty}^z z\phi(z) dz
                = -\frac{1}{\sqrt{2\pi}}e^{-\frac{z^2}{2}}
        """
        return _kernels.norm1d_pm1(z, out)

    def _pm1(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.pm1`
        """
        z = np.asarray(z)
        if out is None:
            out = np.empty(z.shape, dtype=z.dtype)
        np.multiply(z, z, out)
        out *= -0.5
        np.exp(out, out)
        out /= -S2PI
        return out

    def pm2(self, z, out=None):
        r"""
        Partial moment of order 2:

        .. math::

            \text{pm2}(z) \triangleq \int_{-\infty}^z z^2\phi(z) dz
                = \frac{1}{2}\text{erf}\left(\frac{z}{2}\right) - \frac{z}{\sqrt{2\pi}}
                e^{-\frac{z^2}{2}} + \frac{1}{2}
        """
        return _kernels.norm1d_pm2(z, out)

    def _pm2(self, z, out=None):
        """
        Full-python implementation of :py:func:`normal_kernel1d.pm2`
        """
        z = np.asarray(z, dtype=float)
        if out is None:
            out = np.empty(z.shape)
        np.divide(z, S2, out)
        erf(out, out)
        out /= 2
        if z.shape:
            zz = np.isfinite(z)
            sz = z[zz]
            out[zz] -= sz * np.exp(-0.5 * sz * sz) / S2PI
        elif np.isfinite(z):
            out -= z * np.exp(-0.5 * z * z) / S2PI
        out += 0.5
        return out

class KernelnD(object):
    cut = 3.
    lower = -np.inf
    upper = np.inf

    def __init__(self, ndim=2):
        self._ndim = ndim

    @property
    def ndim(self):
        return self._ndim

    def for_ndim(self, ndim):
        """
        Create a version of the same kernel, but for dimension ``ndim``
        """
        if ndim == self.ndim:
            return self
        new_ker = shallowcopy(self)
        new_ker._ndim = ndim
        return new_ker

    def pdf(self, y, out=None):
        r"""
        Returns the density of the kernel on the points `z`. This is the funtion :math:`K(z)` itself.

        Parameters
        ----------
        z: ndarray
            Array of points to evaluate the function on. This should be at least a 2D array, with the last dimension 
            corresponding to the dimension of the problem.
        out: ndarray
            If provided, it will be of the same shape as `z` and the result should be stored in it. Ideally, it should 
            be used for as many intermediate computation as possible.
        """
        raise NotImplementedError()

    def __call__(self, z, out=None):
        """
        Alias for :py:meth:`KernelnD.pdf`
        """
        return self.pdf(z, out=out)

    def cdf(self, z, out=None):
        z = np.asfarray(z)
        try:
            comp_cdf = self.__comp_cdf
        except AttributeError:
            def pdf(*xs):
                return self.pdf(xs)
            lower = self.lower
            upper = self.upper
            ndim = self.ndim
            @make_ufunc(ndim)
            def comp_cdf(*xs):
                if any(x <= lower for x in xs):
                    return 0
                xs = np.minimum(xs, upper)
                return integrate.nquad(pdf, [(lower, x) for x in xs])[0]
            self.__comp_cdf = comp_cdf
        if out is None:
            out = np.empty(z.shape[1:], dtype=float)
        return comp_cdf(*z, out=out)

class normal_kernel(KernelnD):
    """
    Returns a function-object for the PDF of a Normal kernel of variance
    identity and average 0 in dimension ``dim``.
    """
    cut = 3

    def for_ndim(self, ndim):
        """
        Create the same kernel but for a different number of dimensions
        """
        if ndim == 1:
            return normal_kernel1d()
        return normal_kernel(ndim)

    def __init__(self, dim=2):
        super(normal_kernel, self).__init__(dim)
        self.factor = 1 / np.sqrt(2 * np.pi) ** dim

    def pdf(self, xs, out=None):
        """
        Return the probability density of the function.

        :param ndarray xs: Array of shape (D,N) where D is the dimension of the kernel
            and N the number of points.
        :returns: an array of shape (N,) with the density on each point of ``xs``
        """
        xs = np.asarray(xs)
        if out is None:
            out = np.empty(xs.shape[1:], dtype=float)
        np.sum(xs*xs, axis=0, out=out)
        out *= -0.5
        np.exp(out, out=out)
        out *= self.factor
        return out

    def cdf(self, xs, out=None):
        """
        Return the CDF of the normal kernel
        """
        xs = np.asarray(xs)
        if out is None:
            out = np.empty(xs.shape[1:], dtype=float)
        tmp = erf(xs / np.sqrt(2))
        tmp += 1
        np.prod(tmp, axis=0, out=out)
        out /= 2**self.ndim
        return out

    __call__ = pdf


from .kernels1d import *
from .kernelsnd import *

kernels1D = [normal_kernel1d, tricube, Epanechnikov, Epanechnikov_order4, normal_order4]
kernelsnD = [normal_kernel]

