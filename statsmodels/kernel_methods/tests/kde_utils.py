import pytest
import numpy as np
from collections import namedtuple
from .. import kde_methods as km
from ..kde_utils import Grid
from scipy import stats, linalg
from .. import kernels, kde, kde_methods
import scipy

def generate(dist, N, low, high):
    start = dist.cdf(low)
    end = dist.cdf(high)
    xs = np.linspace(1 - start, 1 - end, N)
    return dist.isf(xs)

def generate_nd(dist, N):
    np.random.seed(1)
    return dist.rvs(N)

def generate_nc(dist, N):
    np.random.seed(1)
    return dist.rvs(N)

def generate_multivariate(N, *dists):
    return np.vstack([d.rvs(N) for d in dists]).T

test_method = namedtuple('test_method',
                         ['instance', 'accuracy', 'grid_accuracy',
                          'normed_accuracy', 'bound_low', 'bound_high'])

methods_1d = [test_method(km.KDE1DMethod, 1e-5, 1e-4, 1e-5, False, False),
              test_method(km.Reflection1D, 1e-5, 1e-4, 1e-5, True, True),
              test_method(km.Cyclic1D, 1e-5, 1e-3, 1e-4, True, True),
              test_method(km.Renormalization, 1e-5, 1e-4, 1e-2, True, True),
              test_method(km.LinearCombination, 1e-1, 1e-1, 1e-1, True, False)]
methods_log = [test_method(km.Transform1D(km.LogTransform), 1e-5, 1e-4, 1e-5, True, False)]

methods_nd = [test_method(km.Cyclic, 1e-5, 1e-4, 1e-5, True, True),
              test_method(km.Cyclic, 1e-5, 1e-4, 1e-5, False, False),
              test_method(km.KDEnDMethod, 1e-5, 1e-4, 1e-5, False, False)]

methods_nc = [test_method(km.Ordered, 1e-5, 1e-4, 1e-5, False, False),
              test_method(km.Unordered, 1e-5, 1e-4, 1e-5, False, False)]

test_kernel = namedtuple('test_kernel', ['cls', 'precision_factor', 'var', 'positive'])

kernels1d = [test_kernel(kernels.Gaussian1D, 1, 1, True),
             test_kernel(kernels.TriCube, 1, 1, True),
             test_kernel(kernels.Epanechnikov, 10, 1, True),
             test_kernel(kernels.GaussianOrder4, 10, 0, False),  # Bad for precision because of high frequencies
             test_kernel(kernels.EpanechnikovOrder4, 1000, 0, False)]  # Bad for precision because of high frequencies

kernelsnc = [test_kernel(kernels.AitchisonAitken, 1, 1, True),
             test_kernel(kernels.WangRyzin, 1, 1, True)]

kernelsnd = [test_kernel(kernels.Gaussian, 1, 1, True)]

class Parameters(object):
    """Empty class to hold values."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'Parameters({0}, ...)'.format(name)

dataset = namedtuple('dataset', ['vs', 'weights', 'adjust', 'lower', 'upper'])

def createParams_norm():
    """
    Create the parameters to test using a 1D Gaussian distribution dataset.
    """
    params = Parameters('norm')
    params.dist = stats.norm(0, 1)
    params.sizes = [128, 256, 201]
    params.args = {}
    params.xs = np.r_[-5:5:512j]
    params.methods = methods_1d
    params.can_adjust = True
    def create_dataset():
        vs = [generate(params.dist, s, -5, 5) for s in params.sizes]
        weights = [params.dist.pdf(v) for v in vs]
        adjust = [1 - ws for ws in weights]
        return dataset(vs, weights, adjust, lower=-5, upper=5)
    params.create_dataset = create_dataset
    return params

def createParams_lognorm():
    """
    Create the parameters to test using a 1D log normal distribution dataset.
    """
    params = Parameters('lognorm')
    params.dist = stats.lognorm(1)
    params.sizes = [128, 256, 201]
    params.args = {}
    params.methods = methods_log
    params.xs = np.r_[0:20:512j]
    params.can_adjust = True
    def create_dataset():
        vs = [generate(params.dist, s, 0.001, 20) for s in params.sizes]
        vs = [v[v < 20] for v in vs]
        weights = [params.dist.pdf(v) for v in vs]
        adjust = [1 - ws for ws in weights]
        return dataset(vs, weights, adjust, lower=0, upper=20)
    params.create_dataset = create_dataset
    return params

def createParams_normnd(ndim):
    """
    Create the parameters to test using a nD Gaussian distribution dataset.
    """
    params = Parameters('normnd{0}'.format(ndim))
    params.dist = stats.multivariate_normal(cov=np.eye(ndim))
    params.sizes = [32, 64, 128]
    params.xs = [np.r_[-5:5:512j]] * ndim
    params.methods = methods_nd
    params.args = {}
    params.can_adjust = False
    def create_dataset():
        vs = [generate_nd(params.dist, s) for s in params.sizes]
        weights = [params.dist.pdf(v) for v in vs]
        return dataset(vs, weights, adjust=None, lower=[-5]*ndim, upper=[5]*ndim)
    params.create_dataset = create_dataset
    return params

def createParams_nc():
    """
    Create the parameters to test using  a nC poisson distribution dataset.
    """
    params = Parameters('nc')
    params.dist = stats.poisson(12)
    params.sizes = [128, 256, 201]
    params.args = {}
    params.methods = methods_nc
    params.can_adjust = False
    def create_dataset():
        vs = [generate_nc(params.dist, s) for s in params.sizes]
        weights = [params.dist.pmf(v) for v in vs]
        return dataset(vs, weights, adjust=None, lower=None, upper=None)
    params.create_dataset = create_dataset
    return params

def createParams_multivariate():
    """
    Create the parameters to test using a poisson distribution and two normals as dataset.
    """
    params = Parameters('multivariate')
    params.d1 = stats.norm(0, 3)
    params.d2 = stats.poisson(12)
    params.sizes = [64, 128, 101]
    params.args = {}
    params.methods1 = methods_1d + methods_nc + methods_nc
    params.methods2 = methods_nc + methods_1d + methods_nc[::-1]
    params.methods = zip(params.methods1, params.methods2)
    params.nb_methods = len(params.methods1)
    params.can_adjust = False
    def create_dataset():
        vs = [generate_multivariate(s, params.d1, params.d2) for s in params.sizes]
        weights = [params.d1.pdf(v[:, 0]) for v in vs]
        upper = [5, max(v[:, 1].max() for v in vs)]
        lower = [-5, 0]
        return dataset(vs, weights, adjust=None, lower=lower, upper=upper)
    params.create_dataset = create_dataset
    return params

knownParameters = dict(
    norm = createParams_norm(),
    norm2d = createParams_normnd(2),
    lognorm = createParams_lognorm(),
    nc = createParams_nc(),
    multivariate = createParams_multivariate())

def createKDE(parameters, data, vs, method):
    all_args = dict(parameters.args)
    k = kde.KDE(vs, **all_args)
    if isinstance(method, test_method):
        if method.instance is None:
            del k.method
        else:
            k.method = method.instance
        if method.bound_low:
            k.lower = data.lower
        else:
            del k.lower
        if method.bound_high:
            k.upper = data.upper
        else:
            del k.upper
    else:
        mv = kde_methods.Multivariate()
        k.method = mv

        n = len(method)
        axis_type = ''
        lower = [-np.inf]*n
        upper = [np.inf]*n
        for i, m in enumerate(method):
            method_instance = m.instance()
            mv.methods[i] = method_instance
            axis_type += str(method_instance.axis_type)
            if method_instance.axis_type != 'C':
                vs[:, i] = np.round(vs[:, i])
            if m.bound_low:
                lower[i] = data.lower[i]
            if m.bound_high:
                upper[i] = data.upper[i]
        k.exog = vs
        k.axis_type = axis_type
        k.lower = lower
        k.upper = upper
    return k

def kde_tester(check):
    def fct(self, name, datasets, index, method, with_adjust, with_weights, method_name):
        params = knownParameters[name]
        data = datasets(name)
        k = createKDE(params, data, data.vs[index], method)
        if with_adjust:
            k.adjust = data.adjust[index]
        if with_weights:
            k.weights = data.weights[index]
        # We expect a lot of division by zero, and that is fine.
        with np.errstate(divide='ignore'):
            check(self, k, method, data)
    return fct

kde_tester_args = 'name,index,method,with_adjust,with_weights,method_name'

def make_name(method):
    if isinstance(method, test_method):
        return method.instance.name
    return ":".join(m.instance.name for m  in method)

def generate_methods_data(parameter_names, indices=None):
    """Generate the set of parameters needed to create tests for all these types of distributions."""
    global parameters
    result = []
    for name in parameter_names:
        params = knownParameters[name]
        if indices is None:
            local_indices = range(len(params.sizes))
        else:
            try:
                local_indices = indices[name]
            except (KeyError, TypeError):
                local_indices = indices
        adjusts = [False, True] if params.can_adjust else [False]
        result += [(name, index, method, with_adjust, with_weights, make_name(method))
                    for index in local_indices
                    for method in params.methods
                    for with_adjust in adjusts
                    for with_weights in [False, True]]
    return result

@pytest.fixture(scope='class')
def datasets(request):
    request.cls.datasets_cache = {}
    def make(name):
        cache = request.cls.datasets_cache
        if name not in cache:
            cache[name] = knownParameters[name].create_dataset()
        return cache[name]
    return make

