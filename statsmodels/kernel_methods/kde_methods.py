"""
:Author: Pierre Barbier de Reuille <pierre.barbierdereuille@gmail.com>

This module contains all the methods for computing the KDE.
"""

from ._kde_methods import KDEMethod, filter_exog  # noqa

from . import _kde1d_methods
from . import _kdenc_methods
from . import _kdend_methods
from . import _kde_multivariate

from ._kde1d_methods import (LogTransform, ExpTransform, Transform, create_transform,  # noqa
                             convolve, generate_grid1d)  # noqa

from ._kdend_methods import generate_grid  # noqa

def _import_methods(module):
    module_variables = globals()
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, KDEMethod):
            module_variables[name] = obj

_import_methods(_kde1d_methods)
_import_methods(_kdenc_methods)
_import_methods(_kdend_methods)
_import_methods(_kde_multivariate)
