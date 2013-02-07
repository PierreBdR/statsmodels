"""Create a mosaic plot from a contingency table.

It allows to visualize multivariate categorical data in a rigorous
and informative way.

see the docstring of the mosaic function for more informations.
"""
# Author: Enrico Giampieri - 21 Jan 2013

from __future__ import division

import numpy as np
from statsmodels.compatnp.collections import OrderedDict
from itertools import product

from numpy import iterable, r_, cumsum, array
from statsmodels.graphics import utils

__all__ = ["mosaic"]


def _normalize_split(proportion):
    """
    return a list of proportions of the available space given the division
    if only a number is given, it will assume a split in two pieces
    """
    if not iterable(proportion):
        if proportion == 0:
            proportion = array([0.0, 1.0])
        elif proportion >= 1:
            proportion = array([1.0, 0.0])
        elif proportion < 0:
            raise ValueError("proportions should be positive,"
                              "given value: {}".format(proportion))
        else:
            proportion = array([proportion, 1.0 - proportion])
    proportion = np.asarray(proportion, dtype=float)
    if np.any(proportion < 0):
        raise ValueError("proportions should be positive,"
                          "given value: {}".format(proportion))
    if np.allclose(proportion, 0):
        raise ValueError("at least one proportion should be"
                          "greater than zero".format(proportion))
    # ok, data are meaningful, so go on
    if len(proportion) < 2:
        return array([0.0, 1.0])
    left = r_[0, cumsum(proportion)]
    left /= left[-1] * 1.0
    return left


def _split_rect(x, y, width, height, proportion, horizontal=True, gap=0.05):
    """
    Split the given rectangle in n segments whose proportion is specified
    along the given axis if a gap is inserted, they will be separated by a
    certain amount of space, retaining the relative proportion between them
    a gap of 1 correspond to a plot that is half void and the remaining half
    space is proportionally divided among the pieces.
    """
    x, y, w, h = float(x), float(y), float(width), float(height)
    if (w < 0) or (h < 0):
        raise ValueError("dimension of the square less than"
                          "zero w={} h=()".format(w, h))
    proportions = _normalize_split(proportion)

    # extract the starting point and the dimension of each subdivision
    # in respect to the unit square
    starting = proportions[:-1]
    amplitude = proportions[1:] - starting

    # how much each extrema is going to be displaced due to gaps
    starting += gap * np.arange(len(proportions) - 1)

    # how much the squares plus the gaps are extended
    extension = starting[-1] + amplitude[-1] - starting[0]

    # normalize everything for fit again in the original dimension
    starting /= extension
    amplitude /= extension

    # bring everything to the original square
    starting = (x if horizontal else y) + starting * (w if horizontal else h)
    amplitude = amplitude * (w if horizontal else h)

    # create each 4-tuple for each new block
    results = [(s, y, a, h) if horizontal else (x, s, w, a)
                for s, a in zip(starting, amplitude)]
    return results


def _reduce_dict(count_dict, partial_key):
    """
    Make partial sum on a counter dict.
    Given a match for the beginning of the category, it will sum each value.
    """
    L = len(partial_key)
    count = sum(v for k, v in count_dict.iteritems() if k[:L] == partial_key)
    return count


def _key_splitting(rect_dict, keys, values, key_subset, horizontal, gap):
    """
    Given a dictionary where each entry  is a rectangle, a list of key and
    value (count of elements in each category) it split each rect accordingly,
    as long as the key start with the tuple key_subset.  The other keys are
    returned without modification.
    """
    result = OrderedDict()
    L = len(key_subset)
    for name, (x, y, w, h) in rect_dict.iteritems():
        if key_subset == name[:L]:
            # split base on the values given
            divisions = _split_rect(x, y, w, h, values, horizontal, gap)
            for key, rect in zip(keys, divisions):
                result[name + (key,)] = rect
        else:
            result[name] = (x, y, w, h)
    return result


def _tuplify(obj):
    """convert an object in a tuple of strings (even if it is not iterable,
    like a single integer number)"""
    if np.iterable(obj):
        res = tuple(str(o) for o in obj)
    else:
        res = (str(obj),)
    return res


def _categories_level(keys):
    """use the Ordered dict to implement a simple ordered set
    return each level of each category
    [[key_1_level_1,key_2_level_1],[key_1_level_2,key_2_level_2]]"""
    res = []
    for i in zip(*(keys)):
        tuplefied = _tuplify(i)
        res.append(list(OrderedDict([(j, None) for j in tuplefied])))
    return res


def _hierarchical_split(count_dict, horizontal=True, gap=0.05):
    """
    Split a square in a hierarchical way given a contingency table.

    Hierarchically split the unit square in alternate directions
    in proportion to the subdivision contained in the contingency table
    count_dict.  This is the function that actually perform the tiling
    for the creation of the mosaic plot.  If the gap array has been specified
    it will insert a corresponding amount of space (proportional to the
    unit lenght), while retaining the proportionality of the tiles.

    Parameters
    ----------
    count_dict : dict
        Dictionary containing the contingency table.
        Each category should contain a non-negative number
        with a tuple as index.  It expects that all the combination
        of keys to be representes; if that is not true, will
        automatically consider the missing values as 0
    horizontal : bool
        The starting direction of the split (by default along
        the horizontal axis)
    gap : float or array of floats
        The list of gaps to be applied on each subdivision.
        If the lenght of the given array is less of the number
        of subcategories (or if it's a single number) it will extend
        it with exponentially decreasing gaps

    Returns
    ----------
    base_rect : dict
        A dictionary containing the result of the split.
        To each key is associated a 4-tuple of coordinates
        that are required to create the corresponding rectangle:

            0 - x position of the lower left corner
            1 - y position of the lower left corner
            2 - width of the rectangle
            3 - height of the rectangle
    """
    # this is the unit square that we are going to divide
    base_rect = OrderedDict([(tuple(), (0, 0, 1, 1))])
    # get the list of each possible value for each level
    categories_levels = _categories_level(count_dict.keys())
    L = len(categories_levels)

    # recreate the gaps vector starting from an int
    if not np.iterable(gap):
        gap = [gap / 1.5 ** idx for idx in range(L)]
    # extend if it's too short
    if len(gap) < L:
        last = gap[-1]
        gap = list(*gap) + [last / 1.5 ** idx for idx in range(L)]
    # trim if it's too long
    gap = gap[:L]
    # put the count dictionay in order for the keys
    # this will allow some code simplification
    count_ordered = OrderedDict([(k, count_dict[k])
                        for k in list(product(*categories_levels))])
    for cat_idx, cat_enum in enumerate(categories_levels):
        # get the partial key up to the actual level
        base_keys = list(product(*categories_levels[:cat_idx]))
        for key in base_keys:
            # for each partial and each value calculate how many
            # observation we have in the counting dictionary
            part_count = [_reduce_dict(count_ordered, key + (partial,))
                            for partial in cat_enum]
            # reduce the gap for subsequents levels
            new_gap = gap[cat_idx]
            # split the given subkeys in the rectangle dictionary
            base_rect = _key_splitting(base_rect, cat_enum, part_count, key,
                                       horizontal, new_gap)
        horizontal = not horizontal
    return base_rect


def _single_hsv_to_rgb(hsv):
    """Transform a color from the hsv space to the rgb."""
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(array(hsv).reshape(1, 1, 3)).reshape(3)


def _create_default_properties(data):
    """"Create the default properties of the mosaic given the data
    first it will varies the color hue (first category) then the color
    saturation (second category) and then the color value
    (third category).  If a fourth category is found, it will put
    decoration on the rectangle.  Doesn't manage more than four
    level of categories"""
    categories_levels = _categories_level(data.keys())
    Nlevels = len(categories_levels)
    # first level, the hue
    L = len(categories_levels[0])
    # hue = np.linspace(1.0, 0.0, L+1)[:-1]
    hue = np.linspace(0.0, 1.0, L + 2)[:-2]
    # second level, the saturation
    L = len(categories_levels[1]) if Nlevels > 1 else 1
    saturation = np.linspace(0.5, 1.0, L + 1)[:-1]
    # third level, the value
    L = len(categories_levels[2]) if Nlevels > 2 else 1
    value = np.linspace(0.5, 1.0, L + 1)[:-1]
    # fourth level, the hatch
    L = len(categories_levels[3]) if Nlevels > 3 else 1
    hatch = ['', '/', '-', '|', '+'][:L + 1]
    # convert in list and merge with the levels
    hue = zip(list(hue), categories_levels[0])
    saturation = zip(list(saturation),
                     categories_levels[1] if Nlevels > 1 else [''])
    value = zip(list(value),
                     categories_levels[2] if Nlevels > 2 else [''])
    hatch = zip(list(hatch),
                     categories_levels[3] if Nlevels > 3 else [''])
    # create the properties dictionary
    properties = {}
    for h, s, v, t in product(hue, saturation, value, hatch):
        hv, hn = h
        sv, sn = s
        vv, vn = v
        tv, tn = t
        level = (hn,) + ((sn,) if sn else tuple())
        level = level + ((vn,) if vn else tuple())
        level = level + ((tn,) if tn else tuple())
        hsv = array([hv, sv, vv])
        prop = {'color': _single_hsv_to_rgb(hsv), 'hatch': tv, 'lw': 0}
        properties[level] = prop
    return properties


def _normalize_data(data):
    """normalize the data to a dict with tuples of strings as keys
    right now it works with:

        0 - dictionary (or equivalent mappable)
        1 - pandas.Series with simple or hierarchical indexes
        2 - numpy.ndarrays
    """
    # can it be used as a dictionary?
    try:
        items = data.iteritems()
    except AttributeError:
        # ok, I cannot use the data as a dictionary
        # it should be an ndarray that I will normalize
        # inot a dictionary
        if isinstance(data, np.ndarray):
            temp = OrderedDict()
            for idx in np.ndindex(data.shape):
                name = tuple(i for i in idx)
                temp[name] = data[idx]
            data = temp
            items = data.iteritems()
        else:
            raise TypeError('Data type not recognized, '
                            'should be a dict or pandas.series'
                            ' or np.ndarray')
    # make all the keys a tuple, even if simple numbers
    data = OrderedDict([_tuplify(k), v] for k, v in items)
    categories_levels = _categories_level(data.keys())
    # fill the void in the counting dictionary
    indexes = product(*categories_levels)
    contingency = OrderedDict([(k, data.get(k, 0)) for k in indexes])
    data = contingency
    return data

def _statistical_coloring(data):
    """evaluate colors from the indipendence properties of the matrix
    It will encounter problem if one category has all zeros"""
    data = _normalize_data(data)
    categories_levels = _categories_level(data.keys())
    Nlevels = len(categories_levels)
    total = 1.0 * sum(v for v in data.values())
    # count the proportion of observation
    # for each level that has the given name
    # at each level
    levels_count = []
    for level_idx in range(Nlevels):
        proportion = {}
        for level in categories_levels[level_idx]:
            proportion[level] = 0.0
            for key, value in data.items():
                if level == key[level_idx]:
                    proportion[level] += value
            proportion[level] /= total
        levels_count.append(proportion)
    # for each key I obtain the expected value
    # and it's standard deviation from a binomial distribution
    # under the hipothesys of independence
    expected = {}
    for key, value in data.items():
        base = 1.0
        for i, k in enumerate(key):
            base *= levels_count[i][k]
        expected[key] = base * total, np.sqrt(total * base * (1.0 - base))
    # now we have the standard deviation of distance from the
    # expected value for each tile. We create the colors from this
    sigmas = {k: (data[k] - m) / s for k, (m, s) in expected.items()}
    props = {}
    for key, dev in sigmas.items():
        red = 0.0 if dev < 0 else (dev / (1+dev))
        blue = 0.0 if dev > 0 else (dev / (-1+dev))
        green = (1.0 - red - blue) / 2.0
        hatch = 'x' if dev > 2 else 'o' if dev < -2 else ''
        props[key] = {'color': [red, green, blue], 'hatch': hatch}
    return props


def mosaic(data, ax=None, horizontal=True, gap=0.005,
           properties=lambda key: None, labelizer=None,
           title='', statistic=False):
    """Create a mosaic plot from a contingency table.

    It allows to visualize multivariate categorical data in a rigorous
    and informative way.

    Parameters
    ----------
    data : dict, pandas.Series, np.ndarray
        The contingency table that contains the data.
        Each category should contain a non-negative number
        with a tuple as index.  It expects that all the combination
        of keys to be representes; if that is not true, will
        automatically consider the missing values as 0.  The order
        of the keys will be the same as the one of insertion.
        If a dict of a Series (or any other dict like object)
        is used, it will take the keys as labels.  If a
        np.ndarray is provided, it will generate a simple
        numerical labels.
    ax : matplotlib.Axes, optional
        The graph where display the mosaic. If not given, will
        create a new figure
    horizontal : bool, optional (default True)
        The starting direction of the split (by default along
        the horizontal axis)
    gap : float or array of floats
        The list of gaps to be applied on each subdivision.
        If the lenght of the given array is less of the number
        of subcategories (or if it's a single number) it will extend
        it with exponentially decreasing gaps
    labelizer : function (key) -> string, optional
        A function that generate the text to display at the center of
        each tile base on the key of that tile
    properties : function (key) -> dict, optional
        A function that for each tile in the mosaic take the key
        of the tile and returns the dictionary of properties
        of the generated Rectangle, like color, hatch or similar.
        A default properties set will be provided fot the keys whose
        color has not been defined, and will use color variation to help
        visually separates the various categories. It should return None
        to indicate that it should use the default property for the tile.
        A dictionary of the properties for each key can be passed,
        and it will be internally converted to the correct function
    statistic: bool, optional (default False)
        if true will use a crude statistical model to give colors to the plot.
        If the tile has a containt that is more than 2 standard deviation
        from the expected value under independence hipotesys, it will
        go from green to red (for positive deviations, blue otherwise) and
        will acquire an hatching when crosses the 3 sigma.
    title : string, optional
        The title of the axis

    Returns
    ----------
    fig : matplotlib.Figure
        The generate figure
    rects : dict
        A dictionary that has the same keys of the original
        dataset, that holds a reference to the coordinates of the
        tile and the Rectangle that represent it

    See Also
    ----------
    A Brief History of the Mosaic Display
        Michael Friendly, York University, Psychology Department
        Journal of Computational and Graphical Statistics, 2001

    Mosaic Displays for Loglinear Models.
        Michael Friendly, York University, Psychology Department
        Proceedings of the Statistical Graphics Section, 1992, 61-68.

    Mosaic displays for multi-way contingecy tables.
        Michael Friendly, York University, Psychology Department
        Journal of the american statistical association
        March 1994, Vol. 89, No. 425, Theory and Methods

    Examples
    ----------
    The most simple use case is to take a dictionary and plot the result

    >>> data = {'a': 10, 'b': 15, 'c': 16}
    >>> mosaic(data, title='basic dictionary')
    >>> pylab.show()

    A more useful example is given by a dictionary with multiple indices.
    In this case we use a wider gap to a better visual separation of the
    resulting plot

    >>> data = {('a', 'b'): 1, ('a', 'c'): 2, ('d', 'b'): 3, ('d', 'c'): 4}
    >>> mosaic(data, gap=0.05, title='complete dictionary')
    >>> pylab.show()

    The same data can be given as a simple or hierarchical indexed Series

    >>> rand = np.random.random
    >>> from itertools import product
    >>>
    >>> tuples = list(product(['bar', 'baz', 'foo', 'qux'], ['one', 'two']))
    >>> index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    >>> data = pd.Series(rand(8), index=index)
    >>> mosaic(data, title='hierarchical index series')
    >>> pylab.show()

    The third accepted data structureis the np array, for which a
    very simple index will be created.

    >>> rand = np.random.random
    >>> data = 1+rand((2,2))
    >>> mosaic(data, title='random non-labeled array')
    >>> pylab.show()

    If you need to modify the labeling you can give a function to
    create the labels starting from the key tuple

    >>> data = {'a': 10, 'b': 15, 'c': 16}
    >>> props = lambda key: {'color': 'r' if 'a' in key else 'gray'}
    >>> mosaic(data, title='colored dictionary', properties=props)
    >>> pylab.show()
    """
    from pylab import Rectangle
    fig, ax = utils.create_mpl_ax(ax)
    # normalize the data to a dict with tuple of strings as keys
    data = _normalize_data(data)
    # split the graph into different areas
    rects = _hierarchical_split(data, horizontal=horizontal, gap=gap)
    # if there is no specified way to create the labels
    # create a default one
    if labelizer is None:
        labelizer = lambda k: "\n".join(k)
    if statistic:
        default_props = _statistical_coloring(data)
    else:
        default_props = _create_default_properties(data)
    if isinstance(properties,dict):
        color_dict = properties
        properties = lambda key: color_dict.get(key,None)
    for k, v in rects.items():
        # create each rectangle and put a label on it
        x, y, w, h = v
        conf = properties(k)
        props = conf if conf else default_props[k]
        text = labelizer(k)
        Rect = Rectangle((x, y), w, h, label=text, **props)
        ax.add_patch(Rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center',
                 va='center', size='smaller')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title(title)
    return fig, rects



if __name__ == '__main__':
    import matplotlib.pyplot as pylab

    N = 80
    data = {('a', 'b'): 2 * N, ('a', 'c'): 4 * N,
            ('d', 'b'): 3 * N, ('d', 'c'): 3 * N}


    #data = array([[1520,266,124,66],
    #              [234,1512,432,78],
    #              [117,362,1772,205],
    #              [36,82,179,492]])

    #props = lambda key: {'color': 'r' if 'a' in key else 'gray'}
    mosaic(data, title='basic dictionary',
        statistic=True)
    pylab.show()