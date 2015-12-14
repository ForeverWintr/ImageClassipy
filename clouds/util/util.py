import os
import errno
from itertools import izip_longest

import plotly.plotly as py
import plotly.graph_objs as go


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def plotlyHistorgram(data):
    """
    Create a plotly histogram of the given data.
    """
    d = [
        go.Histogram(
            x=data
        )
    ]
    plot_url = py.plot(d, filename='basic-histogram')
