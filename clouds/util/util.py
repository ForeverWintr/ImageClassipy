import os
import errno
from itertools import zip_longest, chain



def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


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
    Plotly is imported within the function, as importing seems to establish a plotly connection.
    """
    import plotly.plotly as py
    import plotly.graph_objs as go
    d = [
        go.Histogram(
            x=data
        )
    ]
    plot_url = py.plot(d, filename='basic-histogram')


def flatten(iterable):
    return chain.from_iterable(iterable)
