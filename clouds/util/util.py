import os
import errno
from itertools import zip_longest, chain
import signal
import sys
import contextlib
from threading import Thread

import numpy as np



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


def rgbToHex(array):
    """
    Convert an RGB array to hex values.
    http://stackoverflow.com/a/26227165/1286571
    """
    array = np.asarray(array, dtype='uint32')
    return ((array[:, :, 0]<<16) + (array[:, :, 1]<<8) + array[:, :, 2])


def debuggingEnabled():
    """
    Return true if we're in debug mode with wing.
    """
    if sys.gettrace():
        return True
    if sys.executable.endswith('wingdb.exe'):
        return True
    return False


class GracefulInterruptHandler(object):

    def __init__(self, sig=signal.SIGINT):
        """
        Adapted from: https://gist.github.com/nonZero/2907502
        """
        self.sig = sig

    def handler(self, signum, frame):
        self.release()
        self.interrupted = True

    def __enter__(self):
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        signal.signal(self.sig, self.handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


class SendStopOnInterrupt(GracefulInterruptHandler):
    def __init__(self, queue, message, sig=signal.SIGINT):
        """
        Send the specified `message` to the specifed `queue` if interupted with `sig` signal.
        """
        super().__init__(sig)
        self.queue = queue
        self.message = message

    def handler(self, signum, frame):
        print("RECIEVED SIGNAL")
        self.queue.put(self.message)
        super().handler(signum, frame)


class enqueue(object):
    def __init__(self, stream, queue):
        """
        Monitor `stream` from a separate thread, and put items read onto `queue`. This allows you
        to perform non-blocking reads on `queue`.

        Based on http://stackoverflow.com/a/4896288/1286571
        """
        self.stream = stream
        self.queue = queue
        self.thread = Thread(target=self._enqueue)
        self.thread.daemon = True

    def _enqueue(self):
        for line in iter(self.stream.readline, b''):
            self.queue.put(line)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()

