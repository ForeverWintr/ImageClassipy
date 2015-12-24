"""
Multiprocessing utilities
"""
import multiprocessing
import logging
from logging import handlers


def mapWithLogging(func, iterable, log, workerCount, *args, **kwargs):
    """
    Map the supplied function to the supplied `iterable` using `workerCount` workers and logging to
    log. Any additional *args and **kwargs will be passed to each `func`.
    """
    #set up a queue listener, to write to the worker processes
    logQ = multiprocessing.Queue()
    pool = multiprocessing.Pool(workerCount, initializer=_initializer,
                                initargs=(logQ, log.name, func, args, kwargs))
    qListener = handlers.QueueListener(logQ, *log.handlers)
    qListener.start()

    try:
        return pool.map(_wrapper, iterable)
    finally:
        pool.close()
        pool.join()
        qListener.stop()


def _wrapper(item):
    return _FUNC(item, *_ARGS, **_KWARGS)


#global variables for passing to child processes
_ARGS = None
_KWARGS = None
_FUNC = None
def _initializer(logQ, loggerName, func, args, kwargs):
    #Set up global variables
    global _ARGS
    global _KWARGS
    global _FUNC
    _ARGS = args
    _KWARGS = kwargs
    _FUNC = func

    #set up log to use queue. I think we need to remove existing handlers, so we don't try to log
    #to file from multiple processes. I wonder if there's a better way to do this...
    log = logging.getLogger(loggerName)
    [log.removeHandler(h) for h in log.handlers]

    qHandler = handlers.QueueHandler(logQ)
    log.addHandler(qHandler)

    log.debug("{} initialized".format(multiprocessing.current_process()))

