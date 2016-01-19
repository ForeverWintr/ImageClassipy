"""
Multiprocessing utilities
"""
import multiprocessing
import logging
from logging import handlers
import contextlib


class mapWithLogging(object):
    #Class attribute variables for passing to child processes
    _ARGS = None
    _KWARGS = None
    _FUNC = None

    def __init__(self, func, iterable, log, *args, workerCount=None, **kwargs):
        """
        Map the supplied function to the supplied `iterable` using `workerCount` workers and
        logging to log. Any additional *args and **kwargs will be passed to each `func`. Returns a
        pool result object. Use like so:

        import logging

        def f(x):
            log.info("x is: {}".format(x))

        mylog=logging.getLogger('myLog')

        # configure the log object, adding handlers, etc.

        with mapWithLogging(f, [1, 2, 3], log) as mapResult:
            mapResult.get()
        """
        self.iterable = iterable

        #set up a queue listener, to write to the worker processes
        logQ = multiprocessing.Queue()
        self.pool = multiprocessing.Pool(workerCount, initializer=self._initializer,
                                    initargs=(logQ, log.name, func, args, kwargs))
        self.qListener = handlers.QueueListener(logQ, *log.handlers)

    def __enter__(self):
        self.qListener.start()
        return self.pool.map_async(self._wrapper, self.iterable)

    def __exit__(self, type, value, traceback):
        self.pool.close()
        self.pool.join()
        self.qListener.stop()

    @classmethod
    def _wrapper(cls, item):
        return cls._FUNC(item, *cls._ARGS, **cls._KWARGS)

    @classmethod
    def _initializer(cls, logQ, loggerName, func, args, kwargs):
        #Set up class attribute variables
        cls._ARGS = args
        cls._KWARGS = kwargs
        cls._FUNC = func

        #set up log to use queue. I think we need to remove existing handlers, so we don't try to log
        #to file from multiple processes. I wonder if there's a better way to do this...
        log = logging.getLogger(loggerName)
        [log.removeHandler(h) for h in log.handlers]

        qHandler = handlers.QueueHandler(logQ)
        log.addHandler(qHandler)

        log.debug("{} initialized".format(multiprocessing.current_process()))

