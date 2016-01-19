import unittest
import logging
import io
import multiprocessing

from clouds.util import multiprocess

class testMultiprocess(unittest.TestCase):

    @staticmethod
    def sendLog(x, prefix='a'):
        log = logging.getLogger("testMapWithLogging")
        log.critical("{}{}".format(prefix, x))

    def testMapWithLogging(self):
        """
        Test that log messages from the child processes are successfully recieved by the parent
        processes.
        """
        log = logging.getLogger("testMapWithLogging")
        log.propagate = False

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        log.addHandler(handler)
        log.setLevel(logging.CRITICAL)

        inp = [1, 2, 3, 4, 5]

        #assert that regular pools don't log properly
        p = multiprocessing.Pool()
        p.map(self.sendLog, inp)
        stream.seek(0)
        self.assertFalse(stream.read())

        #assert that mapWithLogging works as a context manager
        with multiprocess.mapWithLogging(self.sendLog, inp, log, 4, 'a') as r:
            r.get()

        stream.seek(0)
        self.assertSequenceEqual(set(stream.read().split()), set(['a{}'.format(x) for x in inp]))


