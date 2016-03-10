"""
This contains tests for the util module (clouds.util). Not test utilities. Those are in
clouds.tests.util.
"""
import unittest
from threading import Thread
from queue import Queue
import io

from clouds import util as realutil
from clouds.tests.util import util as testutil


class testUtils(unittest.TestCase):

    def testEnqueue(self):
        inQ = Queue()
        outQ = Queue()

        mockStream = testutil.MockStream(inQ)

        with realutil.enqueue(mockStream, outQ):
            inQ.put('test')

            #test should appear in outQ
            self.assertEqual(outQ.get(timeout=1), 'test')

