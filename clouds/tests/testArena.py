import unittest
import shutil
import tempfile
import os
import mock
import json
import subprocess
import sys
import signal
from queue import Queue

import numpy as np

from clouds.obj.subject import Subject
from clouds.obj.arena import Arena
from clouds.obj.classifier import Classifier
from clouds.tests import util
from clouds.tests.util import abortableSim
from clouds.util import enqueue

TESTDATA = './data'

class testArena(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.seterr(all='raise', under='warn')
        cls.workspace = tempfile.mkdtemp(prefix="testSim_")
        cls.storedClassifier = os.path.join(cls.workspace, 'sc')
        shutil.copytree(os.path.join(TESTDATA, 'xorClassifier'), cls.storedClassifier)

        cls.xors = {x[0]: x[1] for x in util.createXors(cls.workspace)}
        cls.sim = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace)

    def setUp(self):
        c = Classifier.loadFromDir(self.storedClassifier)

        #Use mock to replace the long running method randomClassifier with one that just returns
        #our xor classifier.
        with mock.patch.object(Arena, 'randomClassifier', return_value=c) as m:
            self.sim = Arena(workingDir=os.path.join(self.workspace, 'sim'),
                             images=self.xors)
            self.sim.spawnSubjects(1)

    def tearDown(self):
        shutil.rmtree(self.sim.workingDir)
        self.sim = None

    def testSimulate(self):
        """
        Test a run of the xor problem.
        """
        self.sim.simulate(numWorkers=1, reportEpochs=10)
        self.assertAlmostEqual(self.sim.subjects[0].fitness, 100)

    def testAbort(self):
        """
        Assert that a simulation can be aborted, and its subjects will be saved.
        """
        #mock pybrain's trainuntilconvergence to sleep a while
        jsonImages = json.dumps(self.xors)

        #assert that classifier has no trained epochs initially
        c = Classifier.loadFromDir(os.path.join(self.workspace, 'sim', 'Subject_0', 'classifier'))
        self.assertEqual(c.epochsTrained, 0)

        with subprocess.Popen([sys.executable, abortableSim.__file__, self.sim.workingDir, '1',
                               jsonImages], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as proc:
            simOut = Queue()
            with enqueue(proc.stdout, simOut):
                #wait for the first print statement
                self.assertIn("Waiting for interrupt signal", simOut.get().decode())

                #send abort
                proc.send_signal(signal.SIGINT)

                #Give the process some time to exit
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.fail("Simulation failed to exit within 5 seconds")

            #assert that the exit code was zero
            self.assertEqual(proc.poll(), 0, "Simulation failed to exit cleanly")

        #check that the classifier was modified
        c = Classifier.loadFromDir(os.path.join(self.workspace, 'sim', 'Subject_0', 'classifier'))
        self.assertEqual(c.epochsTrained, 1)
        pass
