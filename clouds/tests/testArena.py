import unittest
import shutil
import tempfile
import os
import mock

import numpy as np

from clouds.obj.subject import Subject
from clouds.obj.arena import Arena
from clouds.obj.classifier import Classifier
from clouds.tests import util

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
        self.sim.simulate(numWorkers=1)
        self.assertAlmostEqual(self.sim.subjects[0].fitness, 100)
