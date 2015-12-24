import unittest
import shutil
import tempfile
import os
import mock

import numpy as np

from clouds.obj.genetics import Subject, Simulation
from clouds.obj.classifier import Classifier
from clouds.tests.testClassifier import testTrainClassifier

TESTDATA = './data'

class testSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.seterr(all='raise', under='warn')
        cls.workspace = tempfile.mkdtemp(prefix="testSim_")
        cls.storedClassifier = os.path.join(cls.workspace, 'sc')
        shutil.copytree(os.path.join(TESTDATA, 'xorClassifier'), cls.storedClassifier)

        cls.xors = {x[0]: x[1] for x in testTrainClassifier.createXors(cls.workspace)}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace)


    def testRunSubject(self):
        """
        Test a run of the xor problem.
        """
        c = Classifier.loadFromDir(self.storedClassifier)

        #Use mock to replace the long running method createClassifier with one that just returns
        #our xor classifier.
        with mock.patch.object(Simulation, 'createClassifier', return_value=c) as m:
            sim = Simulation(workingDir=os.path.join(self.workspace, 'sim'), subjectCount=1,
                             images=self.xors)

        sim.simulate(numWorkers=1)
        print(asdf)

