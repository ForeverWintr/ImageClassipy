import unittest
import shutil
import tempfile
import os

import numpy as np

from clouds.obj.subject import Subject
from clouds.obj.classifier import Classifier
from clouds.tests.testClassifier import testTrainClassifier

TESTDATA = './data'

class testSubject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.seterr(all='raise', under='warn')
        cls.workspace = tempfile.mkdtemp(prefix="testSubject_")
        cls.storedClassifier = os.path.join(cls.workspace, 'sc')
        shutil.copytree(os.path.join(TESTDATA, 'xorClassifier'), cls.storedClassifier)

        cls.xors = testTrainClassifier.createXors(cls.workspace)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace)

    def testStoreSubject(self):
        d = os.path.join(self.workspace, 'testSubject')
        c = Classifier.loadFromDir(self.storedClassifier)
        s = Subject(d, c)

        tgt = os.path.join(self.workspace, 'storedSubect')

        #write network
        s.dump(tgt)

        #read subject
        loaded = Subject.loadFromDir(tgt)

        self.assertEqual(s, loaded)

        #assert that the xor still works
        for img, status in self.xors:
            self.assertEqual(loaded.classifier.classify(img)[0], status)

        print('Done')

