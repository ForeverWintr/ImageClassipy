import os
import unittest
import sys
import tempfile
import shutil
import queue

import mock
import numpy as np
import PIL
from pybrain.tools.customxml import NetworkReader
import camel

from clouds.util.constants import HealthStatus, Command
from clouds.obj import classifier
from clouds.tests import util

TESTDATA = './data'
XOR = os.path.join(TESTDATA, 'xor.xml')

class testTrainClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.seterr(all='raise', under='warn')
        cls.workspace = tempfile.mkdtemp(prefix="testClassifier_")
        cls.storedXor = os.path.join(os.path.dirname(__file__), XOR)

        #find test images
        tiffs = [
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/clouds/tests/data/healthSegments/2015-05-23_RE_75328/rgb/rgb.tif',
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/clouds/tests/data/healthSegments/2015-06-11_RE_109891/rgb/rgb.tif',
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/clouds/tests/data/healthSegments/2015-06-16_RE_112814/rgb/rgb.tif',
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/clouds/tests/data/healthSegments/2015-07-21_RE_162812/rgb/rgb.tif',
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/clouds/tests/data/healthSegments/2015-07-31_RE_195671/rgb/rgb.tif',
        ]

        #Replace tiffs with pngs if they exist
        cls.testImages = []
        for path in tiffs:
            png = os.path.splitext(path)[0] + '.png'
            if os.path.exists(png):
                cls.testImages.append(png)
            else:
                cls.testImages.append(path)

        cls.statuses = [
            HealthStatus.CLOUDY,
            HealthStatus.GOOD,
            HealthStatus.GOOD,
            HealthStatus.CANOLA,
            HealthStatus.GOOD,
        ]

        cls.xorImages = util.createXors(cls.workspace)


    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace)


    def testXOR(self):
        """
        Test that classifier can solve the xor problem. Only sometimes succeeds.
        Not enough data? Too few iterations? Bad net spec?
        """
        c = classifier.Classifier(
            set(list(zip(*self.xorImages))[1]), imageSize=(2, 2), hiddenLayers=(8, ),
            convergenceThreshold=4
        )

        c.train(*list(zip(*self.xorImages)))

        for image, expected in self.xorImages:
            self.assertEqual(c.classify(image)[0], expected)

        print('done')


    def testAbort(self):
        """
        Assert that we stop and save if STOP is recieved in the command queue.
        """
        c = classifier.Classifier(
            set(list(zip(*self.xorImages))[1]), imageSize=(2, 2), hiddenLayers=(8, ),
            convergenceThreshold=4
        )

        commandQ = queue.Queue()
        commandQ.put(Command.STOP)

        c.train(*list(zip(*self.xorImages)), reportInterval=1, commandQ=commandQ)

        self.assertEqual(c.epochsTrained, 1)


    def testTrain(self):
        """
        Just make sure we don't crash
        """
        seed = np.random.randint(2 ** 32)
        #seed = 2095592106 #validation will fail
        print('Seed:', seed)
        np.random.seed(seed)

        possibleStatuses = set(self.statuses)
        c = classifier.Classifier(possibleStatuses, imageSize=(20, 20))

        result = c.train(
            self.testImages, self.statuses, )
        print(result)


    def testClassify(self):
        """
        Return healthstatus and how sure we are.
        """
        c = classifier.Classifier(set(self.statuses), imageSize=(20, 20))

        c.net.activate = lambda x: 0.001

        result = c.classify(self.testImages[0])

        self.assertEqual(result, (HealthStatus.GOOD, 0.001))


    def testSaveClassifier(self):
        """
        Save a network, make sure it's valid.
        """
        xor = NetworkReader.readFrom(self.storedXor)
        c = classifier.Classifier([HealthStatus.GOOD, HealthStatus.CLOUDY],
                                  imageSize=(2, 2), hiddenLayers=(8, ))
        c.net = xor

        storedPath = os.path.join(self.workspace, 'testNetDir')
        c.dump(storedPath, overwrite=False)

        newC = classifier.Classifier.loadFromDir(storedPath)

        self.assertEqual(c, newC)

        #Make sure the net still works
        for image, expected in self.xorImages:
            self.assertEqual(c.classify(image)[0], expected)

