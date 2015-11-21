import os
import unittest
import sys
import tempfile
import shutil

import mock
import numpy as np
import PIL

from clouds.util.constants import HealthStatus
from clouds.obj import classifier


class testTrainClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.mkdtemp(prefix="testClassifier_")

        #find test images
        tiffs = [
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/healthSegments/2015-05-23_RE_75328/rgb/rgb.tif',
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/healthSegments/2015-06-11_RE_109891/rgb/rgb.tif',
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/healthSegments/2015-06-16_RE_112814/rgb/rgb.tif',
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/healthSegments/2015-07-21_RE_162812/rgb/rgb.tif',
            '/Users/tomrutherford/Dropbox/Code/Wing/clouds/healthSegments/2015-07-31_RE_195671/rgb/rgb.tif',
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

        #create test xor images
        xorIn = [
            ((255, 255, 255, 255), HealthStatus.GOOD),
            ((255, 255, 0, 0), HealthStatus.CLOUDY),
            ((0, 0, 0, 0), HealthStatus.GOOD),
            ((0, 0, 255, 255), HealthStatus.CLOUDY),
        ]
        cls.xorImages = []
        for ar, expected in xorIn:
            npar = np.array(ar, dtype=np.uint8).reshape(2, 2)

            image = PIL.Image.fromarray(npar)

            #pybrain needs a lot of test input. We'll make 20 of each image
            for i in range(20):
                path = tempfile.mktemp(suffix=".png", prefix='xor_', dir=cls.workspace)
                image.save(path)
                cls.xorImages.append((path, expected))


    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace)


    def testXOR(self):
        """
        Test that classifier can solve the xor problem
        """
        c = classifier.Classifier(imageSize=(2, 2), netSpec=(8, 1))

        c.train(*zip(*self.xorImages))

        for image, expected in self.xorImages:
            self.assertEqual(c.classify(image)[0], expected)


    def testTrain(self):
        """
        Just make sure we don't crash
        """
        seed = np.random.randint(2 ** 32)
        #seed = 3558371868 # 2
        #seed = 3547349022 # works
        print 'Seed:', seed
        np.random.seed(seed)

        c = classifier.Classifier(imageSize=(20, 20))

        result = c.train(self.testImages, self.statuses)
        print result


    def testClassify(self):
        """
        Return healthstatus and how sure we are.
        """
        c = classifier.Classifier(imageSize=(20, 20))

        c.net.activate = lambda x: 0.001

        result = c.classify(self.testImages[0])

        self.assertEqual(result, (HealthStatus.GOOD, 0.001))
