import os
import unittest
import sys

import mock
import numpy as np

from clouds.constants import HealthStatus
from clouds.obj import classifier


class testTrainClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

        c.net.sim = lambda x: [[0.001]]

        result = c.classify(self.testImages[0])

        self.assertEqual(result, (HealthStatus.GOOD, 0.001))
