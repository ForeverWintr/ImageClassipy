"""
A classifier is an entity that evaluates images for status (cloudy, canola, etc.)
"""
import os
import time
from operator import mul, itemgetter
from itertools import izip

import PIL.Image
import wand.image
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import trainers
from pybrain.datasets import SupervisedDataSet
import numpy as np

from clouds.util.constants import HealthStatus


class Classifier(object):

    def __init__(self, imageSize=(128, 128), netSpec=(1, ),
                 trainMethod=trainers.BackpropTrainer,
                 trainDataset=SupervisedDataSet):

        self.netSpec = [mul(*imageSize)] + netSpec
        self.imageSize = tuple(float(x) for x in imageSize)
        self.trainMethod = trainMethod
        self.trainDataset = trainDataset

        #self.net = neurolab.net.newff(self.inputSpec, self.netSpec)
        self.net = buildNetwork(*self.netSpec)

        #statistics
        self.avgCertainty = None
        self.avgClassifyTime = None
        self.numClassifications = 0
        self.trainTime = None
        self.error = None

    def __repr__(self):
        return "<Classifier net {}>".format([self.net.ci]+self.netSpec)

    def train(self, images, statuses):
        ds = self.trainDataset(mul(*self.imageSize), 1)
        [ds.addSample(self._loadToArray(i), e.value) for i, e in izip(images, statuses)]

        trainer = self.trainMethod(self.net, dataset=ds)

        start = time.clock()
        trainErrors, validationErrors = trainer.trainUntilConvergence()

        trainTime = time.clock() - start

        self.trainTime = float(trainTime) / (len(trainErrors) + len(validationErrors))
        self.error = validationErrors[-1]
        return trainErrors, validationErrors


    def classify(self, imagePath):
        """
        Return a HealthStatus enum and a measure of our certainty.
        """
        start = time.clock()
        guess = self.net.activate(self._loadToArray(imagePath))

        print "Guess is", guess

        possibleStatuses = HealthStatus._value2member_map_

        closestStatus = min(
            [(abs(guess - s), s) for s in possibleStatuses.keys()], key=itemgetter(0))

        self.numClassifications += 1
        self.avgClassifyTime = time.clock() - start / self.numClassifications

        return possibleStatuses[closestStatus[1]], closestStatus[0]


    def _loadToArray(self, imagePath):
        """
        Creates input array. Applies scale factor to each image.
        """
        try:
            image = PIL.Image.open(imagePath)
        except IOError as e:
            #print("Trying to open by converting to png")
            png = os.path.splitext(imagePath)[0] + '.png'
            wand.image.Image(filename=imagePath).convert('PNG').save(filename=png)
            image = PIL.Image.open(png)

        #resize
        scaleFactor = np.divide(self.imageSize, image.size)
        newSize = tuple(x * s for x, s in zip(image.size, scaleFactor))
        image.thumbnail(newSize)

        #greyscale
        image = image.convert('L')

        # neurolab seems to expect 1d input, so rescale the images in the
        # input array as linear (the network does't know about shape anyway)
        imageArray = np.array(image)
        newSize = mul(*imageArray.shape)
        return imageArray.reshape(newSize)


