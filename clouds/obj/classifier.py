"""
A classifier is an entity that evaluates images for status (cloudy, canola, etc.)
"""
import os
import time
from operator import mul, itemgetter
from itertools import izip
import codecs
from collections import namedtuple

import PIL.Image
import wand.image
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import trainers
from pybrain import datasets
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml import NetworkWriter, NetworkReader
import numpy as np
import camel

from clouds.util.constants import HealthStatus
from clouds import util

sr = namedtuple('SerializeResult', ['classifier', 'net'])

class Classifier(object):
    _NET_NAME = 'net.xml'
    _CLASSIFIER_NAME = 'classifier.yaml'

    def __init__(self, imageSize=(128, 128), netSpec=(1, ),
                 trainMethod=trainers.BackpropTrainer,
                 datasetMethod=SupervisedDataSet):

        self.netSpec = (mul(*imageSize), ) + netSpec
        self.imageSize = tuple(float(x) for x in imageSize)
        self.trainMethod = trainMethod
        self.datasetMethod = datasetMethod

        #self.net = neurolab.net.newff(self.inputSpec, self.netSpec)
        self.net = buildNetwork(*self.netSpec)

        #statistics
        self.avgCertainty = None
        self.avgClassifyTime = None
        self.numClassifications = 0
        self.trainTime = None
        self.error = None

    def __repr__(self):
        return "<Classifier net {}>".format(self.netSpec)

    def __eq__(self, other):
        if isinstance(other, Classifier):
            return self._comparisonKey() == other._comparisonKey()
        return NotImplemented

    def _comparisonKey(self):
        """
        A tuple of attributes that can be used for comparison.
        """
        return (
            self.imageSize,
            self.netSpec,
            repr(self.trainMethod),
            repr(self.datasetMethod),
        )

    def train(self, images, statuses):
        ds = self.datasetMethod(mul(*self.imageSize), 1)
        [ds.addSample(self._loadToArray(i), e.value) for i, e in izip(images, statuses)]

        trainer = self.trainMethod(self.net, dataset=ds)

        start = time.clock()
        trainErrors, validationErrors = trainer.trainUntilConvergence()

        trainTime = time.clock() - start

        iterations = len(trainErrors) + len(validationErrors)
        print "Training took {} iterations".format(iterations)
        print "Errors: {}, {}".format(trainErrors[-1], validationErrors[-1])

        self.trainTime = float(trainTime) / iterations
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
        newSize = tuple(round(x * s) for x, s in zip(image.size, scaleFactor))
        image.thumbnail(newSize)

        #greyscale
        image = image.convert('L')

        # neurolab seems to expect 1d input, so rescale the images in the
        # input array as linear (the network does't know about shape anyway)
        imageArray = np.array(image)
        newSize = mul(*imageArray.shape)
        return imageArray.reshape(newSize)


    def dump(self, dirPath):
        """
        Save a representation of this classifier and it's network at the given path.
        """
        if os.path.isdir(dirPath) and os.listdir(dirPath):
            raise IOError("The directory exists and is not empty: {}".format(dirPath))
        util.mkdir_p(dirPath)

        #save network
        NetworkWriter.writeToFile(self.net, os.path.join(dirPath, self._NET_NAME))

        #save classifier
        with open(os.path.join(dirPath, self._CLASSIFIER_NAME), 'w') as f:
            f.write(serializer.dump(self))


    @classmethod
    def loadFromDir(cls, dirPath):
        """
        Return a classifier, loaded from the given directory.
        """
        with codecs.open(os.path.join(dirPath, cls._CLASSIFIER_NAME), encoding='utf-8') as f:
            c = serializer.load(f.read())

        c.net = NetworkReader.readFrom(os.path.join(dirPath, cls._NET_NAME))
        return c


classifierRegistry = camel.CamelRegistry()

serializer = camel.Camel((camel.PYTHON_TYPES, classifierRegistry))


####################### DUMPERS #######################

@classifierRegistry.dumper(Classifier, 'Classifier', 1)
def _dumpClassifier(obj):
    return {
        u"imageSize": obj.imageSize,
        u'netSpec': obj.netSpec[1:],
        u'trainMethodName': unicode(obj.trainMethod.__name__),
        u'datasetMethodName': unicode(obj.datasetMethod.__name__),
    }


####################### LOADERS #######################

@classifierRegistry.loader('Classifier', 1)
def _loadClassifier(data, version):
    trainMethod = getattr(trainers.backprop, data.pop('trainMethodName'))
    datasetMethod = getattr(datasets, data.pop('datasetMethodName'))
    data['trainMethod'] = trainMethod
    data['datasetMethod'] = datasetMethod

    return Classifier(**data)
