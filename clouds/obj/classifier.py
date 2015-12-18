"""
A classifier is an entity that evaluates images for status (cloudy, canola, etc.)
"""
import os
import time
from operator import mul, itemgetter
import codecs
from collections import namedtuple

import PIL.Image
import wand.image
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import trainers
from pybrain import datasets
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.customxml import NetworkWriter, NetworkReader
from pybrain.structure.modules import SoftmaxLayer
import numpy as np
import camel

from clouds.util.constants import HealthStatus, healthStatusRegistry
from clouds import util


class Classifier(object):
    _NET_NAME = 'net.xml'
    _CLASSIFIER_NAME = 'classifier.yaml'

    def __init__(self, possible_statuses, imageSize=(128, 128), hiddenLayers=None,
                 trainMethod=trainers.BackpropTrainer,
                 datasetMethod=ClassificationDataSet,
                 outclass=SoftmaxLayer):
        hiddenLayers = hiddenLayers or tuple()

        self.possibleStatuses = possible_statuses
        self.netSpec = (mul(*imageSize), ) + hiddenLayers + (len(possible_statuses), )
        self.imageSize = tuple(float(x) for x in imageSize)
        self.trainMethod = trainMethod
        self.datasetMethod = datasetMethod

        #self.net = neurolab.net.newff(self.inputSpec, self.netSpec)
        self.net = buildNetwork(*self.netSpec, outclass=outclass)

        #statistics
        self.avgCertainty = None
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
        numStatuses = len(self.possibleStatuses)
        ds = self.datasetMethod(mul(*self.imageSize), 1, numStatuses)
        [ds.addSample(self._loadToArray(i), e.value) for i, e in zip(images, statuses)]

        #convert to one output per class. Apparently this is a better format?
        # http://pybrain.org/docs/tutorial/fnn.html
        ds._convertToOneOfMany()

        trainer = self.trainMethod(self.net, dataset=ds)

        start = time.clock()
        trainErrors, validationErrors = trainer.trainUntilConvergence(convergence_threshold=4)

        trainTime = time.clock() - start

        iterations = len(trainErrors) + len(validationErrors)
        print("Training took {} iterations".format(iterations))
        if trainErrors:
            print("Errors: {}, {}".format(trainErrors[-1], validationErrors[-1]))
        else:
            print("Training unsuccesfull. Trainerrors is empty.")

        self.trainTime = float(trainTime) / iterations
        self.error = validationErrors[-1]
        return trainErrors, validationErrors


    def classify(self, imagePath):
        """
        Return a HealthStatus enum and a measure of our certainty.
        """
        start = time.clock()
        result = self.net.activate(self._loadToArray(imagePath))

        print("Result is:", result)

        guess = HealthStatus._value2member_map_[np.argmax(result)]

        return guess, result


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
        #return imageArray


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

serializer = camel.Camel((camel.PYTHON_TYPES, classifierRegistry, healthStatusRegistry))


####################### DUMPERS #######################

@classifierRegistry.dumper(Classifier, 'Classifier', 1)
def _dumpClassifier(obj):
    return {
        'possible_statuses': obj.possibleStatuses,
        "imageSize": obj.imageSize,
        'hiddenLayers': obj.netSpec[1:-1],
        'trainMethodName': str(obj.trainMethod.__name__),
        'datasetMethodName': str(obj.datasetMethod.__name__),
    }


####################### LOADERS #######################

@classifierRegistry.loader('Classifier', 1)
def _loadClassifier(data, version):
    trainMethod = getattr(trainers.backprop, data.pop('trainMethodName'))
    datasetMethod = getattr(datasets, data.pop('datasetMethodName'))
    data['trainMethod'] = trainMethod
    data['datasetMethod'] = datasetMethod

    return Classifier(**data)
