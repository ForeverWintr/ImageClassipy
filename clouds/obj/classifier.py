"""
A classifier is an entity that evaluates images for status (cloudy, canola, etc.)
"""
import os
import time
from operator import mul, itemgetter
import logging
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


log = logging.getLogger('SimulationLogger')


class Classifier(object):
    _NET_NAME = 'net.xml'
    _camelName = 'classifier.yaml'

    def __init__(self, possibleStatuses, imageSize=(128, 128), hiddenLayers=None,
                 trainMethod=trainers.BackpropTrainer,
                 datasetMethod=ClassificationDataSet,
                 outclass=SoftmaxLayer, convergenceThreshold=10,
                 imageMode='L'):
        hiddenLayers = hiddenLayers or tuple()

        self.possibleStatuses = possibleStatuses
        self.netSpec = (mul(*imageSize), ) + hiddenLayers + (len(possibleStatuses), )
        self.imageSize = tuple(float(x) for x in imageSize)
        self.trainMethod = trainMethod
        self.datasetMethod = datasetMethod
        self.net = buildNetwork(*self.netSpec, outclass=outclass)
        self.convergenceThreshold = convergenceThreshold
        self.imageMode = imageMode

        #statistics
        self.avgCertainty = None
        self.trainTime = None
        self.error = None
        self.epochsTrained = None

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

    def train(self, images, statuses, reportInterval=None, commandQ=None, resultQ=None):
        numStatuses = len(self.possibleStatuses)
        ds = self.datasetMethod(mul(*self.imageSize), 1, numStatuses)

        log.debug("{}: Getting images".format(self))
        [ds.addSample(self._loadToArray(i), e.value) for i, e in zip(images, statuses)]
        log.debug("{} done".format(self))

        #convert to one output per class. Apparently this is a better format?
        # http://pybrain.org/docs/tutorial/fnn.html
        ds._convertToOneOfMany()

        trainer = self.trainMethod(self.net, dataset=ds)

        start = time.clock()
        while True:
            trainErrors, validationErrors = trainer.trainUntilConvergence(
                convergence_threshold=self.convergenceThreshold, maxEpochs=reportInterval)



            if self._trainingBreakCondition(trainer.trainingErrors, reportInterval):
                break

        trainTime = time.clock() - start

        iterations = len(trainErrors) + len(validationErrors)
        log.debug("Training took {} iterations".format(iterations))
        if trainErrors:
            log.debug("Errors: {}, {}".format(trainErrors[-1], validationErrors[-1]))
        else:
            log.debug("Training unsuccesfull. Trainerrors is empty.")

        self.trainTime = float(trainTime) / iterations
        self.error = validationErrors[-1]
        return trainErrors, validationErrors

    def _trainingBreakCondition(self, errorList, maxEpochs):
        """
        Return true if we consider training finished. Currently this just checks to see if the
        number of times we trained was less than the number of epochs we requested.
        """
        return not maxEpochs or (len(errorList) - maxEpochs) < 2

    def classify(self, imagePath):
        """
        Return a HealthStatus enum and a measure of our certainty.
        """
        start = time.clock()
        result = self.net.activate(self._loadToArray(imagePath))

        #log.debug("Result is:", result)

        guess = HealthStatus._value2member_map_[np.argmax(result)]

        return guess, result


    def _loadToArray(self, imagePath):
        """
        Creates input array. Applies scale factor to each image.
        """
        try:
            image = PIL.Image.open(imagePath)
        except IOError as e:
            raise
            #print("Trying to open by converting to png")
            png = os.path.splitext(imagePath)[0] + '.png'
            wand.image.Image(filename=imagePath).convert('PNG').save(filename=png)
            image = PIL.Image.open(png)

        #resize
        scaleFactor = np.divide(self.imageSize, image.size)
        newSize = tuple(round(x * s) for x, s in zip(image.size, scaleFactor))
        image.thumbnail(newSize)

        image = image.convert(self.imageMode)

        # rescale the images in the input array as linear (the network does't know about shape
        # anyway)
        imageArray = np.array(image)

        #if we're using rgb mode, convert pixels to 32 bit hex colours
        if self.imageMode.upper() == 'RGB':
            imageArray = util.rgbToHex(imageArray)

        newSize = mul(*imageArray.shape)
        return imageArray.reshape(newSize)


    def dump(self, dirPath, overwrite):
        """
        Save a representation of this classifier and it's network at the given path.
        """
        if not overwrite and os.path.isdir(dirPath) and os.listdir(dirPath):
            raise IOError("The directory exists and is not empty: {}".format(dirPath))
        util.mkdir_p(dirPath)

        #save network
        NetworkWriter.writeToFile(self.net, os.path.join(dirPath, self._NET_NAME))

        #save classifier
        with open(os.path.join(dirPath, self._camelName), 'w') as f:
            f.write(serializer.dump(self))


    @classmethod
    def loadFromDir(cls, dirPath):
        """
        Return a classifier, loaded from the given directory.
        """
        with open(os.path.join(dirPath, cls._camelName)) as f:
            c = serializer.load(f.read())

        log.debug("{}: loading network".format(c))
        c.net = NetworkReader.readFrom(os.path.join(dirPath, cls._NET_NAME))
        return c


classifierRegistry = camel.CamelRegistry()
serializer = camel.Camel((camel.PYTHON_TYPES, classifierRegistry, healthStatusRegistry))

####################### DUMPERS #######################

@classifierRegistry.dumper(Classifier, 'Classifier', 2)
def _dumpClassifier(obj):
    return {
        'possibleStatuses': obj.possibleStatuses,
        "imageSize": obj.imageSize,
        'hiddenLayers': obj.netSpec[1:-1],
        'trainMethodName': str(obj.trainMethod.__name__),
        'datasetMethodName': str(obj.datasetMethod.__name__),
        'convergenceThreshold': obj.convergenceThreshold,
        'imageMode': obj.imageMode,
    }


####################### LOADERS #######################

@classifierRegistry.loader('Classifier', all)
def _loadClassifier(data, version):
    if version == 1:
        data['possibleStatuses'] = data.pop('possible_statuses')

    trainMethod = getattr(trainers.backprop, data.pop('trainMethodName'))
    datasetMethod = getattr(datasets, data.pop('datasetMethodName'))
    data['trainMethod'] = trainMethod
    data['datasetMethod'] = datasetMethod

    return Classifier(**data)
