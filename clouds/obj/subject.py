"""
Simulate for a certain number of epochs before stopping to compute next generation

each subject runs in a separate process? < only one at a time though, or else
we might mess up timing

    Simulator kills processes if they simulate too long.
"""

import random
import sys
import os
from collections import defaultdict
import multiprocessing
import logging
from pprint import pformat
import tempfile
import shutil
from contextlib import contextmanager

import numpy as np
import camel

from clouds.obj import classifier
from clouds.obj.classifier import Classifier
from clouds.obj import genome
from clouds.util import util
from clouds.util import constants, multiprocess

log = logging.getLogger('SimulationLogger')





class Subject(object):
    _camelName = 'subject.yaml'

    def __init__(self, outputDir, classifier_=None, chunkSize=10, isAlive=True,
                 fitness=0):
        """
        A container for a single classifier.
        """
        self.name = os.path.basename(outputDir)
        self.outputDir = outputDir
        self.classifier = classifier_

        #how many images to train on at once
        self.chunkSize = chunkSize
        self.runtimes = []
        self.errors = []
        self.imagesTrainedOn = 0

        self.successPercentage = None
        self.fitness = fitness

        #set to false if this classifier is rejected (e.g., it has an error
        #which causes it to crash)
        self.isAlive = isAlive

    def __repr__(self):
        return "<Subject {}>".format(self.name)

    def __eq__(self, other):
        if isinstance(other, Subject):
            return self._comparisonKey() == other._comparisonKey()
        return NotImplemented

    def _comparisonKey(self):
        """
        A tuple of attributes that can be used for comparison.
        """
        return (
            self.name,
            self.classifier,
            self.chunkSize
        )

    def save(self):
        """
        Save self to our directory. Overwriting old data.

        Saves to a temporary directory, then overwrites our existing dir.
        """
        d = tempfile.mkdtemp(prefix="saving{}".format(self.name))
        self.dump(dirPath=d)
        shutil.rmtree(self.outputDir, ignore_errors=True)
        shutil.move(d, self.outputDir)

    def dump(self, dirPath=None, overwrite=False):
        """
        Save a representation of self in the given directory.
        """
        if not dirPath:
            dirPath = self.outputDir
        if not overwrite and os.path.isdir(dirPath) and os.listdir(dirPath):
            raise IOError("The directory exists and is not empty: {}".format(dirPath))
        util.mkdir_p(dirPath)

        self.classifier.dump(os.path.join(dirPath, 'classifier'), overwrite)
        with open(os.path.join(dirPath, self._camelName), 'w') as f:
            f.write(serializer.dump(self))

    @classmethod
    def loadFromDir(cls, dirPath):
        """
        Return a subject, loaded from the given directory.
        """
        with open(os.path.join(dirPath, cls._camelName)) as f:
            s = serializer.load(f.read())

            s.classifier = Classifier.loadFromDir(os.path.join(dirPath, 'classifier'))
        return s


    @classmethod
    @contextmanager
    def workon(cls, dirPath):
        """
        A context manager that will return a subject from the given directory, then save the
        subject on exit.
        """
        s = cls.loadFromDir(dirPath)
        try:
            yield s
        finally:
            s.save()

    def train(self, imageDict, commandQ, resultQ, maxEpochs=None):
        """
        Train our classifier by feeding it images and statuses.
        """
        try:
            self.classifier.train(*list(zip(*imageDict.items())), maxEpochs, commandQ, resultQ)
            self.errors.append(self.classifier.error)
        except Exception as e:
            log.exception("Subject {} Died".format(self.name))
            self.isAlive = False
            self.runtimes.append(sys.maxsize)
        else:
            self.imagesTrainedOn += len(imageDict)
            self.runtimes.append(self.classifier.trainTime)

    def evaluateFitness(self, imageDict, tests=100):
        """
        Calculate the performance of our classifier. Test it 'tests' times against a random
        selection of training data.
        """
        #partition by classification type
        statuses = defaultdict(dict)
        for k, v in imageDict.items():
            statuses[v][k] = v

        numTests = min(tests, len(statuses))
        images = [x for st in list(statuses.keys()) for
                  x in random.sample(statuses[st].keys(), numTests)]
        selectedStatuses = [imageDict[k] for k in images]

        correct = [self.classifier.classify(i)[0] == s for i, s in zip(images, selectedStatuses)]

        self.successPercentage = np.mean(correct) * 100
        self.runtimes.append(self.classifier.trainTime)

        #TODO: incorporate runtimes
        self.fitness = self.successPercentage


geneticsRegistry = camel.CamelRegistry()
serializer = camel.Camel((camel.PYTHON_TYPES, classifier.classifierRegistry,
                          geneticsRegistry, constants.healthStatusRegistry))

####################### DUMPERS #######################

@geneticsRegistry.dumper(Subject, 'Subject', 1)
def _dumpSubject(obj):
    return {
        "isAlive": obj.isAlive,
        'chunkSize': obj.chunkSize,
        'outputDir': obj.outputDir,
        'fitness': float(obj.fitness),
    }


####################### LOADERS #######################

@geneticsRegistry.loader('Subject', 1)
def _loadSubject(data, version):
    data.pop('imageDict', None)
    return Subject(**data)


