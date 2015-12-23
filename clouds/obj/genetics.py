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

import numpy as np
import camel

from clouds.obj import classifier
from clouds.obj.classifier import Classifier
from clouds.obj import genome
from clouds.util import util
from clouds.util import constants, multiprocess

log = logging.getLogger('SimulationLogger')

MAX_TRAIN_S = 60 * 60 * 2


class Simulation(object):

    def __init__(self, workingDir, subjectCount, images={}, logQ=None):
        """
        """
        self.logQ = logQ
        self.subjects = []
        self.images = images
        self.workingDir = workingDir

        #self.loadExistingSubjects(workingDir)
        self.createSubjects(subjectCount)

    def createSubjects(self, subjectCount):
        """
        Create subjects. Load existing subjects from workingDir.
        """
        log.debug("Creating {} Test Subjects".format(subjectCount))
        for i in range(subjectCount):
            name = 'Subject_{}'.format(i)
            subjectDir = os.path.join(self.workingDir, name)

            if os.path.exists(subjectDir):
                s = Subject.loadFromDir(subjectDir)
            else:
                log.debug("Creating new {}".format(name))
                s = Subject(subjectDir, classifier_=self.createClassifier(), imageDict=self.images)

            #still dump even if subject already exists, in case format is out of date
            log.debug('{} spawned. saving...'.format(s))
            s.save()
            self.subjects.append(s)


    def createClassifier(self):
        """
        Create a random classifier.
        """
        hiddenLayers = genome.HiddenLayers()
        trainMethod = genome.TrainMethod()
        imageSize = genome.ImageSize()
        datasetMethod = genome.DatasetMethod()
        outClass = genome.OutClass()

        kwargs = dict(
            possible_statuses=set(self.images.values()),
            imageSize=imageSize.parameter,
            hiddenLayers=hiddenLayers.parameter,
            trainMethod=trainMethod.parameter,
            datasetMethod=datasetMethod.parameter,
            outclass=outClass.parameter,
        )
        log.debug("Creating classifier with:\n{}".format(pformat(kwargs)))

        return Classifier(**kwargs)


    def simulate(self, numWorkers=None):
        """
        Train each subject in increments of `epochs` times, and evaluate. Continue until ?
        """
        assert self.subjects, "Can't simulate without subjects!"
        if not numWorkers:
            numWorkers = self._getWorkerCount(len(self.subjects))

        #If we've only got 1 worker, don't bother with a pool
        if numWorkers <= 1:
            result = [self._runSubject(s.outputDir) for s in self.subjects]
        else:
            multiprocess.mapWithLogging(
                self._runSubject,
                [s.outputDir for s in self.subjects],
                log,
                numWorkers
            )
            #simPool = multiprocessing.Pool(
                #processes=numWorkers,
                #initializer=self.workerInit, initargs=(self.logQ, )
            #)
            print(asfd)

    @staticmethod
    def workerInit(logQ):
        """
        Initializer for worker processes.
        """
        print(log.handlers)

    @staticmethod
    def _runSubject(subjectDir):
        """
        Run a single subject, loaded from the given dir. This method is static, and the subject is
        loaded from a directory in order to work around multiprocessing's inability to pickle non
        static class methods.
        """
        s = Subject.loadFromDir(subjectDir)
        s.train()
        s.save()

    @staticmethod
    def _createSubject():
        raise NotImplemented("TODO")

    @staticmethod
    def _getWorkerCount(jobCount):
        """
        Determine how many workers to use.
        """
        return min(multiprocessing.cpu_count(), jobCount)

    def summarize(self):
        """
        Print a results summary.
        """
        print("Classifier fitness:")
        for c in sorted(self.subjects, key=lambda s: s.fitness):
            print((c.fitness))



class Subject(object):
    _camelName = 'subject.yaml'

    def __init__(self, outputDir, classifier_=None, imageDict={}, chunkSize=10, isAlive=True):
        """
        A container for a single classifier.
        """
        self.name = os.path.basename(outputDir)
        self.outputDir = outputDir
        self.classifier = classifier_
        self.imageDict = imageDict

        #how many images to train on at once
        self.chunkSize = chunkSize
        self.runtimes = []
        self.errors = []
        self.imagesTrainedOn = 0

        self.successPercentage = None
        self.fitness = 0

        #set to false if this classifier is rejected (e.g., it has an error
        #which causes it to crash)
        self.isAlive = isAlive

        #partition by classification type
        self.statuses = defaultdict(dict)
        for k, v in imageDict.items():
            self.statuses[v][k] = v

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


    def run(self):
        #randomize image order
        keyOrder = list(self.imageDict.keys())
        random.shuffle(keyOrder)

        #pass in a chunk of randomly selected images, then evaluate runtime
        for keys in util.grouper(keyOrder, self.chunkSize):
            self.train(keys, [self.imageDict[k] for k in keys])

            #if it takes too long to train this network, it dies.
            if self.runtimes[-1] > MAX_TRAIN_S:
                continue

        #now evaluate fitness after training.
        #choose 5 of each status randomly to evaluate
        evaluateKeys = [x for st in list(self.statuses.keys()) for
                        x in random.sample(self.statuses[st], 5)]
        self.evaluateFitness(evaluateKeys, [self.imageDict[k] for k in evaluateKeys])


    def train(self, maxEpochs=1000):
        """
        Train our classifier by feeding it images and statuses.
        """

        try:
            self.classifier.train(*list(zip(*self.imageDict.items())))
            self.errors.append(self.classifier.error)
        except Exception as e:
            log.exception("Subject {} Died".format(self.name))
            self.isAlive = False
            self.runtimes.append(sys.maxsize)
        else:
            self.imagesTrainedOn += len(self.imageDict)
            self.runtimes.append(self.classifier.trainTime)

    def evaluateFitness(self, images, statuses):
        """
        Calculate the performance of our classifier. Test it against the
        images and statuses given.
        """
        correct = [self.classifier.classify(i)[0] == s for i, s in zip(images, statuses)]

        self.successPercentage = np.mean(correct) * 100
        self.runtimes.append(self.classifier.avgClassifyTime)

        #TODO: incorporate runtimes
        self.fitness = self.successPercentage


geneticsRegistry = camel.CamelRegistry()
serializer = camel.Camel((camel.PYTHON_TYPES, classifier.classifierRegistry,
                          geneticsRegistry, constants.healthStatusRegistry))

####################### DUMPERS #######################

@geneticsRegistry.dumper(Subject, 'Subject', 1)
def _dumpSubject(obj):
    return {
        'imageDict': obj.imageDict,
        "isAlive": obj.isAlive,
        'chunkSize': obj.chunkSize,
        'outputDir': obj.outputDir,
    }


####################### LOADERS #######################

@geneticsRegistry.loader('Subject', 1)
def _loadSubject(data, version):
    return Subject(**data)


