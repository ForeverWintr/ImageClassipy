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

import numpy as np

from clouds.obj import classifier
from clouds.obj import genome
from clouds.util import util

log = logging.getLogger('SimulationLogger')

MAX_TRAIN_S = 60 * 60 * 2


class Simulation(object):

    def __init__(self, workingDir, subjectCount, images={}):
        """
        """
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

            s = Subject(subjectDir, classifier=self.createClassifier())
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

        return classifier.Classifier(
            possible_statuses=set(self.images.values()),
            imageSize=imageSize.parameter,
            hiddenLayers=hiddenLayers.parameter,
            trainMethod=trainMethod.parameter,
            datasetMethod=datasetMethod.parameter,
            outclass=outClass.parameter,
        )


    def simulate(self, epochs=50):
        """
        Train each subject in increments of `epochs` times, and evaluate. Continue until ?
        """

        print(asdf)


    def summarize(self):
        """
        Print a results summary.
        """
        print("Classifier fitness:")
        for c in sorted(self.subjects, key=lambda s: s.fitness):
            print((c.fitness))



class Subject(object):

    def __init__(self, outputDir, classifier=None, imageDict={}, chunkSize=10):
        """
        A container for a single classifier.
        """
        self.classifier = classifier or aasdf
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
        self.isAlive = True

        #partition by classification type
        self.statuses = defaultdict(dict)
        for k, v in imageDict.items():
            self.statuses[v][k] = v

    def newClassifier(self):
        """
        Randomize a new classifier.
        """
        imgSize = random.randint(5, 50)

        hiddenLayers = genome.HiddenLayers()
        s = classifier.Classifier(
            imageSize=(imgSize, imgSize),
            hiddenLayers=hiddenLayers.parameter
        )
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


    def train(self, images, statuses):
        """
        Train our classifier by feeding it images and statuses.
        """

        try:
            self.classifier.train(images, statuses)
            self.errors.append(self.classifier.error)
        except Exception:
            self.isAlive = False
            self.runtimes.append(sys.maxsize)
        else:
            self.imagesTrainedOn += len(images)
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




