"""
Simulate for a certain number of epochs before stopping to compute next generation

each subject runs in a separate process? < only one at a time though, or else
we might mess up timing

    Simulator kills processes if they simulate too long.
"""

import random
import sys
from collections import defaultdict
import multiprocessing

import numpy as np

from clouds.obj import classifier
from clouds.util import util

MAX_TRAIN_S = 60 * 60 * 2


class Simulation(object):

    def __init__(self, subjectCount, images={}):
        self.subjects = []
        self.images = images

        self.setUp(subjectCount)

    def setUp(self, subjectCount):
        """
        Create subjects
        """
        print("Creating {} Test Subjects".format(subjectCount))
        for i in range(subjectCount):
            imgSize = random.randint(5, 50)

            #netspec is all layers after inputs, including output layer, which has to be 1
            hiddenLayers = random.randint(0, 1)
            netspec = [random.randint(1, 100) for x in range(hiddenLayers)] + [1]
            s = classifier.Classifier(
                imageSize=(imgSize, imgSize),
                netSpec=netspec,
            )

            ### TESTING XXXX
            #s = classifier.Classifier(
                #imageSize=(20, 20),
                #netSpec=[1],
            #)

            self.subjects.append(Subject(s, self.images))


    def simulate(self):
        """
        Run one generation.
        Train and evaluate each subject
        """
        for s in self.subjects:
            s.start()
            s.join()


    def summarize(self):
        """
        Print a results summary.
        """
        print("Classifier fitness:")
        for c in sorted(self.subjects, key=lambda s: s.fitness):
            print(c.fitness)



class Subject(multiprocessing.Process):

    def __init__(self, classifier, imageDict={}, chunkSize=10):
        """
        A container for a single classifier.
        """
        multiprocessing.Process.__init__(self)
        self.classifier = classifier
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
        for k, v in imageDict.iteritems():
            self.statuses[v][k] = v


    def run(self):
        #randomize image order
        keyOrder = self.imageDict.keys()
        random.shuffle(keyOrder)

        #pass in a chunk of randomly selected images, then evaluate runtime
        for keys in util.grouper(keyOrder, self.chunkSize):
            self.train(keys, [self.imageDict[k] for k in keys])

            #if it takes too long to train this network, it dies.
            if self.runtimes[-1] > MAX_TRAIN_S:
                continue

        #now evaluate fitness after training.
        #choose 5 of each status randomly to evaluate
        evaluateKeys = [x for st in self.statuses.keys() for
                        x in random.sample(self.statuses[st], 5)]
        s.evaluateFitness(evaluateKeys, [self.images[k] for k in evaluateKeys])


    def train(self, images, statuses):
        """
        Train our classifier by feeding it images and statuses.
        """

        try:
            self.classifier.train(images, statuses)
            self.errors.append(self.classifier.error)
        except Exception:
            self.isAlive = False
            self.runtimes.append(sys.maxint)
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




