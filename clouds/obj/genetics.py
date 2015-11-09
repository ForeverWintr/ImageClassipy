"""
Simulate for a certain number of epochs before stopping to compute next generation

each subject runs in a separate process? < only one at a time though, or else
we might mess up timing

    Simulator kills processes if they simulate too long.
"""

import random
import sys
#import multiprocessing

from clouds.obj import classifier
from clouds.util import util

MAX_TRAIN_S = 60 * 60 * 2


class Simulation(object):

    def __init__(self, subjectCount, images=[]):
        self.images = images

        print("Creating {} Test Subjects".format(subjectCount))
        self.subjects = []
        for i in range(subjectCount):
            imgSize = random.randint(5, 200)

            #netspec is all layers after inputs, including output layer, which has to be 1
            hiddenLayers = random.randint(0, 10)
            netspec = [random.randint(1, 100) for x in range(hiddenLayers)] + [1]
            s = classifier.Classifier(
                imageSize=(imgSize, imgSize),
                netSpec=netspec,
            )

            self.subjects.append(Subject(s))

    def simulate(self, chunk_size=10):
        """
        Run one generation.
        Train and evaluate each subject
        """
        #randomize image order
        random.shuffle(self.images)

        for s in self.subjects:
            #pass in a chunk of randomly selected images, then evaluate runtime
            for images in util.grouper(self.images, chunk_size):
                s.train(images)

                #if it takes too long to train this network, it dies.
                if s.runtimes[-1] > MAX_TRAIN_S:
                    continue

            #now evaluate fitness after training.

            pass




class Subject(object):

    def __init__(self, classifier):
        """
        A container for a single classifier.
        """
        self.classifier = classifier
        self.runtimes = []
        self.imagesTrainedOn = 0

        self.fitness = 0

        #set to false if this classifier is rejected (e.g., it has an error
        #which causes it to crash)
        self.isAlive = True

    def train(self, images, statuses):
        """
        Train our classifier by feeding it images and statuses.
        """
        self.classifier.train(images, statuses)
        self.imagesTrainedOn += len(images)

        try:
            self.runtimes.append(self.classifier.trainTime)
        except Exception:
            self.isAlive = False
            self.runtimes.append[sys.maxint]

    def evaluateFitness(self, images, statuses):
        """
        Calculate the performance of our classifier.
        """

