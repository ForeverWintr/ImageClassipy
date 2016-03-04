"""
Arena. A container that holds subjects and methods for training them.
"""
import multiprocessing
import os
import logging
import signal
import sys

from clouds.util import multiprocess
from clouds.util import SendStopOnInterrupt
from clouds.util import debuggingEnabled
from clouds.obj.subject import Subject
from clouds.obj.classifier import Classifier
from clouds.obj import genome
from clouds.util.constants import Command

log = logging.getLogger('SimulationLogger')

DEBUG = debuggingEnabled()
#DEBUG = False

class Arena(object):

    def __init__(self, workingDir, images={}, maxWorkers=multiprocessing.cpu_count()):
        self.subjects = []
        self.images = images
        self.workingDir = workingDir
        self.maxWorkers = maxWorkers

        #set up command and result queues
        self.commandQ = multiprocessing.Queue()
        self.reportQ = multiprocessing.Queue()

    def spawnSubjects(self, subjectCount=None, subjectNames=[]):
        """
        Load existing subjects from workingDir. Create new subjects. Optionally, specify a number
        of subjects, and/or a list of subject names. If subjectCount is specified, we'll load that
        many subjects, starting with those named in subjectNames.
        """
        if subjectCount is None:
            subjectCount = len(subjectNames)

        log.debug("Spawning {} Test Subjects".format(subjectCount))
        dirs = []
        nameIter = iter(subjectNames)
        for i in range(subjectCount):
            name = next(nameIter, 'Subject_{}'.format(i-len(subjectNames)))
            subjectDir = os.path.join(self.workingDir, name)

            dirs.append(subjectDir)

        if dirs:
            #assigning subjects like this will only work so long as they're picklable
            with multiprocess.mapWithLogging(self._loadSubject, dirs, log,
                                             self.images,
                                             workerCount=self._getWorkerCount(len(dirs))) as r:
                self.subjects = r.get()

    @staticmethod
    def randomClassifier(possibleStatuses):
        """
        Create a random classifier.
        """
        #child processes inherit the random state of their parents. Re-seed here.
        genome.seed()

        hiddenLayers = genome.HiddenLayers()
        trainMethod = genome.TrainMethod()
        imageSize = genome.ImageSize()
        datasetMethod = genome.DatasetMethod()
        outClass = genome.OutClass()
        imageMode = genome.ImageMode()

        kwargs = dict(
            possibleStatuses=possibleStatuses,
            imageSize=imageSize.parameter,
            hiddenLayers=hiddenLayers.parameter,
            trainMethod=trainMethod.parameter,
            datasetMethod=datasetMethod.parameter,
            outclass=outClass.parameter,
            imageMode=imageMode.parameter,
        )
        log.debug("Creating classifier with:\n{}".format(pformat(kwargs)))

        return Classifier(**kwargs)


    def simulate(self, numWorkers=None, reportEpochs=None, maxEpochs=10000):
        """
        Train each subject in increments of `epochs` times, and evaluate. Continue until
        convergence, death, or maxEpochs is reached.

        Set reportEpochs to report training progress after training for `reportEpochs` epochs.
        """
        assert self.subjects, "Can't simulate without subjects!"
        if not numWorkers:
            numWorkers = self._getWorkerCount(len(self.subjects))

        #If we've only got 1 worker, don't bother with a pool
        with SendStopOnInterrupt(self.commandQ, Command.STOP) as h:
            if DEBUG and numWorkers == 1:
                result = [self._runSubject(s.outputDir, self.images, reportEpochs, self.commandQ,
                                           self.reportQ) for s in self.subjects]
            else:
                with multiprocess.mapWithLogging(self._runSubject,
                                                 [s.outputDir for s in self.subjects], log,
                                                 self.images, reportEpochs, self.commandQ,
                                                 self.reportQ, workerCount=numWorkers) as r:
                    result = r.get()

            if h.interrupted:
                log.warning("Attempted graceful shutdown.")
                sys.exit(0)

        #update our local objects' fitness
        for s, r in zip(self.subjects, result):
            s.fitness = r

    def createSubject(self, name, **kwargs):
        """
        Create a subject using the given arguments. This allows for manual subject injection into
        the arena. See classifier definition for a list of possible kwargs. Use spawnSubjects to
        create random subjects.
        """
        subjectDir = os.path.join(self.workingDir, name)

        if 'possibleStatuses' not in kwargs:
            kwargs['possibleStatuses'] = set(self.images.values())

        c = Classifier(**kwargs)
        s = Subject(
            subjectDir,
            classifier_=c,
        )
        s.save()
        self.subjects.append(s)
        return s


    @staticmethod
    def _runSubject(subjectDir, imageDict, reportEpochs, commandQ, resultQ):
        """
        Run a single subject, loaded from the given dir. This method is static, and the subject is
        loaded from a directory in order to work around multiprocessing's inability to pickle non
        static class methods.
        """
        with Subject.workon(subjectDir) as s:
            isinstance(s, Subject)
            log.info('{} Loaded. Training'.format(s))
            s.train(imageDict, reportEpochs, commandQ, resultQ)
            log.info('{} Training complete'.format(s))
            s.evaluateFitness(imageDict)
            log.info('{} Fitness is {}'.format(s, s.fitness))

        return s.fitness

    @staticmethod
    def _loadSubject(subjectDir, imageDict):
        """
        Initialize a subject at `subjectDir`. Either creating, or load and save if the subject
        exists.
        """
        name = os.path.basename(subjectDir)
        if os.path.exists(subjectDir):
            s = Subject.loadFromDir(subjectDir)
        else:
            log.debug("Creating new {}".format(name))
            s = Subject(
                subjectDir,
                classifier_=Arena.randomClassifier(set(imageDict.values())),
            )

        #still dump even if subject already exists, in case format is out of date
        log.debug('{} spawned. saving...'.format(s))
        s.save()
        return s

    def _getWorkerCount(self, jobCount):
        """
        Determine how many workers to use.
        """
        return min(self.maxWorkers, jobCount)

    def summarize(self):
        """
        Print a results summary.
        """
        print("Classifier fitness:")
        for c in sorted(self.subjects, key=lambda s: s.fitness):
            print((c.fitness))


