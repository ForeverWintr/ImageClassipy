import sys
import logging
from logging import handlers

# Having these lines here defies PEP8, but we need to get the logger before
# other packages import it
logFormat = '%(levelname)s %(asctime)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=logFormat)
log = logging.getLogger('SimulationLogger')

import datetime
import os
import multiprocessing

import numpy as np

from clouds.obj import genetics
from clouds.util import farmglue

#TODO:
# implement logging
# Subjects are directories. Worker processes
# Balance input in training

FARMDIR = r'/Users/tomrutherford/Documents/Hervalense'
WORKINGDIR = '/Users/tomrutherford/Documents/Data/clouds'

def main(argv):

    np.seterr('raise')
    farmDir = FARMDIR


    #logging setup
    qListener, logQ = loggingSetup()

    log.debug("Get images")
    images = farmglue.imagesAndStatuses(farmDir)

    qListener.start()
    sim = genetics.Simulation(WORKINGDIR, 2, images, logQ)
    log.debug("Simulating.")

    sim.simulate()
    qListener.stop()

    sim.summarize()
    log.debug("Done")
    pass


def loggingSetup():
    logQ = multiprocessing.Queue()

    logFile = os.path.join(
        WORKINGDIR,
        'SimulationLog_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')))
    fileHandler = logging.FileHandler(logFile)
    fileHandler.setFormatter(logging.Formatter(logFormat))
    log.addHandler(fileHandler)

    qListener = handlers.QueueListener(logQ, *log.handlers)
    return qListener, logQ


if __name__ == '__main__':
    main(sys.argv)
