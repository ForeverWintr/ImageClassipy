import sys
import logging
from logging import handlers
import datetime
import os
import multiprocessing
import sys

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
    log = loggingSetup()

    log.debug("Get images")
    images = farmglue.imagesAndStatuses(farmDir)

    sim = genetics.Simulation(WORKINGDIR, 2, images)
    log.debug("Simulating.")

    sim.simulate()

    sim.summarize()
    log.debug("Done")
    pass


def loggingSetup():
    log = logging.getLogger('SimulationLogger')
    log.setLevel(logging.DEBUG)

    #this is the top level logger. Don't propagate up to root
    log.propagate = False

    fmt = ('%(levelname)s %(asctime)s %(processName)s (%(process)d) '
           '[%(funcName)s (%(filename)s:%(lineno)s)]: %(message)s')
    fileFormatter = logging.Formatter(fmt)

    fmt = ('%(levelname)s %(processName)s (%(process)d): %(message)s')
    streamFormatter = logging.Formatter(fmt)

    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(streamFormatter)
    streamHandler.setLevel(logging.DEBUG)

    logFile = os.path.join(
        WORKINGDIR,
        'SimulationLog_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')))
    fileHandler = logging.FileHandler(logFile)
    fileHandler.setFormatter(fileFormatter)
    fileHandler.setLevel(logging.INFO)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)
    return log


if __name__ == '__main__':
    main(sys.argv)
    logging.shutdown()
