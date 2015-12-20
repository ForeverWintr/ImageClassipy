import sys
import logging

# Having these lines here defies PEP8, but we need to get the logger before
# other packages import it
logFormat = '%(levelname)s %(asctime)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=logFormat)
log = logging.getLogger('SimulationLogger')

import datetime
import os

import numpy as np

from clouds.obj import genetics
from clouds.util import farmglue

#TODO:
# str and repr for classifiers
# Pickle a simulation/classifiers
# implement logging
# Subjects are directories. Worker processes
# Balance input in training

FARMDIR = r'/Users/tomrutherford/Documents/Hervalense'
WORKINGDIR = '/Users/tomrutherford/Documents/Data/clouds'

def main(argv):

    np.seterr('raise')
    farmDir = FARMDIR


    #logging setup
    logFile = os.path.join(
        WORKINGDIR,
        'SimulationLog_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')))
    fileHandler = logging.FileHandler(logFile)
    fileHandler.setFormatter(logging.Formatter(logFormat))
    log.addHandler(fileHandler)

    log.debug("Get images")
    images = farmglue.imagesAndStatuses(farmDir)

    sim = genetics.Simulation(WORKINGDIR, 100, images)
    print(sim.subjects)
    print(sim.subjects[0].classifier)
    log.debug("Simulating.")
    sim.simulate()

    sim.summarize()
    log.debug("Done")
    pass


if __name__ == '__main__':
    main(sys.argv)
