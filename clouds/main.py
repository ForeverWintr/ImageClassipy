import sys

import numpy as np

from clouds.obj import genetics
from clouds.util import farmglue

#TODO:
# str and repr for classifiers
# Pickle a simulation/classifiers
# implement logging
# Subjects are directories. Worker processes
#

FARMDIR = r'/Users/tomrutherford/Documents/Hervalense'
def main(argv):

    np.seterr('raise')
    farmDir = FARMDIR
    images = farmglue.imagesAndStatuses(farmDir)

    sim = genetics.Simulation(1, images)

    print sim.subjects
    print sim.subjects[0].classifier
    print("Simulating.")
    sim.simulate()

    sim.summarize()
    print "Done"
    pass


if __name__ == '__main__':
    main(sys.argv)
