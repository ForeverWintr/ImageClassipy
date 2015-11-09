import sys

from clouds.obj import genetics
from clouds.util import farmglue

#TODO:
# str and repr for classifiers
# Pickle a simulation/classifiers
# implement logging

FARMDIR = r'/Users/tomrutherford/Documents/Hervalense'
def main(argv):

    farmDir = FARMDIR
    images = farmglue.imagesAndStatuses(farmDir)

    sim = genetics.Simulation(1000, images)

    sim.simulate()

    pass


if __name__ == '__main__':
    main(sys.argv)
