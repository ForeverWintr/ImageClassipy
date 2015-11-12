import sys

from clouds.obj import genetics
from clouds.util import farmglue

#TODO:
# str and repr for classifiers
# Pickle a simulation/classifiers
# implement logging
# subjects are processes
# Test the train process

FARMDIR = r'/Users/tomrutherford/Documents/Hervalense'
def main(argv):

    farmDir = FARMDIR
    images = farmglue.imagesAndStatuses(farmDir)

    sim = genetics.Simulation(1, images)

    print("Simulating.")
    sim.simulate()

    sim.summarize()
    print "Done"
    pass


if __name__ == '__main__':
    main(sys.argv)
