import sys
import logging
from logging import handlers
import datetime
import os
import multiprocessing
import sys
import glob
import re

import numpy as np

from clouds.obj import genetics
from clouds.util import farmglue
from clouds.util import constants
from clouds import util

#TODO:
# implement logging
# Subjects are directories. Worker processes
# Balance input in training

FARMDIR = r'/Users/tomrutherford/Documents/Hervalense'
IMAGEDIR = r'/Users/tomrutherford/Documents/Data/CHImages'
WORKINGDIR = '/Users/tomrutherford/Documents/Data/clouds'

def main(argv):

    farmDir = FARMDIR

    #logging setup
    log = loggingSetup()

    #np.seterrcall(log)
    np.seterr('raise', under='warn')

    #convert to png
    tifToPng(IMAGEDIR)

    log.debug("Get images")
    images = farmglue.imagesAndStatuses(farmDir)
    images.update(imagesFrom(IMAGEDIR, extensions=('png', )))

    sim = genetics.Arena(WORKINGDIR, images)

    #manual subject creation
    #sim.createSubject(
        #'Manual_0',
        #possibleStatuses=set(images.values()),
        #imageSize=(128, 128),
        #hiddenLayers=None,
    #)

    sim.spawnSubjects(1, ['Manual_0'])
    log.debug("Simulating.")

    sim.simulate()

    sim.summarize()
    log.debug("Done")


def padImages(images):
    """
    even out the number of images by duplicating minorities.
    """
    from collections import Counter
    #can't do this because images is a dict...


def tifToPng(dir_):
    """
    Convert all tiffs in the directory to pngs. Recursive.
    """
    for t in glob.iglob(os.path.join(dir_, '**', '.*tif'), recursive=True):
        png = os.path.splitext(t)[0] + '.png'
        wand.image.Image(filename=t).convert('PNG').save(filename=png)
        #image = PIL.Image.open(png)


def imagesFrom(dir_, extensions=('tif', 'tiff', 'png')):
    images = {}
    for p in util.flatten(glob.glob(os.path.join(dir_, '**', '*.{}'.format(e)), recursive=True)
              for e in extensions):
        status = getStatusFromName(os.path.basename(p))
        images[p] = status

    return images


def getStatusFromName(name):
    matcher = re.compile('^({})_'.format('|'.join(constants.HealthStatus._member_names_)))
    match = matcher.match(name)
    if not match:
        return ''

    return constants.HealthStatus._member_map_[match.groups()[0]]


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
