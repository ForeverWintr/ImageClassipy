"""
'Genes' are objects that encode for certain behaviours
"""
import random
import sys
import abc

import numpy as np
from pybrain.supervised import trainers


SEED = random.randint(0, np.iinfo(np.uint32).max)
random.seed(SEED)
np.random.seed(SEED)

class Gene(object):
    __metaclass__ = abc.ABCMeta

    #the seed that was used to create this object
    _seed = SEED

    @abc.abstractproperty
    def parameter(self):
        pass

#class ImageSize(gene):
    #def __init__(self):
        #self._imageSize = _




class TrainMethod(Gene):
    def __init__(self):
        self._trainer = trainers.BackpropTrainer

    @property
    def trainer(self):
        return self._trainer


class HiddenLayers(Gene):
    def __init__(self, layerlist=None):
        if layerlist is None:
            self._layerlist = self.randomHiddenLayers()
        else:
            self._layerlist = layerlist

    @property
    def layerlist(self):
        return self._layerlist

    @staticmethod
    def randomHiddenLayers(numRange=(0, 100), neuronRange=(1, 10000)):
        count = _normalRandInt(*numRange)
        return [_normalRandInt(*neuronRange) for c in range(count)]


def _normalRandInt(a, b):
    """
    Select an integer between a and b, calculated using a normal
    distribution, centered around a with b at the 1st standard deviation.
    """
    assert a < b
    validRange = b - a
    stdDev = validRange / 2.0

    selection = -1
    while not (0 <= selection <= validRange):
        selection = np.random.normal(0, stdDev)
    return int(round(abs(selection) + a))


