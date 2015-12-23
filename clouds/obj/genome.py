"""
'Genes' are objects that encode for certain behaviours
"""
import random
import sys
import abc

import numpy as np
from pybrain.supervised import trainers
from pybrain import datasets
from pybrain.structure import modules



class Gene(object, metaclass=abc.ABCMeta):
    _seed = None

    @property
    @abc.abstractmethod
    def parameter(self):
        pass


class DatasetMethod(Gene):
    def __init__(self):
        self._dataset = datasets.ClassificationDataSet

    @property
    def parameter(self):
        return self._dataset


class OutClass(Gene):
    def __init__(self):
        self._outClass = modules.SoftmaxLayer

    @property
    def parameter(self):
        return self._outClass


class ImageSize(Gene):
    def __init__(self):
        size = _normalRandInt(5, 400)
        self._imageSize = (size, size)

    @property
    def parameter(self):
        return self._imageSize


class TrainMethod(Gene):
    def __init__(self):
        self._trainer = trainers.BackpropTrainer

    @property
    def parameter(self):
        return self._trainer


class HiddenLayers(Gene):
    def __init__(self, layers=None):
        if layers is None:
            self._layers = self.randomHiddenLayers()
        else:
            self._layers = layers

    @property
    def parameter(self):
        return self._layers

    @staticmethod
    def randomHiddenLayers(numRange=(0, 10), neuronRange=(1, 500)):
        count = _normalRandInt(*numRange)
        return tuple(_normalRandInt(*neuronRange) for c in range(count))


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

def seed(seed=None):
    """
    Seed the random number generators used to generate genes.
    """

    Gene._seed = seed

    if not Gene._seed:
        #seed random with its default method
        random.seed()

        #seed again, keeping the seed for future reference. This has limited utility, because
        #regenerating the genome from a seed will also require generating it in the original
        #sequence, but lets try it anyway.
        Gene._seed = random.randint(0, np.iinfo(np.uint32).max)

    random.seed(Gene._seed)
    np.random.seed(Gene._seed)
seed()
