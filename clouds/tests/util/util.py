"""
Test utils
"""
import tempfile

import PIL
import numpy as np

from clouds.util.constants import HealthStatus

def createXors(tgt):
    #create test xor images
    xorIn = [
        ((255, 255, 255, 255), HealthStatus.GOOD),
        ((255, 255, 0, 0), HealthStatus.CLOUDY),
        ((0, 0, 0, 0), HealthStatus.GOOD),
        ((0, 0, 255, 255), HealthStatus.CLOUDY),
    ]
    xorImages = []
    for ar, expected in xorIn:
        npar = np.array(ar, dtype=np.uint8).reshape(2, 2)

        image = PIL.Image.fromarray(npar)

        #pybrain needs a lot of test input. We'll make 20 of each image
        for i in range(20):
            path = tempfile.mktemp(suffix=".png", prefix='xor_', dir=tgt)
            image.save(path)
            xorImages.append((path, expected))
    return xorImages


class MockStream(object):
    def __init__(self, inputQueue):
        """
        A class used as a replacement for stream objects. As data are recieved on the inputQueue,
        make them available to `readline`.
        """
        self.q = inputQueue

    def read(self):
        return [l for l in self.readline()]

    def readline(self):
        """
        Block until an item appears in the queue.
        """
        return self.q.get()

    def close(self):
        pass
