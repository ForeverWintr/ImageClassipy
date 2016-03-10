"""
This is just a simple module that mocks the pybrain train function in a subject to sit and wait to
be interrupted. It's called from the arena tests.
"""

import time
import sys
import json
import contextlib

import sys

#print("IN ABORTABLE MAIN", file=sys.stderr)
#print(sys.path, file=sys.stderr)
#print(sys.executable, file=sys.stderr)
#print(sys.version, file=sys.stderr)

sys.path.append('/Users/tomrutherford/Dropbox/Code/Wing/clouds')
from clouds.obj import arena, subject
from clouds.util import constants


def main(workingDir, numWorkers, jsonImages):
    images = json.loads(jsonImages)
    images = {n: constants.HealthStatus(s) for n, s in images.items()}

    sim = arena.Arena(workingDir, images, maxWorkers=int(numWorkers))
    sim.spawnSubjects(1)

    #mock train to just sleep for a second
    subject.Subject.workon = workonWrapper

    #because we've mocked the train method to never converge, this is an infinite loop. Abort it by
    #sending sigint
    print("Entering infinite loop. Waiting for interrupt signal")
    sim.simulate()

    print("This should never print")
    pass


# dummy functions to replace the classifier's trainUntilConvergence call
class DummyTrainer(object):
    validationErrors = [2, 1]
    def trainUntilConvergence(*args, **kwargs):
        time.sleep(1)
        return [1], [.1]

#Dummy backprop trainer (same name to avoid corrupting the classifier)
def BackpropTrainer(*args, **kwargs):
    return DummyTrainer()


WORKON = subject.Subject.workon
@contextlib.contextmanager
def workonWrapper(dirPath, *args, **kwargs):
    with WORKON(dirPath, *args, **kwargs) as s:
        s.classifier.trainMethod = BackpropTrainer
        yield s


if __name__ == '__main__':
    #print (sys.argv)
    main(*sys.argv[1:])
