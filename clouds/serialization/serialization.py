"""
Serialization stuff goes here
"""

import camel

from clouds.obj.classifier import Classifier

registry = camel.CamelRegistry()

####################### DUMPERS #######################

@registry.dumper(Classifier, 'Classifier', 1)
def _dumpClassifier(obj):
    return {
        #u"imageSize": obj.imageSize
    }


####################### LOADERS #######################

@registry.loader('Classifier', 1)
def _loadClassifier(data, version):
    return Classifier()
