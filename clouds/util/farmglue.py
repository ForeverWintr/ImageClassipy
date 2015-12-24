"""
Glue for interfacing with a farm object
"""
import os
import ast
from .constants import HealthStatus

def imagesAndStatuses(outputDir):
    """
    Return a dictionary of images and corresponding human assigned statuses from output.
    """
    controlPath = os.path.join(outputDir, 'controls', 'HealthImageryControl.txt')
    fieldsPath = os.path.join(outputDir, 'fields')

    control = _loadControl(controlPath)

    images = {}
    for field in _dirs(fieldsPath):
        fieldname = os.path.basename(field)
        for seg in _dirs(os.path.join(field, 'HealthSegments')):
            segname = os.path.basename(seg)

            if segname.endswith('_MTL'):
                segname += '.txt'

            if control[fieldname][segname]['humanAssignedStatus'].lower() != 'yes':
                continue

            segStatus = _pickStatus(control[fieldname][segname]['healthStatus'])

            segRGBTiff = os.path.join(seg, 'rgb', 'rgb.tif')
            segRGBPng = os.path.join(seg, 'rgb', 'rgb.png')

            #choose png if it exists, as it'll save us converting later
            if os.path.exists(segRGBPng):
                images[segRGBPng] = segStatus
            else:
                images[segRGBTiff] = segStatus
    return images


def _pickStatus(statusString):
    s = statusString.lower()

    if s == 'cloudy imagery' or statusString == 'CLOUDY':
        return HealthStatus.CLOUDY
    if s == 'valid crop health' or statusString == 'VALID':
        return HealthStatus.GOOD
    if s == 'insufficient image coverage' or statusString == 'INSUFFICIENT_COVERAGE':
        return HealthStatus.INSUFFICIENT_COVERAGE
    if s == 'full bloom canola' or statusString == 'BLOOMING_CANOLA':
        return HealthStatus.CANOLA
    if s == 'image rejected - other' or statusString == 'REJECTED_OTHER':
        return HealthStatus.REJECTED_OTHER
    raise KeyError("Unrecognized status: '{}'".format(statusString))


def _loadControl(controlPath):
    """
    Load and format the control, discarding irrelevant information like the latest date.
    """
    with open(controlPath) as f:
        c = ast.literal_eval(f.read())

    control = {}
    for f, v in c.items():
        subdict = {s.pop('name'): s for s in v[1]}
        control[f] = subdict

    return control


def _dirs(directory):
    """
    Return directories in the given one.
    """
    return (os.path.join(directory, x) for x in next(os.walk(directory))[1])


if __name__ == '__main__':
    path = r'\\ADA\AutoCropHealth'
    extractTo = r'D:\Scratch\CHImages'

    from elvyra.workflows.automation.crophealth import getExtantFarms
    from elvyra.util import fileutil
    import shutil

    dn = os.path.dirname
    farmDirs = getExtantFarms(path)

    for d in farmDirs:
        farmout = os.path.join(d, 'output')
        images = imagesAndStatuses(farmout)

        farmdir = fileutil.createDirectory(os.path.join(extractTo, os.path.basename(d)))
        print("Copying to {} images to {}".format(len(images), farmdir))
        for i, s in images.iteritems():
            fieldname = os.path.basename(dn(dn(dn(dn(i)))))
            segname = os.path.basename(dn(dn(i)))

            fieldir = fileutil.createDirectory(farmdir, fieldname)
            image = os.path.join(fieldir, "{}__{}.tif".format(s.name, segname))
            shutil.copy(i, image)



    print("done")
