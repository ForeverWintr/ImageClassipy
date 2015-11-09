"""
Glue for interfacing with a farm object
"""
import os
import ast
from constants import HealthStatus

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

    if s == 'cloudy imagery':
        return HealthStatus.CLOUDY
    if s == 'valid crop health':
        return HealthStatus.GOOD
    if s == 'insufficient image coverage':
        return HealthStatus.INSUFFICIENT_COVERAGE
    if s == 'full bloom canola':
        return HealthStatus.CANOLA
    if s == 'image rejected - other':
        return HealthStatus.REJECTED_OTHER
    raise KeyError("Unrecognized status: '{}'".format(statusString))


def _loadControl(controlPath):
    """
    Load and format the control, discarding irrelevant information like the latest date.
    """
    with open(controlPath) as f:
        c = ast.literal_eval(f.read())

    control = {}
    for f, v in c.iteritems():
        subdict = {s.pop('name'): s for s in v[1]}
        control[f] = subdict

    return control


def _dirs(directory):
    """
    Return directories in the given one.
    """
    return (os.path.join(directory, x) for x in next(os.walk(directory))[1])


if __name__ == '__main__':
    path = '/Users/tomrutherford/Documents/Hervalense'
    images = imagesAndStatuses(path)


    print "done"
