import numpy as np
import scipy
from scipy import misc
import neurolab
import PIL
import wand.image
import os

RGB = r'/Users/tomrutherford/Dropbox/Code/Wing/clouds/healthSegments/2015-06-16_RE_112814/rgb/rgb.png'

# scale factor?
scaleFactor = 0.32
# sliding window?
# different network types?
#     newff takes a list of activation functions per layer
# random samples?
# hidden layer size
# training functions

def main():
    #image = misc.imread(RGB)
    image = _loadToArray(RGB)
    #image = wand.image.Image(filename=RGB)

    #try all inputs, first.
    inputSpec = [[0, 255] for x in range(image.size)]
    netSpec = [1]
    net = neurolab.net.newff(inputSpec, netSpec)

    #train
    shapedImage = image.reshape(1, image.size)
    shapedTarget = np.zeros([1, 1])

    print "Training"
    net.train(shapedImage, shapedTarget, epochs=1, show=1)

    print "Done"
    pass


def _loadToArray(imagePath):
    try:
        image = PIL.Image.open(imagePath)
    except IOError as e:
        print("Trying to open by converting to png")
        png = os.path.splitext(imagePath)[0] + '.png'
        wand.image.Image(filename=imagePath).convert('PNG').save(filename=png)
        image = PIL.Image.open(png)

    #resize
    newSize = tuple(x * scaleFactor for x in image.size)
    image.thumbnail(newSize)

    #greyscale
    image = image.convert('F')

    return np.array(image)




if __name__ == '__main__':
    main()
