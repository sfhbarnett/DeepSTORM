import os
import tifffile as tf
from TiffStack import TiffStack
from scipy.ndimage import gaussian_filter
import numpy as np
from random import randint
import re

def natsort(tosort):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(tosort, key=alphanum_key)

# This is a conversion of the data generation script in the original DEEPSTORM repository from matlab to python

tiffpath = 'DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.tif'
csvpath = 'DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.csv'
outdirectory = 'augments/'

camerapixelsize = 160

tiffstack = TiffStack(tiffpath)  # make containiner for tif data

gaussian_sigma = 1  # gaussian sigma for blurring the spike image
pathsize = 26*tiffstack.upsample
numpatches = 500  # number of cropped sections per thunderstorm image
maxexamples = 10_000  # max number of examples


highres_width = tiffstack.width*tiffstack.upsample
highres_height = tiffstack.height*tiffstack.upsample

highres_pixelsize = camerapixelsize/tiffstack.upsample

ntrain = min([tiffstack.nfiles*numpatches, maxexamples])

# Read in molecule positions
localisations = []
with open(csvpath, 'r') as f:
    lines = f.readlines()
    for line in lines[2:]:
        line = line.strip('\n').split(',')
        line = [float(x) for x in line]
        localisations.append(line)

k = 0

XB = np.linspace(0,7,8)
YB = np.linspace(0,7,8)
X,Y = np.meshgrid(XB,YB)

for index in range(1, tiffstack.nfiles):
    print(f'{index}')
    largeimg = tiffstack.getimageupsampled(index)

    points = np.asarray([x for x in localisations if int(x[1]) == index+1])


    #xpos = [int(max(min(round(points[point,2] / highres_pixelsize), highres_width - 1), 0)) for point in
    #                range(len(points))]
    #ypos = [int(max(min(round(points[point,3] / highres_pixelsize), highres_height - 1), 0)) for point in
    #                range(len(points))]

    xpos = np.round(points[:, 2] / highres_pixelsize)
    ypos = np.round(points[:, 3] / highres_pixelsize)

    xpos[xpos < 0] = 0
    xpos[xpos > (highres_width-1)] = highres_width-1
    ypos[ypos < 0] = 0
    ypos[ypos > (highres_height-1)] = highres_height-1

    spike_images = np.zeros((highres_width, highres_height))
    spike_images[ypos, xpos] += 1
    spike_images = gaussian_filter(spike_images, gaussian_sigma)

    if k > ntrain:
        break
    else:
        for i in range(numpatches):
            xcrop = randint(0, (512-208))
            ycrop = randint(0, (512-208))
            patch = largeimg[xcrop:xcrop+208, ycrop:ycrop+208]
            spike = spike_images[xcrop:xcrop+208, ycrop:ycrop+208]
            tf.imwrite(os.path.join(outdirectory, 'img/'+str(k)+'.tif'), patch)
            tf.imwrite(os.path.join(outdirectory, 'spikes/'+str(k)+'.tif'), spike)

            # Plot the two images on top of one another to make sure spikes are in right location
            # scale1 = patches[:, :, k]/np.max(patches[:, :, k])*255
            # scale2 = heatmaps[:, :, k]/np.max(heatmaps[:, :, k])*255
            # blend = Image.blend(Image.fromarray(scale1).convert('RGBA'),Image.fromarray(scale2).convert('RGBA'),0.5)
            # plt.imshow(blend)
            # plt.show()
            # plt.draw()
            # plt.pause(0.1)
            k += 1


path = '/Users/sbarnett/PycharmProjects/DeepSTORM/DeepSTORM/augments/spikes'


ims = os.listdir(path)
ims.remove('.DS_Store')

ims = natsort(ims)

stack = np.zeros((2000,208,208))

for index in range(2000):
    print(index)
    stack[index,:,:] = tf.imread(os.path.join(path,ims[index]))

tf.imwrite('spikes.tif',stack)