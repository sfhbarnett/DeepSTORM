import os

import tifffile as tf
from TiffStack import TiffStack
from scipy.ndimage import gaussian_filter
import numpy as np
from PIL import Image
from random import randint
import matplotlib.pyplot as plt

tiffpath = 'DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.tif'
csvpath = 'DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.csv'
outdirectory = 'augments/'

upsamplingfactor = 8

camerapixelsize = 160

gaussian_sigma = 1
pathsize = 26*upsamplingfactor
numpatches = 500
maxexamples = 10_000
tiffstack = TiffStack(tiffpath)

highres_width = tiffstack.width*upsamplingfactor
highres_height = tiffstack.height*upsamplingfactor

highres_patchsize = camerapixelsize/upsamplingfactor

ntrain = min([tiffstack.nfiles*numpatches, maxexamples])

patches = np.zeros((pathsize,pathsize,ntrain))
heatmaps = np.zeros((pathsize,pathsize,ntrain))
spikes = np.zeros((pathsize,pathsize,ntrain))

localisations = []
with open(csvpath,'r') as f:
    lines = f.readlines()
    for line in lines[2:]:
        line = line.strip('\n').split(',')
        line = [float(x) for x in line]
        localisations.append(line)

k = 0

for index in range(1, tiffstack.nfiles):
    img = tiffstack.getimage(index)
    newsize = (tiffstack.width*upsamplingfactor,tiffstack.height*upsamplingfactor)
    largeimg = np.array(Image.fromarray(img).resize(newsize, Image.Resampling.NEAREST))

    points = np.asarray([x for x in localisations if int(x[1]) == index+1])
    xpos = np.round(points[:, 2] / highres_patchsize)
    ypos = np.round(points[:, 3] / highres_patchsize)

    spike_images = np.zeros((highres_width+1, highres_height+1))
    spike_images[ypos.astype(int),xpos.astype(int)] = 1
    spike_images = gaussian_filter(spike_images,gaussian_sigma)
    #plt.imshow(spike_images)
    # plt.figure()
    # plt.imshow(img)
    #plt.show()

    if k > ntrain:
        break
    else:
        for i in range(numpatches):
            xcrop = randint(0, 512-208)
            ycrop = randint(0, 512-208)
            patch = largeimg[xcrop:xcrop+208,ycrop:ycrop+208]
            spike = spike_images[xcrop:xcrop+208,ycrop:ycrop+208]
            tf.imwrite(os.path.join(outdirectory, 'img/'+str(k)+'.tif'),patch)
            tf.imwrite(os.path.join(outdirectory, 'spikes/'+str(k)+'.tif'), spike)

            #patches[:, :, k] = largeimg[xcrop:xcrop+208,ycrop:ycrop+208]
            #heatmaps[:, :, k] = spike_images[xcrop:xcrop+208,ycrop:ycrop+208]

            #scale1 = patches[:, :, k]/np.max(patches[:, :, k])*255
            #scale2 = heatmaps[:, :, k]/np.max(heatmaps[:, :, k])*255
            #blend = Image.blend(Image.fromarray(scale1).convert('RGBA'),Image.fromarray(scale2).convert('RGBA'),0.5)
            #plt.imshow(blend)
            #plt.show()
            #plt.draw()
            #plt.pause(0.1)
            k += 1