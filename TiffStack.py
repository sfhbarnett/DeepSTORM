import tifffile as tf
from PIL import Image
import numpy as np


class TiffStack:
    """
    Tiffstack holds information about the images to process and accesses them in a memory-efficient manner
    improvements: implement a TiffFile class for each image to more readily access parameters such as width/height
    :param pathname to the tif image
    """
    def __init__(self, pathname):
        self.ims = tf.TiffFile(pathname)
        self.nfiles = len(self.ims.pages)
        page = self.ims.pages[0]
        self.width = page.shape[0]
        self.height = page.shape[1]
        self.mean = 0.12578635345602895
        self.std = 0.09853518682642659
        self.upsample = 8
        # self.getstats()

    def getimage(self, index):
        img = self.ims.pages[index].asarray()
        return img

    def getimageupsampled(self, index):
        img = self.ims.pages[index].asarray()
        newsize = tuple([self.upsample*x for x in img.shape])
        img = np.array(Image.fromarray(img).resize(newsize, Image.Resampling.NEAREST))
        return img.astype('float32')

    def getstats(self):
        totimage = np.zeros((self.width, self.height, self.nfiles))
        for index in range(self.nfiles):
            print(index)
            image = self.getimage(index)
            image = image.astype('float32')
            image = image-np.min(image)
            image = image / np.max(image)
            totimage[:, :, index] = image
        self.mean = np.mean(totimage)
        self.std = np.std(totimage)
        print(self.mean)
        print(self.std)
