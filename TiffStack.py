import tifffile as tf
from PIL import Image

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
        self.upsample = 8

    def getimage(self, index):
        img = self.ims.pages[index].asarray()
        #ewsize = tuple([self.upsample*x for x in img.shape])
        #img = np.array(Image.fromarray(img).resize(newsize, Image.Resampling.NEAREST))
        return img