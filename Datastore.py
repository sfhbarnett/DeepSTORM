import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

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

    def getimage(self, index):
        return self.ims.pages[index].asarray()


def generate_masks(csvpath):
    csvpath = 'DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.csv'
    points = []

    with open(csvpath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.split(',')
            line[1] = int(float(line[1]))
            points.append(line)

    width = 64
    height = 64
    pixel_size = 160
    upsample = 8
    mask = np.zeros([width * upsample, height * upsample,20])

    for frame in range(20):
        for p in points:
            if p[1] == frame+1:
                x = round(float(p[2]) // pixel_size) * upsample
                y = round(float(p[3]) // pixel_size) * upsample
                mask[x, y, frame] += 100
    # plt.imshow(mask[:,:,-1])
    # plt.show()
    return mask


def transform(image, mask):
    mask = torch.from_numpy(mask).unsqueeze(0)
    # Random horizontal flipping
    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)
    # Random vertical flipping
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)
    return image, mask


class Datastore(Dataset):
    def __init__(self, filelist, masklist, root_dir, transforms=None):
        self.images = filelist
        self.masks = masklist
        self.root_dir = root_dir
        self.trainimagepath = os.path.join(self.root_dir, 'image')
        self.trainmaskpath = os.path.join(self.root_dir, 'label')
        self.transform = transforms
        self.masks = generate_masks('DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.csv')
        self.imstack = TiffStack('DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.tif')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.trainimagepath, self.images[idx])
        if self.transform is not None:
            image = self.transform(self.imstack.getimage(idx))
            masktransform = transforms.Compose([transforms.ToTensor()])
            mask = self.masks[:, :, idx]
            mask = masktransform(mask)
            # Flip image and elastic deform
            image, mask = transform(image, mask)
            sample = {'image': image, 'mask': mask}
        else:
            image = self.imstack.getimage(idx)
            mask = self.masks[:,:,idx]
            sample = {'image': image, 'mask': mask}
        return sample

if __name__ == '__main__':
    generate_masks()