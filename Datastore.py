import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import os


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
        newsize = tuple([self.upsample*x for x in img.shape])
        img = np.array(Image.fromarray(img).resize(newsize, Image.Resampling.NEAREST))
        return img.astype('float32')

class TiffStackPredict:
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
        #self.getstats()

    def getimage(self, index):
        img = self.ims.pages[index].asarray()
        newsize = tuple([self.upsample*x for x in img.shape])
        img = np.array(Image.fromarray(img).resize(newsize, Image.Resampling.NEAREST))
        return img.astype('float32')

    def getstats(self):
        totimage = np.zeros((self.width,self.height,self.nfiles))
        for index in range(self.nfiles):
            print(index)
            image = self.getimage(index)
            image = image.astype('float32')
            image = image-np.min(image)
            image = image / np.max(image)
            totimage[:,:,index] = image
        self.mean = np.mean(totimage)
        self.std = np.std(totimage)
        print(self.mean)
        print(self.std)



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
    mask = np.zeros([width * upsample, height * upsample, 20])

    for frame in range(20):
        for p in points:
            if p[1] == frame+1:
                x = round(float(p[3]) / pixel_size * upsample)-1
                y = round(float(p[2]) / pixel_size * upsample)-1
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


class DatastoreOTF(Dataset):
    def __init__(self, transforms=None):
        self.transform = transforms
        self.masks = generate_masks('DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.csv')
        self.imstack = TiffStack('DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.tif')


    def __len__(self):
        return self.imstack.nfiles

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.imstack.getimage(idx))
            image -= torch.min(image)
            image /= torch.max(image)
            masktransform = transforms.Compose([transforms.ToTensor()])
            mask = self.masks[:, :, idx]
            # Flip image and elastic deform
            image, mask = transform(image, mask)
            mask -= torch.min(mask)
            mask /= torch.max(mask)
            sample = {'image': image.float(), 'mask': mask.float()}
        else:
            image = self.imstack.getimage(idx)
            mask = self.masks[:, :, idx]
            sample = {'image': image, 'mask': mask}
        return sample


class Datastore(Dataset):
    def __init__(self, transforms=None):
        self.transform = transforms
        self.path = 'augments'
        self.imgfiles = os.listdir(os.path.join(self.path, 'img'))
        self.masks = os.listdir(os.path.join(self.path, 'spikes'))
        self.imgfiles.remove('.DS_Store')
        self.masks.remove('.DS_Store')
        fordims = tf.imread(os.path.join(self.path,'img/'+self.imgfiles[0]))
        (self.width, self.height) = fordims.shape
        self.mean = 0.20957219784919204
        self.std = 0.14851551727749593
        #self.calculate_stats()

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(np.float32(tf.imread(os.path.join(self.path,'img/'+self.imgfiles[idx])))).float()
            plt.imshow(image.detach().numpy().squeeze())
            image -= torch.min(image)
            image /= torch.max(image)
            image = (image-self.mean)/self.std
            masktransform = transforms.Compose([transforms.ToTensor()])
            mask = tf.imread(os.path.join(self.path,'spikes/'+self.masks[idx]))
            # Flip image and elastic deform
            image, mask = transform(image, mask)
            # mask -= torch.min(mask)
            # mask /= torch.max(mask)
            sample = {'image': image.float(), 'mask': mask.float()}
        else:
            image = self.imstack.getimage(idx)
            mask = self.masks[:, :, idx]
            sample = {'image': image, 'mask': mask}
        return sample

    def calculate_stats(self):
        totalim = np.zeros((self.width,self.height,len(self.imgfiles)))
        for i in range(len(self.imgfiles)):
            print(i)
            image = tf.imread(os.path.join(self.path,'img/'+self.imgfiles[i]))
            image = image.astype('float32')
            image -= np.min(image)
            image /= np.max(image)
            totalim[:,:,i] = image
        self.mean = np.mean(totalim)
        self.std = np.std(totalim)
        print(self.mean)
        print(self.std)



if __name__ == '__main__':
    tforms = transforms.Compose([transforms.ToTensor()])
    ds = Datastore(tforms)
    files = ds.imgfiles
    for f in files:
        if f[-4:] != ".tif":
            print(f)
    #sample = ds[1]
    #plt.imshow(sample['image'].detach().numpy().squeeze())
    #plt.figure()
    #plt.imshow(sample['mask'].detach().numpy().squeeze())
    #plt.show()


