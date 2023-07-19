import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import SimpleITK as sitk
import random
import cv2
import torch
import torchvision.transforms as transforms


class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()

class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return  img, mask


def normize(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    # newimg = (newimg * 255).astype(np.uint8)
    return newimg


class Prostate(Dataset):
    '''
    Six prostate dataset (BIDMC, HK, I2CVB, ISBI, ISBI_1.5, UCL)
    '''
    def __init__(self, site, channel=1, info_path=None, data_path=None, split='train', transform=None):
        channels = {'BIDMC':channel, 'HK':channel, 'I2CVB':channel, 'RUNMC':channel, 'BMC':channel, 'UCL':channel}
        assert site in list(channels.keys())
        self.split = split
        self.transform = transform
        self.channel = channels[site]
        info_path = info_path if info_path is not None else './dataset/Prostate'
        data_path = data_path if data_path is not None else'/data2/lsy/ProstateMRI/processed/channel1/'

        with open(os.path.join(info_path, site+f'_{split}.txt'),'r') as f:
            f_names = [line.rstrip() for line in f.readlines()]
        images, labels = [], []
        for f_name in f_names:
            image = np.load(os.path.join(data_path,site,f_name + '.npy'))
            image = normize(image)
            images.append(image)
            labels.append(np.load(os.path.join(data_path,site,f_name  + '_segmentation.npy')))

        images = np.array(images)
        labels = np.array(labels).astype(int)
        
        self.images = images
        self.labels = labels

        # self.labels = self.labels.astype(np.long).squeeze()
        self.labels = self.labels.squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        '''
        if self.split == 'train':
            R1 = RandomRotate90()
            image, label = R1(image, label)
            R2 = RandomFlip()
            image, label = R2(image, label)
        '''
        if self.channel == 3:
            image = np.transpose(image,(2, 0, 1))
        
        elif self.channel == 1:
            image = np.expand_dims(image, axis=0)

        # label = np.expand_dims(label, axis=0)

        image = torch.Tensor(image)
        label = torch.Tensor(label)

        sample = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample


if __name__=='__main__':

    sites = ['RUNMC', 'BMC', 'I2CVB', 'UCL','BIDMC', 'HK']
    trainset = Prostate(site=sites[1], channel=1, split='train', data_path='/data/lsy/ProstateMRI/processed/channel1/')
    trainset[0]
    pass

