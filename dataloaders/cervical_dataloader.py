
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import sys, os
import random
import logging
from PIL import Image

from os import listdir
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def normize(img_ndarray):

    #   window=[-1024, 1023]
    #     img[np.where(img < window[0])] = window[0]
    #     img[np.where(img > window[1])] = window[1]
    #     img -= window[0]

    ##origin 0-2048    normize to -1024-1024
    img_ndarray = img_ndarray - 1024 
    # img_ndarray[img_ndarray < -512] = -512.
    # img_ndarray[img_ndarray > 512] = 512.
    # img_ndarray = img_ndarray + 512. 
    # img_ndarray = img_ndarray / 1024

    img_ndarray[img_ndarray < -160] = -160.
    img_ndarray[img_ndarray > 240] = 240.
    img_ndarray = img_ndarray + 160. 
    img_ndarray = img_ndarray / 400

    # img_ndarray[img_ndarray < -360] = -360.
    # img_ndarray[img_ndarray > 440] = 440.
    # img_ndarray = img_ndarray + 360. 
    # img_ndarray = img_ndarray / 800

    # newimg = (newimg * 255).astype(np.uint8)
    return img_ndarray



class Cervical(Dataset):
    '''
    Nine cervical dataset
    '''
    def __init__(self, site_index, channel=1, info_path=None, data_path=None, split='train', transform=None):
        self.site_index = site_index
        self.split = split
        self.channel = channel
        self.info_path = info_path if info_path is not None else './dataset/cervical/cervical_info.csv'
        self.data_path = data_path if data_path is not None else '/data2/lsy/LNM_1123'
        self.transform = transform

        data_info = pd.read_csv(self.info_path)
        data_info = data_info.loc[(data_info['ID'] == self.site_index) & (data_info['split'] == self.split)]

        imgs_paths = [os.path.join(self.data_path, 'imgs', name + '.npy') for name in list(data_info['name'])]
        images = [np.load(path) for path in imgs_paths]
        mask_paths = [os.path.join(self.data_path,'masks', name + '.npy') for name in list(data_info['name'])]
        labels = [np.load(path) for path in mask_paths]

        images = np.array(images)
        images = normize(images)
        labels = np.array(labels).astype(int)
        
        self.images = images
        self.labels = labels

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

        sample = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        _, h, w = image.size()

        i = int(round((h - self.size[0]) / 2.))
        j = int(round((w - self.size[1]) / 2.))

        image = image[:, i:i+self.size[0], j:j+self.size[1]]
        label = label[:, i:i+self.size[0], j:j+self.size[1]]
        return {'image': image, 'label': label}
    
    
class SegmentationTransform(object):
    def __init__(self, output_size=None, flip_prob=0.5, angle_range=None, scale_range=None, crop_range=None, to_tensor=True):
        self.output_size = output_size
        self.flip_prob = flip_prob
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.crop_range = crop_range
        self.to_tensor = to_tensor

    def __call__(self, sample):
        image, label = sample['image'][0], sample['label']

        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = np.fliplr(image)
            label = np.fliplr(label)

        # Random rotation
        if self.angle_range:
            angle = random.uniform(self.angle_range[0], self.angle_range[1])
            center = (image.shape[1] // 2, image.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)
        # Random scaling
        if self.scale_range:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            new_h, new_w = int(scale * image.shape[0]), int(scale * image.shape[1])
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Random Center crop
        if self.crop_range:
            cy, cx = image.shape[0] // 2, image.shape[1] // 2
            if self.crop_range[0] != self.crop_range[1]:
                rand_crop_size = random.randint(self.crop_range[0], self.crop_range[1])
            else:
                rand_crop_size = self.crop_range[0]
            h, w = rand_crop_size, rand_crop_size
            image = image[cy-h//2:cy+h//2, cx-w//2:cx+w//2]
            label = label[cy-h//2:cy+h//2, cx-w//2:cx+w//2]
        # Resize
        if self.output_size:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.output_size, interpolation=cv2.INTER_NEAREST)

        # add dimension
        image = np.expand_dims(image, axis=0)
        
        # to tensor
        if self.to_tensor:
            image = torch.Tensor(image)
            label = torch.Tensor(label)   

        return {'image': image, 'label': label}


if __name__ == '__main__':
    train_transform = transforms.Compose([
        SegmentationTransform(
            flip_prob=0.5,
            angle_range=(-15, 15),
            scale_range=(0.9, 1.1),
            crop_range=(256, 286),
            output_size=(256, 256),
            to_tensor=True
        ),
    ])
    train_ds = Cervical(site_index=1, split='train', transform=train_transform, info_path='./dataset/cervical/cervical_info.csv')
    train_ds[4]
    # import matplotlib.pyplot as plt
    # plt.imsave('/home/lsy/PMC-Fed/test_image.png',train_ds[4]['image'][0]*255)
    import cv2
    gray_image = (train_ds[4]['image'][0]*255).astype(np.uint8)
    cv2.imwrite('saved_gray_image400.jpg', gray_image)
    print(len(train_ds))