import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import fnmatch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn as nn
import os
# import random
# import math
# import torchvision.transforms.functional as TF
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
# import torch.optim as optim
# from torch.autograd import Variable

####################################################################################################
####################################################################################################
### Dataset Object ###

class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)  # to make it agrees with max_lab dimensions. We choose a max_lab to say: no more than 14 persons could stand in one picture
        return image, label

    def pad_and_scale(self, img, lab): # this method for taking a non-square image and make it square by filling the difference in w and h with gray
                                       # needed to keep proportions
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize)) # make a square image of dim 416 x 416
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)  # padding of the labels to have a pad_size = max_lab (14 here).
                                                                   # add 1s to make dimensions = max_lab x batch_size (14 x 6) after the images lines,
                                                                   # whose number is not known a priori
        else:
            padded_lab = lab
        return padded_lab

class VOCmask(Dataset):
    
    def __init__(self, img_dir, mask_dir, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_masks = len(fnmatch.filter(os.listdir(mask_dir), '*.pt'))
        assert n_images == n_masks, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.mask_paths = []
        for img_name in self.img_names:
            mask_path = os.path.join(self.mask_dir, img_name).replace('.jpg', '.pt').replace('.png', '.pt')
            self.mask_paths.append(mask_path)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        # print(self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.img_names[idx]).replace('.jpg', '.pt').replace('.png', '.pt')
        image = Image.open(img_path).convert('RGB')
        mask = torch.load(mask_path)
        mask.unsqueeze_(0)
        
        image, mask = self.pad_and_scale(image, mask)
        transform = transforms.ToTensor()
        image = transform(image)
        # label = self.pad_lab(label)  # to make it agrees with max_lab dimensions. We choose a max_lab to say: no more than 14 persons could stand in one picture
        return image, mask, self.img_names[idx]

    def pad_and_scale(self, img, mask): # this method for taking a non-square image and make it square by filling the difference in w and h with gray
                                       # needed to keep proportions
        """

        Args:
            img:

        Returns:

        """
        h, w = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = int((h - w) / 2)
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                mask = F.pad(input=mask, pad=(padding, padding, 0, 0), mode='constant', value=0)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                mask = F.pad(input=mask, pad=(0, 0, padding, padding), mode='constant', value=0)
        resize = transforms.Resize((self.imgsize,self.imgsize)) # make a square image of dim 416 x 416
        padded_img = resize(padded_img)     #choose here
        # mask = mask.expand(3,-1,-1)
        mask = resize(mask)
        mask.squeeze_(0)
        return padded_img, mask