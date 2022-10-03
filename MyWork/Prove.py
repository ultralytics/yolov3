from email.mime import image
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os
import fnmatch
from torch.utils.data import Dataset
from PIL import Image
from patch_functions import *
from loss_functions import *
from dataset_functions import *
import torch.optim as optim
import shutil
import scipy

# Transforming from PIL to Tensor
transform1 = transforms.ToTensor()

# Transforming from Tensor to PIL
transform2 = transforms.ToPILImage()

# images = [f for f in os.listdir('/home/andread98/yolov3/MyWork/data_mask') if f.endswith('.jpg')]

# for img in images:
#     image_path = '/home/andread98/yolov3/MyWork/data_mask/' + img
#     print(image_path)
#     mask_path = '/home/andread98/yolov3/MyWork/data_mask/mask/' + img[:-4] + '.pt'
#     print(mask_path)
#     image_PIL = Image.open(image_path).convert('RGB')
#     h, w = image_PIL.size
#     print(h,w)
#     mask_tensor = torch.load(mask_path)
#     print(mask_tensor.shape)
#     image_tensor = transform1(image_PIL)
#     print(image_tensor.shape)
#     random_attack = torch.ones((3,w,h))
#     image_final_tensor = random_attack*mask_tensor + image_tensor*(1 - mask_tensor)
#     image_final_tensor_PIL = transform2(image_final_tensor)
#     image_final_tensor_PIL.show() 
#     image_final_tensor_PIL.show()

# Masks to pad
images = [f for f in os.listdir('/home/andread98/yolov3/MyWork/data_mask2') if f.endswith('.jpeg')]

for image in images:
    start_path = '/home/andread98/yolov3/MyWork/data_mask2' + '/' + image
    end_path = '/home/andread98/yolov3/MyWork/data_mask2' + '/' + image[:-5] + '.jpg'
    os.rename(start_path,end_path)

# masks = [f for f in os.listdir('/home/andread98/yolov3/MyWork/data_mask/mask')]

# for mask in masks:
#     mask_path = '/home/andread98/yolov3/MyWork/data_mask/mask/' + '/' + mask
#     mask_tensor = torch.load(mask_path)
#     if len(mask_tensor.shape) == 2:
#         mask_tensor = mask_tensor.unsqueeze(0)
#         mask_tensor = mask_tensor.expand(3,-1,-1)
#         torch.save(mask_tensor, mask_path)