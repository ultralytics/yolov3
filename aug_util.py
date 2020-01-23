"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageDraw
import skimage.filters as filters


"""
Image augmentation utilities to be used for processing the dataset.  Importantly, these utilities modify
    the images as well as their respective bboxes (for example, in rotation).  Includes:
    rotation, shifting, salt-and-pepper, gaussian blurring.  Also includes a 'draw_bboxes' function
    for visualizing augmented images and bboxes
"""


def rotate_image_and_boxes(img, deg, pivot, boxes):
    """
    Rotates an image and corresponding bounding boxes.  Bounding box rotations are kept axis-aligned,
        so multiples of non 90-degrees changes the area of the bounding box.

    Args:
        img: the image to be rotated in array format
        deg: an integer representing degree of rotation
        pivot: the axis of rotation. By default should be the center of an image, but this can be changed.
        boxes: an (N,4) array of boxes for the image

    Output:
        Returns the rotated image array along with correspondingly rotated bounding boxes
    """

    if deg < 0:
        deg = 360-deg
    deg = int(deg)
        
    angle = 360-deg
    padX = [img.shape[0] - pivot[0], pivot[0]]
    padY = [img.shape[1] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX, [0,0]], 'constant').astype(np.uint8)
    #scipy ndimage rotate takes ~.7 seconds
    #imgR = ndimage.rotate(imgP, angle, reshape=False)
    #PIL rotate uses ~.01 seconds
    imgR = Image.fromarray(imgP).rotate(angle)
    imgR = np.array(imgR)
    
    theta = deg * (np.pi/180)
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    #  [(cos(theta), -sin(theta))] DOT [xmin, xmax] = [xmin*cos(theta) - ymin*sin(theta), xmax*cos(theta) - ymax*sin(theta)]
    #  [sin(theta), cos(theta)]        [ymin, ymax]   [xmin*sin(theta) + ymin*cos(theta), xmax*cos(theta) + ymax*cos(theta)]

    newboxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        #The 'x' values are not centered by the x-center (shape[0]/2)
        #but rather the y-center (shape[1]/2)
        
        xmin -= pivot[1]
        xmax -= pivot[1]
        ymin -= pivot[0]
        ymax -= pivot[0]

        bfull = np.array([ [xmin,xmin,xmax,xmax] , [ymin,ymax,ymin,ymax]])
        c = np.dot(R,bfull) 
        c[0] += pivot[1]
        c[0] = np.clip(c[0],0,img.shape[1])
        c[1] += pivot[0]
        c[1] = np.clip(c[1],0,img.shape[0])
        
        if np.all(c[1] == img.shape[0]) or np.all(c[1] == 0):
            c[0] = [0,0,0,0]
        if np.all(c[0] == img.shape[1]) or np.all(c[0] == 0):
            c[1] = [0,0,0,0]

        newbox = np.array([np.min(c[0]),np.min(c[1]),np.max(c[0]),np.max(c[1])]).astype(np.int64)

        if not (np.all(c[1] == 0) and np.all(c[0] == 0)):
            newboxes.append(newbox)
    
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]], newboxes

def shift_image(image,bbox):
    """
    Shift an image by a random amount on the x and y axis drawn from discrete  
        uniform distribution with parameter min(shape/10)

    Args:
        image: the image to be shifted in array format
        bbox: an (N,4) array of boxes for the image

    Output:
        The shifted image and corresponding boxes
    """
    shape = image.shape[:2]
    maxdelta = min(shape)/10
    dx,dy = np.random.randint(-maxdelta,maxdelta,size=(2))
    newimg = np.zeros(image.shape,dtype=np.uint8)
    
    nb = []
    for box in bbox:
        xmin,xmax = np.clip((box[0]+dy,box[2]+dy),0,shape[1])
        ymin,ymax = np.clip((box[1]+dx,box[3]+dx),0,shape[0])

        #we only add the box if they are not all 0
        if not(xmin==0 and xmax ==0 and ymin==0 and ymax ==0):
            nb.append([xmin,ymin,xmax,ymax])
    
    newimg[max(dx,0):min(image.shape[0],image.shape[0]+dx),
           max(dy,0):min(image.shape[1],image.shape[1]+dy)] = \
    image[max(-dx,0):min(image.shape[0],image.shape[0]-dx),
          max(-dy,0):min(image.shape[1],image.shape[1]-dy)]
    
    return newimg, nb

def salt_and_pepper(img,prob=.005):
    """
    Applies salt and pepper noise to an image with given probability for both.

    Args:
        img: the image to be augmented in array format
        prob: the probability of applying noise to the image

    Output:
        Augmented image
    """

    newimg = np.copy(img)
    whitemask = np.random.randint(0,int((1-prob)*200),size=img.shape[:2])
    blackmask = np.random.randint(0,int((1-prob)*200),size=img.shape[:2])
    newimg[whitemask==0] = 255
    newimg[blackmask==0] = 0
        
    return newimg


def gaussian_blur(img, max_sigma=1.5):
    """
    Use a gaussian filter to blur an image

    Args:
        img: image to be augmented in array format
        max_sigma: the maximum variance for gaussian blurring

    Output:
        Augmented image
    """
    return filters.gaussian(img,np.random.random()*max_sigma,multichannel=True)*255

def draw_bboxes(img,boxes):
    """
    A helper function to draw bounding box rectangles on images

    Args:
        img: image to be drawn on in array format
        boxes: An (N,4) array of bounding boxes

    Output:
        Image with drawn bounding boxes
    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])

    idx = 0

    for b in boxes:
        xmin,ymin,xmax,ymax = b
        
        for j in range(3):
            draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    return source