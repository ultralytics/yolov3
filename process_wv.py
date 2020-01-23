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


from PIL import Image
import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import logging
import argparse
import os
import json
import wv_util as wv
import tfr_util as tfr
import aug_util as aug
import csv

"""
  A script that processes xView imagery. 
  Args:
      image_folder: A folder path to the directory storing xView .tif files
        ie ("xView_data/")

      json_filepath: A file path to the GEOJSON ground truth file
        ie ("xView_gt.geojson")

      test_percent (-t): The percentage of input images to use for test set

      suffix (-s): The suffix for output TFRecord files.  Default suffix 't1' will output
        xview_train_t1.record and xview_test_t1.record

      augment (-a): A boolean value of whether or not to use augmentation

  Outputs:
    Writes two files to the current directory containing training and test data in
        TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""


def get_images_from_filename_array(coords,chips,classes,folder_names,res=(250,250)):
    """
    Gathers and chips all images within a given folder at a given resolution.

    Args:
        coords: an array of bounding box coordinates
        chips: an array of filenames that each coord/class belongs to.
        classes: an array of classes for each bounding box
        folder_names: a list of folder names containing images
        res: an (X,Y) tuple where (X,Y) are (width,height) of each chip respectively

    Output:
        images, boxes, classes arrays containing chipped images, bounding boxes, and classes, respectively.
    """

    images =[]
    boxes = []
    clses = []

    k = 0
    bi = 0   
    
    for folder in folder_names:
        fnames = glob.glob(folder + "*.tif")
        fnames.sort()
        for fname in tqdm(fnames):
            #Needs to be "X.tif" ie ("5.tif")
            name = fname.split("\\")[-1]
            arr = wv.get_image(fname)
            
            img,box,cls = wv.chip_image(arr,coords[chips==name],classes[chips==name],res)

            for im in img:
                images.append(im)
            for b in box:
                boxes.append(b)
            for c in cls:
                clses.append(cls)
            k = k + 1
            
    return images, boxes, clses

def shuffle_images_and_boxes_classes(im,box,cls):
    """
    Shuffles images, boxes, and classes, while keeping relative matching indices

    Args:
        im: an array of images
        box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
        cls: an array of classes

    Output:
        Shuffle image, boxes, and classes arrays, respectively
    """
    assert len(im) == len(box)
    assert len(box) == len(cls)
    
    perm = np.random.permutation(len(im))
    out_b = {}
    out_c = {}
    
    k = 0 
    for ind in perm:
        out_b[k] = box[ind]
        out_c[k] = cls[ind]
        k = k + 1
    return im[perm], out_b, out_c

'''
Datasets
_multires: multiple resolutions. Currently [(500,500),(400,400),(300,300),(200,200)]
_aug: Augmented dataset
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_folder", help="Path to folder containing image chips (ie 'Image_Chips/' ")
    parser.add_argument("json_filepath", help="Filepath to GEOJSON coordinate file")
    parser.add_argument("-t", "--test_percent", type=float, default=0.333,
                    help="Percent to split into test (ie .25 = test set is 25% total)")
    parser.add_argument("-s", "--suffix", type=str, default='t1',
                    help="Output TFRecord suffix. Default suffix 't1' will output 'xview_train_t1.record' and 'xview_test_t1.record'")
    parser.add_argument("-a","--augment", type=bool, default=False,
    				help="A boolean value whether or not to use augmentation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    #resolutions should be largest -> smallest.  We take the number of chips in the largest resolution and make
    #sure all future resolutions have less than 1.5times that number of images to prevent chip size imbalance.
    #res = [(500,500),(400,400),(300,300),(200,200)]
    res = [(300,300)]

    AUGMENT = args.augment
    SAVE_IMAGES = False
    images = {}
    boxes = {}
    train_chips = 0
    test_chips = 0

    #Parameters
    max_chips_per_res = 100000
    train_writer = tf.python_io.TFRecordWriter("xview_train_%s.record" % args.suffix)
    test_writer = tf.python_io.TFRecordWriter("xview_test_%s.record" % args.suffix)

    coords,chips,classes = wv.get_labels(args.json_filepath)

    for res_ind, it in enumerate(res):
        tot_box = 0
        logging.info("Res: %s" % str(it))
        ind_chips = 0

        fnames = glob.glob(args.image_folder + "*.tif")
        fnames.sort()

        for fname in tqdm(fnames):
            #Needs to be "X.tif", ie ("5.tif")
            #Be careful!! Depending on OS you may need to change from '/' to '\\'.  Use '/' for UNIX and '\\' for windows
            name = fname.split("/")[-1]
            arr = wv.get_image(fname)

            im,box,classes_final = wv.chip_image(arr,coords[chips==name],classes[chips==name],it)

            #Shuffle images & boxes all at once. Comment out the line below if you don't want to shuffle images
            im,box,classes_final = shuffle_images_and_boxes_classes(im,box,classes_final)
            split_ind = int(im.shape[0] * args.test_percent)

            for idx, image in enumerate(im):
                tf_example = tfr.to_tf_example(image,box[idx],classes_final[idx])

                #Check to make sure that the TF_Example has valid bounding boxes.  
                #If there are no valid bounding boxes, then don't save the image to the TFRecord.
                float_list_value = tf_example.features.feature['image/object/bbox/xmin'].float_list.value
                
                if (ind_chips < max_chips_per_res and np.array(float_list_value).any()):
                    tot_box+=np.array(float_list_value).shape[0]
                    
                    if idx < split_ind:
                        test_writer.write(tf_example.SerializeToString())
                        test_chips+=1
                    else:
                        train_writer.write(tf_example.SerializeToString())
                        train_chips += 1
     
                    ind_chips +=1

                    #Make augmentation probability proportional to chip size.  Lower chip size = less chance.
                    #This makes the chip-size imbalance less severe.
                    prob = np.random.randint(0,np.max(res))
                    #for 200x200: p(augment) = 200/500 ; for 300x300: p(augment) = 300/500 ...

                    if AUGMENT and prob < it[0]:
                        
                        for extra in range(3):
                            center = np.array([int(image.shape[0]/2),int(image.shape[1]/2)])
                            deg = np.random.randint(-10,10)
                            #deg = np.random.normal()*30
                            newimg = aug.salt_and_pepper(aug.gaussian_blur(image))

                            #.3 probability for each of shifting vs rotating vs shift(rotate(image))
                            p = np.random.randint(0,3)
                            if p == 0:
                                newimg,nb = aug.shift_image(newimg,box[idx])
                            elif p == 1:
                                newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
                            elif p == 2:
                                newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
                                newimg,nb = aug.shift_image(newimg,nb)
                                

                            newimg = (newimg).astype(np.uint8)

                            if idx%1000 == 0 and SAVE_IMAGES:
                                Image.fromarray(newimg).save('process/img_%s_%s_%s.png'%(name,extra,it[0]))

                            if len(nb) > 0:
                                tf_example = tfr.to_tf_example(newimg,nb,classes_final[idx])

                                #Don't count augmented chips for chip indices
                                if idx < split_ind:
                                    test_writer.write(tf_example.SerializeToString())
                                    test_chips += 1
                                else:
                                    train_writer.write(tf_example.SerializeToString())
                                    train_chips+=1
                            else:
                                if SAVE_IMAGES:
                                    aug.draw_bboxes(newimg,nb).save('process/img_nobox_%s_%s_%s.png'%(name,extra,it[0]))
        if res_ind == 0:
            max_chips_per_res = int(ind_chips * 1.5)
            logging.info("Max chips per resolution: %s " % max_chips_per_res)

        logging.info("Tot Box: %d" % tot_box)
        logging.info("Chips: %d" % ind_chips)

    logging.info("saved: %d train chips" % train_chips)
    logging.info("saved: %d test chips" % test_chips)
    train_writer.close()
    test_writer.close() 