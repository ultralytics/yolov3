# Original work Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2018 Defense Innovation Unit Experimental.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from PIL import Image
import tensorflow as tf
import io
import numpy as np

'''
TensorflowRecord (TFRecord) processing helper functions to be re-used by any scripts
    that create or read TFRecord files.
'''

def to_tf_example(img, boxes, class_num):
    """
    Converts a single image with respective boxes into a TFExample.  Multiple TFExamples make up a TFRecord.

    Args:
        img: an image array
        boxes: an array of bounding boxes for the given image
        class_num: an array of class numbers for each bouding box

    Output:
        A TFExample containing encoded image data, scaled bounding boxes with classes, and other metadata.
    """
    encoded = convertToJpeg(img)

    width = img.shape[0]
    height = img.shape[1]

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    
    for ind,box in enumerate(boxes):
        xmin.append(box[0] / width)
        ymin.append(box[1] / height)
        xmax.append(box[2] / width)
        ymax.append(box[3] / height) 
        classes.append(int(class_num[ind]))

    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/encoded': bytes_feature(encoded),
            'image/format': bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmin),
            'image/object/bbox/xmax': float_list_feature(xmax),
            'image/object/bbox/ymin': float_list_feature(ymin),
            'image/object/bbox/ymax': float_list_feature(ymax),
            'image/object/class/label': int64_list_feature(classes),
    }))
    
    return example

def convertToJpeg(im):
    """
    Converts an image array into an encoded JPEG string.

    Args:
        im: an image array

    Output:
        an encoded byte string containing the converted JPEG image.
    """
    with io.BytesIO() as f:
        im = Image.fromarray(im)
        im.save(f, format='JPEG')
        return f.getvalue()

def create_tf_record(output_filename, images, boxes):
    """ DEPRECIATED
    Creates a TFRecord file from examples.

    Args:
        output_filename: Path to where output file is saved.
        images: an array of images to create a record for
        boxes: an array of bounding box coordinates ([xmin,ymin,xmax,ymax]) with the same index as images
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    k = 0
    for idx, image in enumerate(images):
        if idx % 100 == 0:
            print('On image %d of %d' %(idx, len(images)))

        tf_example = to_tf_example(image,boxes[idx],fname)
        if np.array(tf_example.features.feature['image/object/bbox/xmin'].float_list.value[0]).any():
            writer.write(tf_example.SerializeToString())
            k = k + 1
    
    print("saved: %d chips" % k)
    writer.close()

## VARIOUS HELPERS BELOW ##

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))