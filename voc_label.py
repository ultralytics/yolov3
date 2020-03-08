import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
sets = ['train', 'test','val']
classes = ["holothurian", "echinus", "scallop", "starfish"]
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def convert_annotation(image_id, Image_root):
    in_file = open('data/Annotations/%s.xml' % (image_id))
    out_file = open('data/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size') # 根据不同的xml标注习惯修改
    if size:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    else:
        jpg_img_patch = Image_root + image_id + '.jpg'
        jpg_img = cv2.imread(jpg_img_patch)
        h, w, _ = jpg_img.shape  # cv2读取的图片大小格式是w,h

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult:
            difficult = obj.find('difficult').text
        else:
            difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
wd = getcwd()
print(wd)
Image_root = './data/images/'

for image_set in sets:
    if not os.path.exists('data/labels/'):
        os.makedirs('data/labels/')
    image_ids = open('data/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open('data/underwater_%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('data/images//%s.jpg\n' % (image_id))  # 数据路径，可在这里修改，存放在这里并写到txt文件中
        convert_annotation(image_id, Image_root)
    list_file.close()
