import glob
import os
import argparse
import numpy as np
from utils.kmeans import kmeans, avg_iou
from utils.parse_config import parse_data_cfg

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']

def load_dataset(path):
    img_files = []
    with open(path, 'r') as f:
        img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
            if os.path.splitext(x)[-1].lower() in img_formats]
    label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                      for x in img_files]
    dataset = np.empty(shape=[0, 2])
    for label_path in label_files:
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

            if x.size > 0:
                x = x[:, 3:]
                dataset = np.append(dataset, x, axis=0)
    return dataset

def gen_anchors():
    data = opt.data
    img_size = opt.img_size
    clusters = opt.clusters

    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']

    data = load_dataset(train_path)
    out = kmeans(data, k=clusters)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    a = sorted(out * img_size, key = lambda x: x[0] * x[1])
    print("Sorted Boxes:\n {}".format(a))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--clusters', type=int, default=9, help='num of clusters for k-means')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    opt = parser.parse_args()
    print(opt)

    gen_anchors()
