import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image, ExifTags

from utils.utils import xyxy2xywh, xywh2xyxy

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
vid_formats = ['.mov', '.avi', '.mp4']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        None

    return s


class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        self.height = img_size

        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'File Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img, *_ = letterbox(img0, new_shape=self.height)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416):
        self.cam = cv2.VideoCapture(0)
        self.height = img_size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cam.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img, *_ = letterbox(img0, new_shape=self.height)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=True, image_weights=False):
        with open(path, 'r') as f:
            img_files = f.read().splitlines()
            self.img_files = [x for x in img_files if os.path.splitext(x)[-1].lower() in img_formats]

        n = len(self.img_files)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        assert n > 0, 'No images found in %s' % path

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Read image shapes
            sp = 'data' + os.sep + path.replace('.txt', '.shapes').split(os.sep)[-1]  # shapefile path
            if not os.path.exists(sp):  # read shapes using PIL and write shapefile for next time (faster)
                s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
                np.savetxt(sp, s, fmt='%g')

            try:
                with open(sp, 'r') as f:  # read existing shapefile
                    s = np.array([x.split() for x in f.read().splitlines()], dtype=np.float64)
                    assert len(s) == n, 'Shapefile out of sync'
            except:
                os.remove(sp)
                print('Shapefile deleted: %s. Please rerun again.' % sp)

            # Sort by aspect ratio
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            ar = ar[i]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [None] * n
        preload_labels = False
        if preload_labels:
            self.labels = [np.zeros((0, 5))] * n
            iter = tqdm(self.label_files, desc='Reading labels') if n > 10 else self.label_files
            extract_bounding_boxes = False
            for i, file in enumerate(iter):
                try:
                    with open(file, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                        if l.shape[0]:
                            assert l.shape[1] == 5, '> 5 label columns: %s' % file
                            assert (l >= 0).all(), 'negative labels: %s' % file
                            assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                            self.labels[i] = l

                            # Extract object detection boxes for a second stage classifier
                            if extract_bounding_boxes:
                                p = Path(self.img_files[i])
                                img = cv2.imread(str(p))
                                h, w, _ = img.shape
                                for j, x in enumerate(l):
                                    f = '%s%sclassification%s%g_%g_%s' % (
                                        p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                                    if not os.path.exists(Path(f).parent):
                                        os.makedirs(Path(f).parent)  # make new output folder
                                    box = xywh2xyxy(x[1:].reshape(-1, 4)).ravel()
                                    box = np.clip(box, 0, 1)  # clip boxes outside of image
                                    result = cv2.imwrite(f, img[int(box[1] * h):int(box[3] * h),
                                                            int(box[0] * w):int(box[2] * w)])
                                    if not result:
                                        print('stop')
                except:
                    pass  # print('Warning: missing labels for %s' % self.img_files[i])  # missing label file
            assert len(np.concatenate(self.labels, 0)) > 0, 'No labels found. Incorrect label paths provided.'

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        img_path = self.img_files[index]
        label_path = self.label_files[index]
        hyp = self.hyp

        # Load image
        img = self.imgs[index]
        if img is None:
            img = cv2.imread(img_path)  # BGR
            assert img is not None, 'File Not Found ' + img_path
            if self.n < 1001:
                self.imgs[index] = img  # cache image into memory

        # Augment colorspace
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50  # must be < 1.0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value

            a = (random.random() * 2 - 1) * fraction + 1
            b = (random.random() * 2 - 1) * fraction + 1
            S *= a
            V *= b

            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        # Letterbox
        h, w, _ = img.shape
        if self.rect:
            shape = self.batch_shapes[self.batch[index]]
            img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='rect')
        else:
            shape = self.img_size
            img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='square')

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    self.labels[index] = x  # save for next time

            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratiow * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratioh * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = ratiow * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratioh * h * (x[:, 2] + x[:, 4] / 2) + padh

        # Augment image and labels
        if self.augment:
            img, labels = random_affine(img, 
                                        labels, 
                                        degrees=hyp['degrees'], 
                                        translate=hyp['translate'], 
                                        scale=hyp['scale'],
                                        shear=hyp['shear'])

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() > 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw


def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratiow, ratioh, dw, dh


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA,
                         borderValue=(128, 128, 128))  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets


def convert_images2bmp():
    # cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
    for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
        folder = os.sep + Path(path).name
        output = path.replace(folder, folder + 'bmp')
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        for f in tqdm(glob.glob('%s*.jpg' % path)):
            save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
            cv2.imwrite(save_name, cv2.imread(f))

    for label_path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
        with open(label_path, 'r') as file:
            lines = file.read()
        lines = lines.replace('2014/', '2014bmp/').replace('.jpg', '.bmp').replace(
            '/Users/glennjocher/PycharmProjects/', '../')
        with open(label_path.replace('5k', '5k_bmp'), 'w') as file:
            file.write(lines)
