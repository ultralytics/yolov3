# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    # YOLOv3 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        """
        Initializes the Albumentations class for optional YOLOv3 data augmentation with a default image size of 640.

        Args:
            size (int): The target size for image augmentation transformations. Defaults to 640.

        Returns:
            None

        Raises:
            ImportError: If the `albumentations` package is not installed.
            Exception: For any other exceptions that occur during initialization.

        Notes:
            This class leverages the Albumentations library to apply a series of data augmentation techniques, such as
            random resized cropping, blurring, grayscale conversion, CLAHE, random brightness/contrast adjustments,
            gamma adjustments, and image compression.

        Examples:
            ```python
            from ultralytics import Albumentations

            augmentor = Albumentations(size=512)
            transformed = augmentor.transform(image=image, bboxes=bboxes, class_labels=labels)
            ```
        """
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        """
        Applies transformations to an image and its bounding boxes with a probability `p`.

        Args:
            im (numpy.ndarray): Input image to be transformed.
            labels (numpy.ndarray): Array of shape (N, 5), where N is the number of bounding boxes. Each bounding box is
                represented as [class, x_center, y_center, width, height] in YOLO format.
            p (float): Probability of applying the transformations. Default is 1.0.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray): The transformed image and corresponding bounding boxes.
        """
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    """
    Normalizes RGB images in BCHW format using ImageNet statistics.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor or array representing the image(s) in BCHW format.
        mean (tuple[float, float, float]): The mean values for each channel used for normalization. Default is IMAGENET_MEAN.
        std (tuple[float, float, float]): The standard deviation values for each channel used for normalization. Default is IMAGENET_STD.
        inplace (bool): If True, performs the normalization in place without creating a new tensor or array. Default is False.

    Returns:
        np.ndarray | torch.Tensor: The normalized image(s) in the same format and type as the input.

    Examples:
        ```python
        import numpy as np
        from ultralytics import normalize

        # Create a random image with shape (1, 3, 224, 224)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # Normalize the image
        normalized_img = normalize(img)
        ```
    Notes:
        - When working with PyTorch tensors, consider setting `inplace=True` to avoid additional memory allocation.
        - Ensure that the input image tensor or array is in the BCHW format, where B is the batch size, C is the number of channels, and H and W are the height and width of the image.
    """
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Converts normalized images back to their original form using ImageNet statistics.

    Args:
        x (torch.Tensor): A tensor representing the normalized images in BCHW format.
        mean (tuple[float, float, float], optional): A tuple representing the mean used for normalization in RGB order.
            Defaults to IMAGENET_MEAN.
        std (tuple[float, float, float], optional): A tuple representing the standard deviation used for normalization in
            RGB order. Defaults to IMAGENET_STD.

    Returns:
        torch.Tensor: The denormalized images tensor in BCHW format.

    Example:
        ```python
        tensor = torch.randn(4, 3, 256, 256)  # a batch of 4 normalized images
        original_tensor = denormalize(tensor)
        ```

    Notes:
        This function assumes that the input images were initially normalized using the same mean and std parameters.
        Adjust the mean and std values accordingly if different parameters were used for normalization.
    """
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    Applies HSV color-space augmentation to an image using specified gains for hue, saturation, and value.

    Args:
        im (np.ndarray): Input image in BGR format.
        hgain (float, optional): Gain factor for hue adjustment. Default is 0.5.
        sgain (float, optional): Gain factor for saturation adjustment. Default is 0.5.
        vgain (float, optional): Gain factor for value (brightness) adjustment. Default is 0.5.

    Returns:
        np.ndarray: Augmented image in BGR format.

    Example:
        ```python
        import cv2
        from ultralytics import augment_hsv

        image = cv2.imread('image.jpg')
        augmented_image = augment_hsv(image)
        cv2.imshow('Augmented Image', augmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ```

    Notes:
        - The function applies random adjustments to the hue, saturation, and value channels of the image in HSV color space.
        - The input image is expected to be in BGR format as used by OpenCV.
        - The gains are randomly generated within the range specified by `hgain`, `sgain`, and `vgain`, and then applied to each channel.
    """
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    """
    Equalizes the histogram of a given BGR or RGB image using either CLAHE or standard equalization.

    Args:
        im (np.ndarray): Image array in BGR or RGB format with shape (height, width, channels).
        clahe (bool, optional): If True, applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve the
            contrast of the image. Defaults to True.
        bgr (bool, optional): If True, interprets the input image as BGR format, otherwise as RGB. Defaults to False.

    Returns:
        np.ndarray: Image array with equalized histogram in the same color space as the input (BGR or RGB).

    Note:
        CLAHE can be beneficial for improving local contrast and enhancing the definition of edges in an image, especially
        in low-light conditions.

    Example:
        ```python
        import cv2
        from ultrlalytics import hist_equalize

        img = cv2.imread('image.jpg')
        img_eq = hist_equalize(img, clahe=True, bgr=True)
        cv2.imwrite('equalized_image.jpg', img_eq)
        ```
    """
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    """
    Replicates the smallest bounding boxes in an image to augment the dataset by creating duplicates of half the
    smallest bounding boxes, updating the image and labels accordingly.

    Args:
        im (np.ndarray): Image array in which bounding boxes are to be duplicated, should be in the shape (H, W, C).
        labels (np.ndarray): Array of bounding box labels in the format [class, x1, y1, x2, y2], with each row representing
            one bounding box.

    Returns:
        None

    Notes:
        The function modifies the input image and labels in place by replicating half of the smallest bounding boxes to new
       , randomly selected positions within the image.

    Example:
        ```python
        import numpy as np
        import cv2

        # Sample image and bounding boxes
        image = cv2.imread('sample.jpg')
        bounding_boxes = np.array([[0, 10, 20, 30, 40], [1, 50, 60, 70, 80]])

        # Apply replicate function
        replicate(image, bounding_boxes)

        # The image and bounding_boxes are updated in place
        ```
    """
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resizes and pads an image to fit a specified shape, maintaining aspect ratio and handling optional parameters like
    scaling and padding.

    Args:
        im (np.ndarray): Image to be resized and padded.
        new_shape (tuple[int, int]): Desired shape (height, width) for the output image. Default is (640, 640).
        color (tuple[int, int, int]): Color value for padding. Default is (114, 114, 114).
        auto (bool): If True, adjusts padding to be a multiple of 'stride'. Default is True.
        scaleFill (bool): If True, stretches the image to fill the 'new_shape'. Default is False.
        scaleup (bool): If True, allows image scaling up. Default is True.
        stride (int): Stride value for padding. Default is 32.

    Returns:
        tuple[np.ndarray, tuple[float, float]]: Resized and padded image, and a tuple containing the image's width and
            height ratios.

    Notes:
        - Preserves aspect ratio of the input image.
        - Handles automatic stride-multiple integer padding when 'auto' is True.
        - Optionally stretches the image to fill the new shape if 'scaleFill' is True.

    Example:
        ```python
        img = cv2.imread("input.jpg")
        resized_img, ratio = letterbox(img, new_shape=(640, 640), color=(128, 128, 128), auto=False, scaleFill=False)
        ```
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    """
    Applies a random perspective transformation to an image and its bounding boxes for data augmentation.

    Args:
        im (np.ndarray): The input image to be augmented.
        targets (np.ndarray, optional): Bounding boxes of the objects in the image in the format [class, x_min, y_min, x_max, y_max]. Defaults to an empty tuple.
        segments (list of np.ndarray, optional): Segmentations of the objects in the image. Defaults to an empty tuple.
        degrees (float, optional): Range for random rotation degrees. Defaults to 10.
        translate (float, optional): Range for random translation as a fraction of image dimensions. Defaults to 0.1.
        scale (float, optional): Range for random scaling. Defaults to 0.1.
        shear (float, optional): Range for random shear in degrees. Defaults to 10.
        perspective (float, optional): Range for random perspective transformation. Defaults to 0.0.
        border (tuple of int, optional): Additional padding (border) to add to the image (height, width). Defaults to (0, 0).

    Returns:
        np.ndarray: Augmented image.
        np.ndarray: Transformed bounding boxes for the augmented image.

    Notes:
        The function applies a combination of translation, rotation, scaling, shear, and perspective transformations to the input image and its bounding boxes. It then filters out invalid bounding boxes that fall outside the image boundaries or have too little area after transformation.

        Example usage:
        ```python
        augmented_image, augmented_boxes = random_perspective(image, targets, degrees=15, translate=0.2, scale=0.2, shear=15)
        ```
    """
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    """
    Applies Copy-Paste augmentation (https://arxiv.org/abs/2012.07177) to an image, its labels, and segmentation masks.
    
    Args:
        im (np.ndarray): The input image in the format of a NumPy array.
        labels (np.ndarray): A ndarray of shape (n, 5) representing n bounding boxes, where each box is defined by
            [class, x1, y1, x2, y2].
        segments (list of np.ndarray): List of n segmentation masks corresponding to the bounding boxes.
        p (float): Probability of applying the Copy-Paste augmentation. Default is 0.5.
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: The augmented image.
            - np.ndarray: Updated labels with new bounding boxes added through the augmentation.
            - list of np.ndarray: Updated list of segmentation masks with new segments added through the augmentation.
    
    Notes:
        This function employs strategies to manage overlaps between pasted objects and existing ones, 
        ensuring up to 30% obscuration of existing labels is allowed.
    
    Examples:
        ```python
        augmented_image, augmented_labels, augmented_segments = copy_paste(im, labels, segments, p=0.5)
        ```
    """
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    """
    Applies cutout augmentation, possibly removing labels with more than 60% obscuration.

    Args:
      im (np.ndarray): The input image on which to apply cutout augmentation.
      labels (np.ndarray): The bounding box coordinates and classes in the shape (num_boxes, 5).
      p (float): The probability of applying the cutout augmentation. Defaults to 0.5.

    Returns:
      None: The function modifies `im` and `labels` in place.

    Notes:
      This function implements the cutout augmentation as described in https://arxiv.org/abs/1708.04552. It creates random
      masks of varying sizes and applies them to random locations in the image. Labels that overlap with the masks more
      than 60% are discarded.

    Example:
      ```python
      # Apply cutout augmentation with a default probability of 0.5
      cutout(image, labels)
      ```
    """
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """
    Applies MixUp augmentation by blending two images and their corresponding labels based on a random ratio.

    Args:
        im (np.ndarray): The primary image used for the mixup, with shape (height, width, channels).
        labels (np.ndarray): The bounding box labels for the primary image, with shape (n, 5), where each label is
            (class, x1, y1, x2, y2).
        im2 (np.ndarray): The secondary image to be blended with the primary image, with shape (height, width, channels).
        labels2 (np.ndarray): The bounding box labels for the secondary image, with shape (m, 5), where each label is
            (class, x1, y1, x2, y2).

    Returns:
        tuple: A tuple containing:
            - im (np.ndarray): The result of mixing the two images, with the same shape as the input images.
            - labels (np.ndarray): The concatenated bounding box labels from both images, with shape (n+m, 5).

    Example:
        ```python
        image1 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        labels1 = np.array([[0, 50, 50, 100, 100]])
        image2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        labels2 = np.array([[1, 30, 30, 80, 80]])

        mixed_image, mixed_labels = mixup(image1, labels1, image2, labels2)
        ```
    Notes:
        - This function performs image augmentation by blending the pixel values of two images based on a beta distribution.
        - The resulting image is a combination of the two input images, and the labels from both images are concatenated.

    Relevant Paper:
        - MixUp: https://arxiv.org/pdf/1710.09412.pdf
    """
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    """
    Evaluates candidate bounding boxes based on width, height, aspect ratio, and area thresholds.

    Args:
        box1 (np.ndarray): Array of shape (4, n) representing the original bounding boxes.
        box2 (np.ndarray): Array of shape (4, n) representing the transformed bounding boxes.
        wh_thr (int, optional): Minimum allowable width and height of the bounding boxes. Defaults to 2.
        ar_thr (int, optional): Maximum allowable aspect ratio of the bounding boxes. Defaults to 100.
        area_thr (float, optional): Minimum allowable area ratio threshold between the original and transformed boxes.
                                   Defaults to 0.1.
        eps (float, optional): Small value to prevent division by zero. Defaults to 1e-16.

    Returns:
        np.ndarray: Boolean array indicating which transformed bounding boxes meet the criteria.

    Note:
        This function is typically used in data augmentation processes to filter out unsuitable bounding boxes after
        transformations such as random perspective, cutout, or other geometric augmentations.

    Example:
        ```python
        box1 = np.array([[10, 20, 30, 40], [15, 25, 35, 45]])
        box2 = np.array([[12, 22, 32, 42], [18, 28, 38, 48]])
        candidates = box_candidates(box1, box2)
        ```
        This example evaluates whether the transformed boxes `box2` meet the criteria specified by the thresholds with
        respect to `box1`.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv3 classification Albumentations (optional, only used if package is installed)
    """
    Generates an Albumentations transform pipeline for image classification with optional augmentations.

    Args:
        augment (bool): Boolean flag to perform augmentation (default is True).
        size (int): Resize size for the image, default is 224.
        scale (tuple[float, float]): Tuple indicating the scale range for random resizing (default is (0.08, 1.0)).
        ratio (tuple[float, float]): Tuple indicating the aspect ratio range for random resize cropping
                                     (default is (0.75, 1.0/0.75)).
        hflip (float): Probability of performing a horizontal flip (default is 0.5).
        vflip (float): Probability of performing a vertical flip (default is 0.0).
        jitter (float): Color jitter factor (default is 0.4).
        mean (tuple[float, float, float]): Mean value for normalization (default is (0.485, 0.456, 0.406)).
        std (tuple[float, float, float]): Standard deviation value for normalization (default is (0.229, 0.224, 0.225)).
        auto_aug (bool): Boolean flag to use auto augmentations (requires additional implementation, default is False).

    Returns:
        albumentations.Compose: A composed list of transformations from the Albumentations library.

    Note:
        - This function requires the Albumentations library to be installed.
        - Auto augmentations are currently not implemented and will log a message instead.

    Example:
    ```python
    transform = classify_albumentations(augment=True, size=224, hflip=0.5)
    ```
    """
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, satuaration, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    """
    Applies classification transforms including center cropping, tensor conversion, and normalization.

    Args:
        size (int): Size to which the image will be resized before other transformations.

    Returns:
        torchvision.transforms.Compose: A composition of transformations including resizing, center cropping, conversion
        to tensor, and normalization using ImageNet statistics.

    Note:
        Ensure `size` is a positive integer.

    Example:
        ```python
        import torchvision.transforms as transforms

        transform = classify_transforms(size=256)
        ```
    """
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    # YOLOv3 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes the LetterBox class for YOLOv3 image preprocessing with optional auto-sizing and stride adjustments.

        Args:
            size (int | tuple[int, int]): Target size for the image, either an integer or a tuple of (height, width).
            auto (bool): If True, computes the padding automatically to ensure dimensions are multiples of the stride. Default is False.
            stride (int): The stride to ensure dimensions are multiples of, used only if `auto` is True. Default is 32.

        Returns:
            None

        Examples:
            ```python
            import torchvision.transforms as T

            # Example usage in a transformation pipeline
            transforms = T.Compose([LetterBox(640), T.ToTensor()])
            ```
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        """
        Resizes and pads image `im` to specified `size` and `stride`, possibly autosizing for the short side.

        Args:
          im (np.ndarray): Input image in HWC format with dtype consistent with OpenCV (usually np.uint8).

        Returns:
          np.ndarray: Resized and padded image.

        The input image is resized proportionally and padded to fit the target size while preserving aspect ratio. If
        `auto` is set to True, the image is resized such that the resulting dimensions are divisible by the stride value.

        Examples:
          ```python
          letterbox = LetterBox(size=640, auto=True, stride=32)
          im_out = letterbox(im)
          ```
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    # YOLOv3 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """
        Initializes the CenterCrop object with the given size.

        Args:
            size (int | tuple): Desired output size of the crop. If int, a square crop (size x size) is made. If tuple, it should
                be (height, width) representing the desired output size.

        Returns:
            CenterCrop: An instance of the CenterCrop class.

        Examples:
            ```python
            # Create a CenterCrop instance with a square crop of 640x640
            center_crop = CenterCrop(640)

            # Create a CenterCrop instance with a rectangular crop of 480x640
            center_crop = CenterCrop((480, 640))
            ```
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        """__call__(im: np.ndarray) -> np.ndarray:"""Crops and resizes an image to specified dimensions by extracting a center crop.
        
            Args:
                im (np.ndarray): Input image represented as a numpy array in HWC format.
        
            Returns:
                np.ndarray: Center-cropped image resized to the specified dimensions.
            
            Example:
                ```python
                cropper = CenterCrop(size=(640, 640))
                cropped_image = cropper(image)
                ```
            """
            imh, imw = im.shape[:2]
            m = min(imh, imw)  # min dimension
            top, left = (imh - m) // 2, (imw - m) // 2
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv3 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """
        Initializes the ToTensor class for converting image arrays to PyTorch tensors, with optional half precision.

        Args:
            half (bool): Whether to convert the tensor to half precision (float16). Default is False.

        Returns:
            None
        """
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        """
        Converts a BGR image in numpy array format to a PyTorch tensor in RGB format, with optional half precision.

        Args:
            im (np.ndarray): Input image in BGR format with shape (height, width, channels).

        Returns:
            torch.Tensor: Image converted to a tensor in RGB format with shape (channels, height, width), normalized to [0, 1].
                The tensor can be in half precision (float16) if specified, otherwise float32.

        Examples:
            ```python
            # Initialize ToTensor object
            to_tensor = ToTensor()

            # Example image in numpy format with shape (HWC)
            image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)

            # Convert image to PyTorch tensor
            tensor = to_tensor(image)
            print(tensor.shape)  # torch.Size([3, 640, 480])
            ```
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
