# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Dataloaders."""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, distributed

from ..augmentations import augment_hsv, copy_paste, letterbox
from ..dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker
from ..general import LOGGER, xyn2xy, xywhn2xyxy, xyxy2xywhn
from ..torch_utils import torch_distributed_zero_first
from .augmentations import mixup, random_perspective

RANK = int(os.getenv("RANK", -1))


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    mask_downsample_ratio=1,
    overlap_mask=False,
    seed=0,
):
    """
    Creates a DataLoader for images and labels with optional augmentations and distributed sampling.

    Args:
        path (str): Path to the dataset.
        imgsz (int | list[int]): Size(s) to resize images.
        batch_size (int): Number of samples per batch.
        stride (int): Model stride for downsizing.
        single_cls (bool, optional): Treat dataset as a single class. Defaults to False.
        hyp (dict, optional): Dictionary of hyperparameters. Defaults to None.
        augment (bool, optional): Apply augmentations to the dataset. Defaults to False.
        cache (bool, optional): Cache images for faster loading. Defaults to False.
        pad (float, optional): Padding value for dataset resizing. Defaults to 0.0.
        rect (bool, optional): Flag to use rectangular training batches. Defaults to False.
        rank (int, optional): Rank of the process for distributed training. Defaults to -1.
        workers (int, optional): Number of worker threads for data loading. Defaults to 8.
        image_weights (bool, optional): Use weighted image sampling. Defaults to False.
        quad (bool, optional): Quad batch processing flag. Defaults to False.
        prefix (str, optional): Prefix logging. Defaults to "".
        shuffle (bool, optional): Shuffle data. Defaults to False.
        mask_downsample_ratio (int, optional): Downsample ratio for mask processing. Defaults to 1.
        overlap_mask (bool, optional): Overlap mask flag. Defaults to False.
        seed (int, optional): Seed for random number generator. Defaults to 0.

    Returns:
        DataLoader (torch.utils.data.DataLoader | InfiniteDataLoader): A DataLoader instance for the dataset.

    Notes:
        - If `rect` is True, `shuffle` will be set to False.
        - Uses `LoadImagesAndLabelsAndMasks` for dataset loading and various functions for optional data augmentations.

    Example:
        ```python
        dataloader = create_dataloader(
            path='data/coco128.yaml',
            imgsz=640,
            batch_size=16,
            stride=32,
            augment=True
        )
        ```
    """
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsAndMasks(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            downsample_ratio=mask_downsample_ratio,
            overlap=overlap_mask,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing
    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0,
        min_items=0,
        prefix="",
        downsample_ratio=1,
        overlap=False,
    ):
        """
        Initializes image, label, and mask loading for training/testing with optional augmentations.

        Args:
            path (str): Path to the dataset directory or file.
            img_size (int): Size to which images are to be resized. Defaults to 640.
            batch_size (int): Number of samples per batch. Defaults to 16.
            augment (bool): If true, data augmentation is applied. Defaults to False.
            hyp (dict | None): Hyperparameters dictionary. Defaults to None.
            rect (bool): If true, rectangular images are used; otherwise, square images are used. Defaults to False.
            image_weights (bool): If true, loads images with weights for a weighted sampling. Defaults to False.
            cache_images (bool): If true, caches images in memory for faster training. Defaults to False.
            single_cls (bool): If true, treats the dataset as a single class. Defaults to False.
            stride (int): Stride value to be used in the model. Defaults to 32.
            pad (float): Padding added to the images. Defaults to 0.
            min_items (int): Minimum number of items (images/labels) required in a batch. Defaults to 0.
            prefix (str): Prefix for logging output. Defaults to an empty string.
            downsample_ratio (int): Ratio for mask downsampling. Defaults to 1.
            overlap (bool): If true, enables mask overlap handling. Defaults to False.

        Returns:
            None.

        Notes:
            This class extends `LoadImagesAndLabels` to include mask handling capabilities for segmentation tasks.
            It supports various augmentations and configurations to facilitate efficient model training and testing.
        """
        super().__init__(
            path,
            img_size,
            batch_size,
            augment,
            hyp,
            rect,
            image_weights,
            cache_images,
            single_cls,
            stride,
            pad,
            min_items,
            prefix,
        )
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap

    def __getitem__(self, index):
        """
        Fetches the dataset item at a given index, handling linear, shuffled, or image-weighted indexing.

        Args:
            index (int): The index of the dataset item to fetch.

        Returns:
            tuple: A tuple containing:
                - img (torch.Tensor): The processed image tensor in RGB format.
                - labels_out (torch.Tensor): The labels tensor corresponding to the image.
                - shapes (tuple | None): A tuple containing image shape transformations details, or None.

        Notes:
            - This method handles both mosaic and regular image loading, along with numerous augmentation techniques
              such as MixUp, HSV color-space adjustment, random perspective, and flipping.
            - It converts the image from HWC to CHW format and ensures it is contiguous in memory.
            - Labels are transformed appropriately to ensure they match the image modifications.
            - In case of overlap defined, masks are managed with sorted indices for proper label–mask alignment.

        Examples:
            ```python
            dataset = LoadImagesAndLabelsAndMasks(path='/data/images', img_size=640)
            img, labels, shapes = dataset[0]
            import matplotlib.pyplot as plt
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.show()
            ```
        """
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        masks = []
        if mosaic:
            # Load mosaic
            img, labels, segments = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels, segments = mixup(img, labels, segments, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()
            if len(segments):
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, segments = random_perspective(
                    img,
                    labels,
                    segments=segments,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                masks, sorted_idx = polygons2masks_overlap(
                    img.shape[:2], segments, downsample_ratio=self.downsample_ratio
                )
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                labels = labels[sorted_idx]
            else:
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        masks = (
            torch.from_numpy(masks)
            if len(masks)
            else torch.zeros(
                1 if self.overlap else nl, img.shape[0] // self.downsample_ratio, img.shape[1] // self.downsample_ratio
            )
        )
        # TODO: albumentations support
        if self.augment:
            # Albumentations
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
                    masks = torch.flip(masks, dims=[1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
                    masks = torch.flip(masks, dims=[2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return (torch.from_numpy(img), labels_out, self.im_files[index], shapes, masks)

    def load_mosaic(self, index):
        """
        Loads a 4-image mosaic for YOLOv3 training, combining one target image with three random images within specified
        border constraints.

        Args:
            index (int): The index of the target image to be used in the mosaic.

        Returns:
            tuple: A tuple containing:
                - img4 (np.ndarray): The combined 4-mosaic image of shape (2 * img_size, 2 * img_size, channels).
                - labels4 (np.ndarray): The combined label array of shape (N, 5) where N is the total number
                  of labels across all 4 images. Each label is in the format [class, x1, y1, x2, y2].
                - segments4 (list[np.ndarray]): A list of segment arrays, where each segment array contains
                  the coordinates of a polygon segmentation mask.

        Notes:
            The mosaic augmentation is a key component in YOLOv3 training, as it helps the model learn to recognize
            objects at varying scales and positional contexts. This method also handles optional augmentations such as MixUp
            and random perspective transformation to further diversify the training dataset.

        Example:
            ```python
            index = 5
            img4, labels4, segments4 = dataloader.load_mosaic(index)
            ```
        """
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels, segments = self.labels[index].copy(), self.segments[index].copy()

            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4, segments4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove
        return img4, labels4, segments4

    @staticmethod
    def collate_fn(batch):
        """
        Batches images, labels, paths, shapes, and masks; modifies label indices for target image association.

        Args:
          batch (list[tuple[torch.Tensor, torch.Tensor, str, Any, torch.Tensor]]):
              A list where each element is a tuple containing an image Tensor,
              label Tensor, a file path, shape information, and a mask Tensor.

        Returns:
          tuple[torch.Tensor, torch.Tensor, list[str], tuple, torch.Tensor]:
              A tuple containing batched images, labels, paths, shapes, and masks.
              - images (torch.Tensor): Batched images.
              - labels (torch.Tensor): Batched labels with index modified for target association.
              - paths (list[str]): List of image file paths.
              - shapes (tuple): Shape information for each image in the batch.
              - masks (torch.Tensor): Batched image masks.

        Notes:
          - The function assumes that the input `batch` is a list of tuples
            with each tuple representing one data point from the dataset.
          - The masks are concatenated along the first dimension to
            form a batched tensor.
        """
        img, label, path, shapes, masks = zip(*batch)  # transposed
        batched_masks = torch.cat(masks, 0)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks


def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Converts a list of polygons into a binary mask.

    Args:
        img_size (tuple[int, int]): Size of the image in the format (height, width).
        polygons (np.ndarray): Array of shape (N, M), where N is the number of polygons, and M is the number of points
            (should be an even number representing (x, y) coordinates).
        color (int, optional): Color value for filling the polygon. Default is 1.
        downsample_ratio (int, optional): Factor to downsample the binary mask by. Default is 1 (no downsampling).

    Returns:
        np.ndarray: A binary mask of the provided image size, with the polygons filled. The mask will be downsampled if
        `downsample_ratio` is specified.
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Converts a list of polygons into corresponding binary masks.

    Args:
        img_size (tuple(int, int)): The size of the image, specified as (height, width).
        polygons (list[np.ndarray]): A list of polygons where each polygon is an array of shape (N, M), with N being the number of polygons and M being the number of points, which should be divisible by 2.
        color (int): The color to use for the mask. Typically, this would be 1 for a binary mask.
        downsample_ratio (int, optional): Factor by which to downsample the mask size relative to the original image size. Default is 1 (no downsampling).

    Returns:
        np.ndarray: A binary mask with the same height and width as specified by `img_size`, with optional downsampling. Shape is (N, H, W), where N is the number of polygons, H is the image height divided by `downsample_ratio`, and W is the image width divided by `downsample_ratio`.

    Example:
        ```python
        img_size = (640, 640)
        polygons = [np.array([10, 10, 20, 10, 20, 20, 10, 20])]
        masks = polygons2masks(img_size, polygons, color=1, downsample_ratio=2)
        ```

    Notes:
        - The function makes use of `cv2.fillPoly` to fill polygons in an image of zeros, thereby creating a binary mask.
        - The `downsample_ratio` helps in adjusting the mask resolution to a smaller scale.
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """lygons2masks_overlap(img_size, segments, downsample_ratio=1):"""Converts polygon segments to an overlap mask.
    
        This function generates a mask where each pixel corresponds to the overlap between multiple polygons. The output 
        mask matrix assigns an integer value to each pixel based on the polygon it belongs to.
    
        Args:
            img_size (tuple): Size of the image as a tuple (height, width).
            segments (list[np.ndarray]): A list of polygon segments, where each segment is an array of shape (N, 2) 
                representing N points of the polygon.
            downsample_ratio (int, optional): Factor to downsample the generated mask. Defaults to 1.
    
        Returns:
            np.ndarray: An overlap mask of shape (height // downsample_ratio, width // downsample_ratio). 
                Integer values indicate unique polygons, with 0 representing background.
            np.ndarray: An array of indices that indicates the sorting order of the polygons by their computed area.
        """
        masks = np.zeros(
            (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
            dtype=np.int32 if len(segments) > 255 else np.uint8,
        )
        areas = []
        ms = []
        for si in range(len(segments)):
            mask = polygon2mask(
                img_size,
                [segments[si].reshape(-1)],
                downsample_ratio=downsample_ratio,
                color=1,
            )
            ms.append(mask)
            areas.append(mask.sum())
        areas = np.asarray(areas)
        index = np.argsort(-areas)
        ms = np.array(ms)[index]
        for i in range(len(segments)):
            mask = ms[i] * (i + 1)
            masks = masks + mask
            masks = np.clip(masks, a_min=0, a_max=i + 1)
        
        return masks, index
    """
    masks = np.zeros(
        (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index
