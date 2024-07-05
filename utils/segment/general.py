# Ultralytics YOLOv3 🚀, AGPL-3.0 license

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def crop_mask(masks, boxes):
    """
    Crops predicted masks by zeroing out regions outside the predicted bounding boxes.

    Args:
        masks (torch.Tensor): A tensor of shape [n, h, w] containing the masks to be cropped.
        boxes (torch.Tensor): A tensor of shape [n, 4] containing bounding box coordinates in relative point form.

    Returns:
        torch.Tensor: A tensor of cropped masks of shape [n, h, w].

    Notes:
        The function is vectorized to optimize performance, courtesy of Chong.

    Example:
        ```python
        import torch
        masks = torch.rand((5, 128, 128))  # example masks
        boxes = torch.tensor([
            [0.1, 0.1, 0.4, 0.4],
            [0.2, 0.2, 0.5, 0.5],
            [0.3, 0.3, 0.6, 0.6],
            [0.4, 0.4, 0.7, 0.7],
            [0.5, 0.5, 0.8, 0.8],
        ])  # example boxes

        cropped_masks = crop_mask(masks, boxes)
        ```
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Process and crop masks using upsampled prototypes.

    Args:
        protos (torch.Tensor): Tensor of shape [mask_dim, mask_h, mask_w] containing mask prototypes.
        masks_in (torch.Tensor): Tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): Tensor of shape [n, 4], representing bounding box coordinates for each mask.
        shape (tuple[int, int]): Tuple representing the input image size in (height, width) format.

    Returns:
        torch.Tensor: A tensor of shape (h, w, n) containing the processed and cropped masks.

    Notes:
        This function performs mask processing by first projecting the masks using the provided prototypes,
        then upsamples the result to the desired shape and crops the masks based on the given bounding boxes.
        The `bilinear` interpolation mode ensures smoother mask transitions.

    Example:
        ```python
        processed_masks = process_mask_upsample(protos, masks_in, bboxes, shape)
        ```
    """

    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Process and crop segmentation masks, optionally upsampling the masks after cropping.

    Args:
        protos (torch.Tensor): Tensor of shape [mask_dim, mask_h, mask_w], containing the mask prototypes.
        masks_in (torch.Tensor): Tensor of shape [n, mask_dim], where `n` is the number of masks after non-maximum suppression.
        bboxes (torch.Tensor): Tensor of shape [n, 4], containing the bounding box coordinates for each mask.
        shape (tuple[int, int]): Tuple representing the input image size (height, width).
        upsample (bool, optional): If True, upsample the cropped masks to the input image size. Defaults to False.

    Returns:
        torch.Tensor: A tensor of shape [height, width, n] containing the processed masks.

    Notes:
        - The function first crops the masks to the bounding box regions defined by `bboxes`.
        - If the `upsample` parameter is set to True, the cropped masks are then upsampled to match the specified `shape`.
        - The sigmoid function is applied to the mask predictions to obtain confidence scores, and a threshold of 0.5 is used
          to determine the final binary masks.

    Example:
        ```python
        masks = process_mask(protos, masks_in, bboxes, shape, upsample=True)
        ```

    References:
        - Ultralytics YOLOv3: https://github.com/ultralytics/ultralytics
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.5)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    rocess_mask_native(protos, masks_in, bboxes, shape)

    Upsamples masks and then crops them using bounding boxes.

    Args:
        protos (torch.Tensor): Tensor of shape [mask_dim, mask_h, mask_w] representing the prototype masks.
        masks_in (torch.Tensor): Tensor of shape [n, mask_dim] where n is the number of masks after non-maximum suppression (NMS).
        bboxes (torch.Tensor): Tensor of shape [n, 4] where n is the number of masks after NMS, representing bounding box coordinates.
        shape (tuple[int, int]): Tuple representing the input image size as (height, width).

    Returns:
        torch.Tensor: A binary mask tensor of shape (height, width, n).

    Notes:
        - This function performs bilinear interpolation to upscale the masks to the input image size before performing the crop operation.
        - This method is computationally optimized and leverages operations in both PyTorch and NumPy for efficient processing.

    Examples:
        ```python
        masks = process_mask_native(protos, masks_in, bboxes, (640, 640))
        ```

    See Also:
        - `crop_mask`: Function to crop masks using bounding boxes.
        - `F.interpolate`: PyTorch function for bilinear interpolation.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    Scale and resize image masks from the model input shape to the original image shape.

    Args:
        im1_shape (Tuple[int, int]): Model input shape as a tuple (height, width).
        masks (np.ndarray): Array of masks with shape (height, width, num_masks) or (height, width).
        im0_shape (Tuple[int, int, int]): Original image shape as a tuple (height, width, channels).
        ratio_pad (Tuple[Tuple[float, float], Tuple[float, float]] | None, optional): Scaling ratio and padding. Defaults to None.

    Returns:
        np.ndarray: Rescaled masks with the same shape as `im0_shape` in height and width.

    Raises:
        ValueError: If the number of dimensions of `masks` is neither 2 nor 3.

    Examples:
        ```python
        im1_shape = (640, 640)
        im0_shape = (1080, 1920, 3)
        masks = np.random.rand(640, 640, 3)

        resized_masks = scale_image(im1_shape, masks, im0_shape)
        ```

    Notes:
        This function rescales the coordinate system of masks from the model's input shape to fit the original image
        dimensions. It calculates the appropriate gain and padding, if not provided, to ensure proper alignment.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) for pairs of predicted and ground truth masks.

    Args:
        mask1 (torch.Tensor): A tensor of shape [N, n], where N is the number of predicted objects and n is the flattened image dimensions (image_width * image_height).
        mask2 (torch.Tensor): A tensor of shape [M, n], where M is the number of ground truth objects and n is the flattened image dimensions (image_width * image_height).
        eps (float): A small epsilon value to avoid division by zero. Default is 1e-7.

    Returns:
        torch.Tensor: A tensor of shape [N, M] representing the IoU scores between each predicted object and each ground truth object.

    Example:
        ```python
        pred_masks = torch.rand(5, 1024)
        gt_masks = torch.rand(3, 1024)
        iou_scores = mask_iou(pred_masks, gt_masks)
        ```
    """
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks_iou(mask1, mask2, eps=1e-7):
    """
    Calculate the Intersection over Union (IoU) for pairs of predicted and ground truth masks.

    Args:
        mask1 (torch.Tensor): Tensor of shape [N, n] representing N predicted masks, where n is the flattened image size
            (width * height).
        mask2 (torch.Tensor): Tensor of shape [N, n] representing N ground truth masks, where n is the flattened image
            size (width * height).
        eps (float, optional): Small constant for numerical stability to avoid division by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: The IoU for each pair of predicted and ground truth masks, with shape (N,).

    Notes:
        - Both `mask1` and `mask2` should have the same shape and be binary masks.
        - IoU is computed as the intersection area divided by the union area of the masks.

    Examples:
        ```python
        import torch
        from ultyalytics import masks_iou

        # Create dummy masks
        pred_masks = torch.tensor([[1, 0, 1], [0, 1, 1]])
        gt_masks = torch.tensor([[1, 1, 0], [0, 1, 1]])

        iou = masks_iou(pred_masks, gt_masks)
        print(iou)  # Output: tensor([0.3333, 1.0000])
        ```
    ```
    """
    intersection = (mask1 * mask2).sum(1).clamp(0)  # (N, )
    union = (mask1.sum(1) + mask2.sum(1))[None] - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks2segments(masks, strategy="largest"):
    """
    Converts binary masks to polygon segments using specified strategies ('largest' or 'concat').

    Args:
        masks (torch.Tensor): A tensor of shape (n, h, w) containing binary masks for n instances.
        strategy (str): Strategy to convert masks to segments. Options are 'largest' to select the largest segment
                        or 'concat' to concatenate all segments. Default is 'largest'.

    Returns:
        List[np.ndarray]: A list of arrays with each array containing (n, 2) coordinates, representing the polygon
                          segments for each mask.

    Note:
        Ensure `masks` tensor is in binary format before passing to this function.

    Examples:
        ```python
        import torch
        masks = torch.randint(0, 2, (5, 640, 640), dtype=torch.uint8)  # Random binary masks
        segments = masks2segments(masks, strategy='largest')
        ```
    """
    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments
