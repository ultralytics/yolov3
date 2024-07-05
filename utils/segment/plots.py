# Ultralytics YOLOv3 🚀, AGPL-3.0 license

import contextlib
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .. import threaded
from ..general import xywh2xyxy
from ..plots import Annotator, colors


@threaded
def plot_images_and_masks(images, targets, masks, paths=None, fname="images.jpg", names=None):
    """
    Plots a grid of images with corresponding annotations and masks, optionally resizing and saving the result.

    Args:
        images (np.ndarray | torch.Tensor): Batch of images with shape (B, C, H, W). Can be a Numpy array or PyTorch tensor.
        targets (np.ndarray | torch.Tensor): Target annotations containing bounding box coordinates and class labels in
            the format (image_index, class, x_center, y_center, width, height [, confidence]). Can be a Numpy array or
            PyTorch tensor.
        masks (np.ndarray | torch.Tensor): Segmentation masks for the images with shape (B, H, W). Can be a Numpy array or
            PyTorch tensor.
        paths (list[str] | None): List of file paths corresponding to the images for display purposes. Default is None.
        fname (str): Filename to save the plotted image grid. Default is "images.jpg".
        names (list[str] | None): List of class names for annotations. Default is None.

    Returns:
        None: Saves the resulting image grid with annotations and masks to the specified filename.

    Notes:
        - Images can be either normalized (values between 0 and 1) or unnormalized (values between 0 and 255).
        - If `images` are normalized, they will be automatically scaled up to the range [0, 255].
        - Bounding boxes and masks are optionally resized to fit within a maximum image size condition.
        - Class labels and confidence scores are displayed if provided.
        - The function supports a maximum of 16 images in a 4x4 grid.

    Example:
    ```python
    import torch
    from ultralytics import plot_images_and_masks

    # Example usage with dummy data
    images = torch.randn(8, 3, 640, 640)
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.2],
        [1, 1, 0.5, 0.5, 0.3, 0.3]
    ])
    masks = torch.randint(0, 2, (8, 640, 640))
    plot_images_and_masks(images, targets, masks, fname='example.jpg')
    ```
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)

    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y : y + h, x : x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            idx = targets[:, 0] == i
            ti = targets[idx]  # image targets

            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype("int")
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f"{cls}" if labels else f"{cls} {conf[j]:.1f}"
                    annotator.box_label(box, label, color=color)

            # Plot masks
            if len(masks):
                if masks.max() > 1.0:  # mean that masks are overlap
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = len(ti)
                    index = np.arange(nl).reshape(nl, 1, 1) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)
                else:
                    image_masks = masks[idx]

                im = np.asarray(annotator.im).copy()
                for j, box in enumerate(boxes.T.tolist()):
                    if labels or conf[j] > 0.25:  # 0.25 conf thresh
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                annotator.fromarray(im)
    annotator.im.save(fname)  # save


def plot_results_with_masks(file="path/to/results.csv", dir="", best=True):
    """
    Plots training results from a CSV file, highlighting either the best or last metrics, with the option to save the
    plots in a specified directory.

    Args:
        file (str): Path to the CSV file containing the training results. Default is "path/to/results.csv".
        dir (str): The directory where the plot image will be saved. Default is an empty string.
        best (bool): Flag to indicate whether to highlight the best metrics (True) or the last metrics (False). Default is True.

    Returns:
        None

    Raises:
        AssertionError: If no results.csv files are found in the specified directory or parent directory of the provided file path.

    Notes:
        - The function plots 16 subplots from the CSV file values, each representing different metrics.
        - If the `best` parameter is True, the best metric values are highlighted on the plots. Otherwise, the last values are highlighted.
        - The generated plot is saved as 'results.png' in the specified directory.

    Examples:
        ```python
        # Plot results highlighting the best metrics and save in the same directory as the results.csv
        plot_results_with_masks(file="runs/exp1/results.csv")

        # Plot results, highlighting the last metrics, and save the plot in a custom directory
        plot_results_with_masks(file="runs/exp1/results.csv", dir="plots/", best=False)
        ```
    """
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            index = np.argmax(
                0.9 * data.values[:, 8] + 0.1 * data.values[:, 7] + 0.9 * data.values[:, 12] + 0.1 * data.values[:, 11]
            )
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=2)
                if best:
                    # best
                    ax[i].scatter(index, y[index], color="r", label=f"best:{index}", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[index], 5)}")
                else:
                    # last
                    ax[i].scatter(x[-1], y[-1], color="r", label="last", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[-1], 5)}")
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()
