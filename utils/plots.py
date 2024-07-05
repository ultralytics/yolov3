# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Plotting utils."""

import contextlib
import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter1d
from ultralytics.utils.plotting import Annotator

from utils import TryExcept, threaded
from utils.general import LOGGER, clip_boxes, increment_path, xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
RANK = int(os.getenv("RANK", -1))
matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a predefined Ultralytics color palette.

        This class provides a set of colors typically used in Ultralytics visualizations and can convert color values from hex to RGB.

        Args:
            None

        Returns:
            None

        Notes:
            The color palette includes a variety of predefined colors represented in hex format, which are converted to RGB upon initialization. The palette is designed to provide visually distinct colors suitable for plotting and visualization tasks.
        """
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """
        Converts an index to a color from a predefined palette, with optional BGR format.

        Args:
            i (int): Index to select color from the palette.
            bgr (bool): If True, return the color in BGR format; otherwise, return in RGB format. Default is False.

        Returns:
            list[int]: Color as a list of three integers [R, G, B] or [B, G, R] depending on the `bgr` parameter.

        Examples:
            ```python
            colors = Colors()
            color_rgb = colors(0)  # Returns first color in RGB format
            color_bgr = colors(0, bgr=True)  # Returns first color in BGR format
            ```
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        """
        Converts a hexadecimal color string to an RGB tuple.

        Args:
            h (str): Hexadecimal color string in the format '#RRGGBB'.

        Returns:
            tuple: A tuple (int, int, int) representing the RGB values.

        Examples:
            ```python
            color = Colors.hex2rgb('#FF5733')
            print(color)  # Output: (255, 87, 51)
            ```
        """
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    Visualizes feature maps of intermediate layers in a neural network.

    Args:
        x (torch.Tensor): The feature maps to be visualized, generally with shape (batch, channels, height, width).
        module_type (str): Type of the module from which features are derived.
        stage (int): The stage or depth of the module within the model, used to name the output file.
        n (int): Maximum number of feature maps to plot. Default is 32.
        save_dir (Path): Directory to save the resulting visualization image. Default is 'runs/detect/exp'.

    Returns:
        None

    Notes:
        - The function will only visualize feature maps if the height and width of `x` are greater than 1.
        - Feature maps are saved as PNG files in the specified `save_dir` with a naming convention based on the module type and stage.
        - The function logs the save operation details including the file path and number of feature maps plotted.

    Example:
        ```python
        import torch
        from pathlib import Path
        from your_module import feature_visualization

        # Assuming x is a tensor of shape (batch, channels, height, width)
        feature_visualization(x, 'Conv2d', 1, n=16, save_dir=Path('runs/detect/exp1'))
        ```
    """
    if "Detect" not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save


def hist2d(x, y, n=100):
    """
    Generates a 2D log-scaled histogram from input arrays `x` and `y`, with `n` bins for each axis.

    Args:
        x (np.ndarray): 1D array of x-coordinates of the input data points.
        y (np.ndarray): 1D array of y-coordinates of the input data points.
        n (int): Number of bins for each axis (default is 100).

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): A tuple containing:
            - hist: 2D array representing log-scaled histogram counts.
            - xedges: 1D array of x-axis bin edges.
            - yedges: 1D array of y-axis bin edges.

    Notes:
        This function uses `numpy.histogram2d` to compute the 2D histogram and then applies digitization to
        clip input coordinates to the appropriate bins.

    Examples:
        ```python
        import numpy as np
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        hist, xedges, yedges = hist2d(x, y, n=50)
        ```
    """
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    """
    Applies a low-pass Butterworth filter to data using forward-backward method.

    Args:
        data (np.ndarray): The input signal or data to be filtered.
        cutoff (float, optional): The cutoff frequency of the filter in Hz. Default is 1500.
        fs (float, optional): The sampling frequency of the data in Hz. Default is 50000.
        order (int, optional): The order of the Butterworth filter. Default is 5.

    Returns:
        np.ndarray: The filtered data after applying the low-pass Butterworth filter.

    Notes:
        This function uses the `butter` and `filtfilt` functions from SciPy's signal processing module.
        Refer to https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy for more details on filter implementation.

    Example:
        ```python
        import numpy as np
        from your_module import butter_lowpass_filtfilt

        # Create a sample signal with noise
        fs = 50000  # Sampling frequency
        t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
        freq = 1234  # Frequency of the signal
        x = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.randn(t.size)

        # Apply low-pass filter
        cutoff = 1500.0  # Desired cutoff frequency of the filter, Hz
        filtered_signal = butter_lowpass_filtfilt(x, cutoff, fs)

        print(filtered_signal)
        ```
    """
    from scipy.signal import butter, filtfilt

    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        """Applies a low-pass Butterworth filter to input data using forward-backward method; see https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype="low", analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def output_to_target(output, max_det=300):
    """
    Converts model output to target format for plotting and analysis, handling up to a specified number of detections.

    Args:
        output (list[torch.Tensor]): List of tensors representing the model's output, where each tensor corresponds to a
            batch element and contains detections as [x1, y1, x2, y2, objectness, class_score, class_id] for each detection.
        max_det (int, optional): Maximum number of detections to include per image. Defaults to 300.

    Returns:
        torch.Tensor: A tensor containing formatted detections in [batch_id, class_id, x_center, y_center, width, height, conf]
        format, where each row represents a detection.

    Notes:
        This function is typically utilized for preparing model outputs for visualization and further analysis.

    Example:
    ```python
    import torch

    # Simulated model output
    output = [torch.rand((10, 7)), torch.rand((5, 7))]

    # Convert to targets
    targets = output_to_target(output, max_det=300)
    print(targets)
    ```
    """
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    return torch.cat(targets, 0).numpy()


@threaded
def plot_images(images, targets, paths=None, fname="images.jpg", names=None):
    """
    Plots a grid of images with annotations for target boxes and class names.

    Args:
        images (torch.Tensor | numpy.ndarray): A batch of images to be plotted, with shape (batch_size, channels, height, width).
        targets (torch.Tensor | numpy.ndarray): Target annotations, with each row containing [batch_index, class_id, x, y, w, h, confidence (optional)].
        paths (list[str | pathlib.Path], optional): List of file paths for each image. Default is None.
        fname (str, optional): Filename to save the resulting image grid. Default is 'images.jpg'.
        names (list[str], optional): List of class names corresponding to class indices. Default is None.

    Returns:
        None

    Notes:
        - The function handles both normalized (0-1) and absolute coordinates for bounding boxes.
        - Up to 16 images are plotted in a square grid format (4x4 maximum).
        - The grid image is saved to the specified filename.

    Examples:
        ```python
        images = torch.rand(8, 3, 640, 640)  # Example batch of images
        targets = torch.tensor([[0, 1, 0.5, 0.5, 0.2, 0.2]])  # Example target annotations
        plot_images(images, targets, fname='output.jpg', names=['class_0', 'class_1'])
        ```
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

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
            ti = targets[targets[:, 0] == i]  # image targets
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
    annotator.im.save(fname)  # save


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=""):
    """
    Simulates and plots the learning rate (LR) schedule over a specified number of epochs and saves the plot to a file.

    Args:
        optimizer (torch.optim.Optimizer): The PyTorch optimizer whose learning rate schedule is to be plotted.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler associated with the optimizer.
        epochs (int, optional): The number of epochs over which to simulate the learning rate schedule. Default is 300.
        save_dir (str | pathlib.Path, optional): Directory where the generated plot will be saved. Default is the current
            directory.

    Returns:
        None: This function does not return any value. It saves the learning rate schedule plot as an image file in the
        specified directory.

    Notes:
        - This function makes a copy of the optimizer and scheduler to avoid modifying the original instances.
        - The plot is saved as 'LR.png' in the `save_dir`.

    Example:
        ```python
        import torch
        import torch.optim as optim
        import torch.optim.lr_scheduler as lr_scheduler
        from pathlib import Path

        # Define a simple optimizer and scheduler
        model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
        optimizer = optim.Adam(model, lr=0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        # Call the function
        plot_lr_scheduler(optimizer, scheduler, epochs=100, save_dir=Path('./results'))
        ```
        The example above will generate and save a plot showing how the learning rate decays over 100 epochs.
    """
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]["lr"])
    plt.plot(y, ".-", label="LR")
    plt.xlabel("epoch")
    plt.ylabel("LR")
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / "LR.png", dpi=200)
    plt.close()


def plot_val_txt():  # from utils.plots import *; plot_val()
    """
    Plots 2D and 1D histograms of object center locations from 'val.txt', saving results as 'hist2d.png' and
    'hist1d.png'.

    Reads bounding box coordinate data from 'val.txt', computes the center locations, and generates histograms to visualize
    their distributions. The 2D histogram ('hist2d.png') visualizes the density of center points, while two separate 1D
    histograms ('hist1d.png') visualize the distributions of the x and y coordinates of the center points.

    Returns:
        None

    Notes:
        Expects a file named 'val.txt' in the current working directory, containing bounding box data in the format
        [x_min, y_min, x_max, y_max, ...]. The first four columns are used to compute the center locations.

    Example:
        ```python
        from utils.plots import plot_val_txt
        plot_val_txt()
        ```
    """
    x = np.loadtxt("val.txt", dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect("equal")
    plt.savefig("hist2d.png", dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig("hist1d.png", dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    """
    Plots histograms for target attributes from 'targets.txt' and saves the resulting visualizations as 'targets.jpg'.

    Args:
        None

    Returns:
        None

    Notes:
        This function reads 'targets.txt' file, which is expected to contain target attribute data in a specific format.
        It then generates histograms for x, y, width, and height attributes of the targets. The generated plots are saved
        as 'targets.jpg'.

    Example:
        ```python
        from utils.plots import plot_targets_txt
        plot_targets_txt()
        ```
    """
    x = np.loadtxt("targets.txt", dtype=np.float32).T
    s = ["x targets", "y targets", "width targets", "height targets"]
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f"{x[i].mean():.3g} +/- {x[i].std():.3g}")
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig("targets.jpg", dpi=200)


def plot_val_study(file="", dir="", x=None):  # from utils.plots import *; plot_val_study()
    """
    Plots validation study results from 'study*.txt' files, comparing model performance and speed.

    Args:
        file (str): Path to a specific study file. Default is an empty string.
        dir (str): Directory containing study files. Default is an empty string.
        x (np.ndarray | None): Optional array of x-values for plotting. Default is None.

    Returns:
        None

    Notes:
        - This function generates a plot that compares various validation study results, focusing on key metrics such as Precision (P), Recall (R), mAP@.5, mAP@.5:.95, preprocessing time, inference time, and NMS time.
        - The plot also includes a comparison with EfficientDet for reference.
        - The function saves the generated plot as 'study.png' in the appropriate directory.
    """
    save_dir = Path(file).parent if file else Path(dir)
    plot2 = False  # plot additional results
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [save_dir / f'study_coco_{x}.txt' for x in ['yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for f in sorted(save_dir.glob("study*.txt")):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ["P", "R", "mAP@.5", "mAP@.5:.95", "t_preprocess (ms/img)", "t_inference (ms/img)", "t_NMS (ms/img)"]
            for i in range(7):
                ax[i].plot(x, y[i], ".-", linewidth=2, markersize=8)
                ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(
            y[5, 1:j],
            y[3, 1:j] * 1e2,
            ".-",
            linewidth=2,
            markersize=8,
            label=f.stem.replace("study_coco_", "").replace("yolo", "YOLO"),
        )

    ax2.plot(
        1e3 / np.array([209, 140, 97, 58, 35, 18]),
        [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
        "k.-",
        linewidth=2,
        markersize=8,
        alpha=0.25,
        label="EfficientDet",
    )

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(25, 55)
    ax2.set_xlabel("GPU Speed (ms/img)")
    ax2.set_ylabel("COCO AP val")
    ax2.legend(loc="lower right")
    f = save_dir / "study.png"
    print(f"Saving {f}...")
    plt.savefig(f, dpi=300)


@TryExcept()  # known issue https://github.com/ultralytics/yolov5/issues/5395
def plot_labels(labels, names=(), save_dir=Path("")):
    """
    Plots dataset labels correlogram, class distribution, and label geometry; saves the plots to the specified
    `save_dir`.

    Args:
        labels (np.ndarray): Array of labels with shape `[num_labels, 5]`, where each row represents `[class, x, y, w, h]`.
        names (dict[int, str] | list[str]): Dictionary or list mapping class indices to class names.
        save_dir (Path | str): Directory to save the resulting plots.

    Returns:
        None

    Notes:
        - The method generates multiple plots, including a correlogram, class distribution histogram, and label geometry
          heatmaps. The results are saved in the specified directory.
        - Requires `seaborn`, `matplotlib`, and `PIL` to be installed.
        - For further details, visit the GitHub repository:
          https://github.com/ultralytics/ultralytics

    Example:
        ```python
        labels = np.array([[0, 0.5, 0.5, 0.2, 0.2], [1, 0.4, 0.4, 0.3, 0.3]])
        names = {0: 'person', 1: 'car'}
        plot_labels(labels, names, save_dir=Path('output/'))
        ```
    """
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=["x", "y", "width", "height"])

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use("svg")  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    with contextlib.suppress(Exception):  # color histogram bars by class
        [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # known issue #3195
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    sn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / "labels.jpg", dpi=200)
    matplotlib.use("Agg")
    plt.close()


def imshow_cls(im, labels=None, pred=None, names=None, nmax=25, verbose=False, f=Path("images.jpg")):
    """
    Displays a grid of classification images with optional labels and predictions, saving the resulting image to a file.

    Args:
      im (Tensor): Tensor of images to be displayed, where the expected shape is (batch_size, channels, height, width).
      labels (Tensor, optional): Tensor of true labels corresponding to the images. Defaults to None.
      pred (Tensor, optional): Tensor of predicted labels corresponding to the images. Defaults to None.
      names (list[str], optional): List of class names. Defaults to None, in which case labels will be displayed as 'classX'.
      nmax (int, optional): Maximum number of images to display. Defaults to 25.
      verbose (bool, optional): If True, logs additional information. Defaults to False.
      f (Path, optional): Path to save the display image file. Defaults to 'images.jpg'.

    Returns:
      None

    Example:
      ```python
      import torch
      from pathlib import Path
      from utils import imshow_cls

      # Dummy data
      images = torch.randn(16, 3, 224, 224)  # 16 images
      labels = torch.randint(0, 10, (16,))  # 16 labels (0-9)
      predictions = torch.randint(0, 10, (16,))  # 16 predictions (0-9)
      class_names = [f"class{i}" for i in range(10)]

      imshow_cls(images, labels=labels, pred=predictions, names=class_names, nmax=16, verbose=True, f=Path("demo.jpg"))
      ```

    Notes:
      By default, the function will save the image grid to 'images.jpg'. To use a different save path, specify the `f` parameter. If the number of images exceeds `nmax`, only the first `nmax` images are displayed.
    """
    from utils.augmentations import denormalize

    names = names or [f"class{i}" for i in range(1000)]
    blocks = torch.chunk(
        denormalize(im.clone()).cpu().float(), len(im), dim=0
    )  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n**0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(0.0, 1.0))
        ax[i].axis("off")
        if labels is not None:
            s = names[labels[i]] + (f"—{names[pred[i]]}" if pred is not None else "")
            ax[i].set_title(s, fontsize=8, verticalalignment="top")
    plt.savefig(f, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose:
        LOGGER.info(f"Saving {f}")
        if labels is not None:
            LOGGER.info("True:     " + " ".join(f"{names[i]:3s}" for i in labels[:nmax]))
        if pred is not None:
            LOGGER.info("Predicted:" + " ".join(f"{names[i]:3s}" for i in pred[:nmax]))
    return f


def plot_evolve(evolve_csv="path/to/evolve.csv"):  # from utils.plots import *; plot_evolve()
    """
    Plots the evolution of hyperparameters from a CSV file, highlighting the best results.

    Args:
        evolve_csv (str): Path to the CSV file containing evolution data. Defaults to "path/to/evolve.csv".

    Returns:
        None. (Generates and saves a plot as a PNG file.)

    Example:
        ```python
        plot_evolve("runs/evolve/evolve.csv")
        ```

    Notes:
        - CSV file is expected to contain columns for fitness metrics and hyperparameters.
        - Best results are identified and highlighted on the plots.
        - Plot is saved in the same directory as the input CSV file.

    References:
        - Hyperparameter optimization for machine learning models.
        - https://github.com/ultralytics/yolov5/issues/5395

    ```
    """
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc("font", **{"size": 8})
    print(f"Best results from row {j} of {evolve_csv}:")
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, f.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print(f"{k:>15}: {mu:.3g}")
    f = evolve_csv.with_suffix(".png")  # filename
    plt.savefig(f, dpi=200)
    plt.close()
    print(f"Saved {f}")


def plot_results(file="path/to/results.csv", dir=""):
    """
    Plots training results from 'results.csv' files.

    Args:
    file (str): Path to the main 'results.csv' file to plot.
    dir (str): Directory containing multiple 'results*.csv' files to plot.

    Returns:
    None

    Examples:
    ```python
    plot_results(file='path/to/results.csv')
    plot_results(dir='path/to/results_dir')
    ```

    Notes:
    - The function searches for 'results*.csv' files in the specified directory or the parent directory of the given file.
    - It plots metrics including loss, accuracy, and learning rate over epochs.
    - Results are smoothed using a Gaussian filter for better visualization.
    - Saves the resulting plot as 'results.png' in the specified save directory.
    """
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.info(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()


def profile_idetection(start=0, stop=0, labels=(), save_dir=""):
    """
    Plots iDetection per-image logs from '*.txt', including metrics like storage and FPS.

    Args:
        start (int): Start time index for plotting.
        stop (int): Stop time index for plotting. If set to 0, defaults to the last index.
        labels (tuple | list): Tuple or list of label strings for the plots. Default is an empty tuple.
        save_dir (str | Path): Directory path where '*.txt' logs are stored and where the plot will be saved.

    Returns:
        None

    Note:
        Ensure '*.txt' files are present in `save_dir` for plotting. Each file should contain entries in per-image log format,
        typically including metrics like storage usage, RAM usage, battery levels, raw and smoothed detection times, etc.

    Example:
        ```python
        # Assuming '*.txt' logs are stored in 'path/to/logs'
        profile_idetection(start=0, stop=1000, labels=('run1', 'run2'), save_dir='path/to/logs')
        ```
    """
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ["Images", "Free Storage (GB)", "RAM Usage (GB)", "Battery", "dt_raw (ms)", "dt_smooth (ms)", "real-world FPS"]
    files = list(Path(save_dir).glob("frames*.txt"))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = results[0] - results[0].min()  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace("frames_", "")
                    a.plot(t, results[i], marker=".", label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel("time (s)")
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ["top", "right"]:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print(f"Warning: Plotting error for {f}; {e}")
    ax[1].legend()
    plt.savefig(Path(save_dir) / "idetection_profile.png", dpi=200)


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True): Saves or returns
    an enhanced crop from an image, defined by bounding box coordinates.

    Args:
        xyxy (torch.Tensor | list | tuple): Bounding box in (x1, y1, x2, y2) format.
        im (np.ndarray): Source image from which the crop is extracted.
        file (Path | str, optional): Destination file path for saving the crop image. Default is 'im.jpg'.
        gain (float, optional): Scaling factor applied to both width and height of the cropping box. Default is 1.02.
        pad (int, optional): Padding added to all sides of the cropping box. Default is 10.
        square (bool, optional): Whether to adjust the box to be square. Default is False.
        BGR (bool, optional): Whether the input image is in BGR format; if False, assumes RGB format. Default is False.
        save (bool, optional): Whether to save the cropped image to disk. If False, the crop is returned. Default is True.

    Returns:
        np.ndarray: Cropped image (only if `save` is False).

    Notes:
        - The bounding box dimensions are adjusted by `gain` and `pad` before extracting the crop.
        - The function ensures that the crop does not exceed image boundaries by clipping the bounding box coordinates.
        - Conversion between BGR and RGB formats is handled based on the `BGR` parameter.
        - Uses the `increment_path` function to avoid file name collisions when saving images.

    Example:
        ```python
        import cv2
        from pathlib import Path
        from ultralytics.utils.plotting import save_one_box

        img = cv2.imread("example.jpg")
        bounding_box = [50, 50, 150, 150]
        save_one_box(bounding_box, img, file=Path("cropped.jpg"))
        ```

    References:
        Ultralytics: https://github.com/ultralytics/ultralytics
    """
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop
