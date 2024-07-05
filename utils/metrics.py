# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import TryExcept, threaded


def fitness(x):
    """
    Calculates model fitness as a weighted sum of key performance metrics: Precision (P), Recall (R), mean Average Precision
    (mAP@0.5), and mean Average Precision over different IoU thresholds (mAP@0.5:0.95).

    Args:
      x (np.ndarray | list[float]): Array or list of metric values [P, R, mAP@0.5, mAP@0.5:0.95].

    Returns:
      float: The computed fitness score.

    Examples:
      ```python
      metrics = [0.8, 0.7, 0.8, 0.6]
      fitness_score = fitness(metrics)
      print(f"Fitness Score: {fitness_score}")
      ```
    """
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    """
    Smooths a given array using a box filter with a specified fractional size, returning the smoothed array.

    Args:
        y (np.ndarray): The array to be smoothed.
        f (float): Fractional size of the box filter; must be between 0 and 1. Higher values yield smoother results.

    Returns:
        np.ndarray: The smoothed array, which has the same length as the input array y.

    Notes:
        The number of elements in the box filter is determined by `nf = round(len(y) * f * 2) // 2 + 1`, ensuring it is
        always odd for symmetry.

    Examples:
        ```python
        import numpy as np
        from ultralytics.utils import smooth

        y = np.array([1, 2, 3, 4, 5])
        smoothed_y = smooth(y, f=0.1)
        print(smoothed_y)
        ```
    """
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Compute the average precision (AP) for each class given true positives, predicted confidence scores, predicted
    classes, and true classes. Additionally, it can plot the precision-recall curve and save it to the specified
    directory.

    Args:
        tp (np.ndarray): Array of true positives (shape: nx1 or nx10).
        conf (np.ndarray): Array of objectness scores ranging from 0 to 1.
        pred_cls (np.ndarray): Array of predicted class labels.
        target_cls (np.ndarray): Array of true class labels.
        plot (bool, optional): Flag to plot the precision-recall curve at mAP@0.5. Default is False.
        save_dir (str, optional): Directory to save the plot if `plot` is True. Default is the current directory '.'.
        names (tuple, optional): Tuple of class names. Default is an empty tuple.
        eps (float, optional): Small constant to avoid division by zero. Default is 1e-16.
        prefix (str, optional): Prefix for the plot filenames. Default is an empty string.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict): Tuple containing:
            - AP (average precision) array for each class and box (shape: nc x 10),
            - Precision array (shape: nc x 1000),
            - Recall array (shape: nc x 1000),
            - F1 score array (shape: nc x 1000),
            - True positives array (shape: nc),
            - False positives array (shape: nc),
            - Dictionary mapping class labels to class names.

    Source:
        https://github.com/rafaelpadilla/Object-Detection-Metrics

    Example:
        ```python
        tp = np.array([1, 0, 1, 1])
        conf = np.array([0.9, 0.95, 0.4, 0.7])
        pred_cls = np.array([0, 0, 1, 1])
        target_cls = np.array([0, 1, 1, 1])
        ap, p, r, f1, tp, fp, names = ap_per_class(tp, conf, pred_cls, target_cls, plot=True, save_dir="/tmp", names=("class1", "class2"))
        ```
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall")

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) from recall and precision curves.

    Args:
        recall (list[float]): The recall curve.
        precision (list[float]): The precision curve.

    Returns:
        float: The average precision, computed as the area under the precision-recall curve.
        numpy.ndarray: The precision curve with sentinel values appended.
        numpy.ndarray: The recall curve with sentinel values appended.

    Notes:
        This function uses linear interpolation by default to compute the area under the precision-recall curve.
        The precision envelope is constructed using the maximum precision value for each recall level, ensuring a
        non-increasing precision as recall increases.

    Example:
        ```python
        recall = [0.0, 0.5, 1.0]
        precision = [1.0, 0.5, 0.0]
        ap, mpre, mrec = compute_ap(recall, precision)
        print(f"AP: {ap}, Precision: {mpre}, Recall: {mrec}")
        ```
        This will produce AP as the area under the precision-recall curve.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """
        Initializes the confusion matrix for object detection with adjustable confidence and IoU thresholds.

        Args:
            nc (int): Number of classes in the dataset.
            conf (float): Confidence threshold for considering detections (default is 0.25).
            iou_thres (float): Intersection over Union (IoU) threshold for considering a detection as True Positive
                (default is 0.45).

        Returns:
            None: This constructor does not return any value.

        Notes:
            The confusion matrix is a square matrix with dimensions `(nc + 1) x (nc + 1)`, where `nc` is the number of
            classes. The last row and column are used to account for False Negatives and False Positives, respectively.
        """
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Processes a batch of detections and labels, updating the confusion matrix for object detection tasks.

        Args:
            detections (torch.Tensor): Detected objects with shape (N, 6), where each row consists of
                [x1, y1, x2, y2, confidence, class].
            labels (torch.Tensor): Ground truth labels with shape (M, 5), where each row consists of
                [class, x1, y1, x2, y2].

        Returns:
            None: The method updates the internal confusion matrix of the class based on IoU between
            detections and labels.

        Notes:
            - Detections with a confidence score below the threshold set during initialization are ignored.
            - The highest IoU match between detection and label is used for updating the matrix.
            - If no detections are provided, all labels are considered as false negatives.

        Examples:
            ```python
            cm = ConfusionMatrix(nc=80, conf=0.25, iou_thres=0.45)
            detections = torch.tensor([[50, 50, 100, 100, 0.9, 1]])
            labels = torch.tensor([[1, 55, 55, 105, 105]])
            cm.process_batch(detections, labels)
            ```
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def tp_fp(self):
        """
        Computes true positives and false positives, excluding the background class, from a confusion matrix.

        Args:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - True positives for each class (excluding the background class).
                - False positives for each class (excluding the background class).

        Note:
            The function assumes that the confusion matrix has been populated, and the last row and column of the matrix
            correspond to the background class.

        Example:
            ```python
            # Assuming cm is an instance of ConfusionMatrix with nc classes
            tp, fp = cm.tp_fp()
            print("True Positives:", tp)
            print("False Positives:", fp)
            ```
        """
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    def plot(self, normalize=True, save_dir="", names=()):
        """
        Plots the confusion matrix as a heatmap.

        Args:
            normalize (bool): Whether to normalize the confusion matrix values by the sum of each column.
                Default is True.
            save_dir (str): Directory where the generated plot will be saved. Default is an empty string.
            names (Iterable[str]): Iterable containing names of the classes. If provided and its length matches
                the number of classes, the names will be used as axis labels.

        Returns:
            None

        Notes:
            Uses seaborn for plotting the confusion matrix heatmap. If the number of classes is less than 30,
            the matrix values will be annotated on the heatmap. If normalization is applied, zero values will
            be displayed as NaN to prevent cluttering the heatmap.

        Example:
            ```python
            cm = ConfusionMatrix(nc=10)
            cm.process_batch(detections, labels)
            cm.plot(normalize=True, save_dir="results", names=["class1", "class2", ..., "class10"])
            ```
        """
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        plt.close(fig)

    def print(self):
        """
        Prints each row of the confusion matrix, where matrix elements are separated by spaces.

        Args:
            None

        Returns:
            None

        Examples:
            ```python
            # Assume cm is an initialized instance of ConfusionMatrix
            cm.print()
            ```
            This will print the confusion matrix to the standard output, with each row's elements separated by spaces.
        """
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates the Intersection over Union (IoU) or its variants (GIoU, DIoU, CIoU) between two bounding boxes.

    Args:
        box1 (torch.Tensor): A tensor representing the first bounding box, either in `xywh` or `xyxy` format.
        box2 (torch.Tensor): A tensor representing the second bounding box, either in `xywh` or `xyxy` format.
        xywh (bool, optional): If True, indicates that the bounding boxes are in `xywh` format. Default is True.
        GIoU (bool, optional): If True, computes the Generalized IoU (GIoU). Default is False.
        DIoU (bool, optional): If True, computes the Distance IoU (DIoU). Default is False.
        CIoU (bool, optional): If True, computes the Complete IoU (CIoU). Default is False.
        eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-7.

    Returns:
        torch.Tensor: A tensor containing the IoU or its specified variant.

    Notes:
        - IoU, GIoU, DIoU, and CIoU are metrics used to evaluate the overlap between two bounding boxes.
        - This function supports both `xywh` (center x, center y, width, height) and `xyxy` (min x, min y, max x, max y) formats.
        - The variants of IoU (GIoU, DIoU, CIoU) improve upon standard IoU by considering additional factors such as the
          distance between box centers and the size of the smallest enclosing box.

    Examples:
        ```python
        import torch
        from utils.metrics import bbox_iou

        box1 = torch.tensor([50, 50, 100, 100])  # xywh format
        box2 = torch.tensor([60, 60, 100, 100])  # xywh format

        iou = bbox_iou(box1, box2, xywh=True)
        print(f"IoU: {iou}")

        ciou = bbox_iou(box1, box2, xywh=True, CIoU=True)
        print(f"CIoU: {ciou}")
        ```
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4]): First set of bounding boxes.
        box2 (Tensor[M, 4]): Second set of bounding boxes.
        eps (float, optional): Small epsilon value to prevent division by zero. Defaults to 1e-7.

    Returns:
        Tensor[N, M]: An NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2.

    Example:
        >>> box1 = torch.tensor([[0, 0, 50, 50], [10, 10, 60, 60]])
        >>> box2 = torch.tensor([[0, 0, 50, 50], [15, 15, 55, 55]])
        >>> iou = box_iou(box1, box2)
        >>> print(iou)
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1, box2, eps=1e-7):
    """
    Returns the intersection over the area of `box2` given `box1` and `box2`.

    Args:
        box1 (np.ndarray): Bounding box in the format [x1, y1, x2, y2].
        box2 (np.ndarray): Multiple bounding boxes in the format [[x1, y1, x2, y2], ...].
        eps (float, optional): Small value to avoid division by zero. Default is 1e-7.

    Returns:
        np.ndarray: Intersection over area values for each box2 relative to box1.

    Notes:
        This function is used to calculate how much of `box1` intersects with each `box2` relative to the area of `box2`.

    Examples:
        ```python
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([[0, 0, 5, 5], [5, 5, 15, 15]])
        print(bbox_ioa(box1, box2))  # Output: array([0.25, 0.0])
        ```
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    """
    Calculates the Intersection over Union (IoU) for pairs of width-height dimensions.

    Args:
        wh1 (torch.Tensor): Tensor of shape [n, 2] representing n width-height pairs.
        wh2 (torch.Tensor): Tensor of shape [m, 2] representing m width-height pairs.
        eps (float): Small epsilon value to avoid division by zero (default is 1e-7).

    Returns:
        torch.Tensor: A tensor of shape [n, m] representing the pairwise IoU values for each combination of width-height pairs.

    Notes:
        The IoU is a measure of the overlap between two width-height pairs, which is useful for object detection tasks to
        determine how closely predicted bounding boxes match the ground truth bounding boxes.
    """
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    """
    Plots a precision-recall curve. Supports per-class curves visualization if there are fewer than 21 classes.

    Args:
        px (np.ndarray): Array of recall values (shape: [n_points]).
        py (list[np.ndarray]): List of precision arrays, each corresponding to a class (each shape: [n_points]).
        ap (np.ndarray): Array of average precision (AP) values for each class (shape: [n_classes, n_thresholds]).
        save_dir (Path | str, optional): File path to save the plot. Defaults to Path('pr_curve.png').
        names (tuple, optional): Class names.

    Returns:
        None
    ```python
    # Example usage
    recall = np.linspace(0, 1, 1000)
    precision = [np.random.rand(1000) for _ in range(5)]
    ap = np.random.rand(5, 1)
    plot_pr_curve(recall, precision, ap, save_dir=Path('plots/pr_curve.png'), names=('class1', 'class2', 'class3', 'class4', 'class5'))
    ```

    Notes:
    - The function utilizes the `threaded` decorator from the utils module to enable asynchronous plotting.
    - Supports up to 20 distinct class curve visualizations; more than 20 classes result in an aggregated plot.
    - Saves the plot as a PNG file in the specified directory with a DPI of 250.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    """
    Plots metric-confidence curve for given classes.

    Args:
        px (np.ndarray): Array of x-axis values, typically confidence scores.
        py (np.ndarray): 2D array of metric values, shape (number of classes, number of x-axis values).
        save_dir (str | Path): Directory to save the plot image.
        names (tuple): Names of the classes, used for the plot legend.
        xlabel (str): Label for the x-axis. Defaults to 'Confidence'.
        ylabel (str): Label for the y-axis, representing the performance metric. Defaults to 'Metric'.

    Returns:
        None

    Examples:
        ```python
        plot_mc_curve(px, py, save_dir="metrics/curve.png", names=("class1", "class2"), xlabel="Confidence", ylabel="Precision")
        ```
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
