# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Model validation metrics."""

import numpy as np

from ..metrics import ap_per_class


def fitness(x):
    """
    Calculates the model fitness as a weighted sum of 8 metrics.

    Args:
        x (np.ndarray): An array of shape [N, 8] representing metrics. Each row corresponds to a different set of
            predictions with 8 columns representing different metric values.

    Returns:
        float: The calculated fitness score, which is a weighted sum of the input metrics.

    Notes:
        The weights used for calculating the fitness score are [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.9].

    Examples:
        ```python
        import numpy as np
        from ultralytics.yolo.utils import fitness

        metrics = np.array([[0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.9, 0.8],
                            [0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.8, 0.7]])

        score = fitness(metrics)
        print(score)
        ```
    """
    w = [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.9]
    return (x[:, :8] * w).sum(1)


def ap_per_class_box_and_mask(
    tp_m,
    tp_b,
    conf,
    pred_cls,
    target_cls,
    plot=False,
    save_dir=".",
    names=(),
):
    """
    Calculates Average Precision (AP) metrics for both bounding boxes and masks.

    Args:
        tp_m (np.ndarray): True positive array for masks.
        tp_b (np.ndarray): True positive array for bounding boxes.
        conf (np.ndarray): Confidence scores of predictions.
        pred_cls (np.ndarray): Predicted class indices.
        target_cls (np.ndarray): True class indices.
        plot (bool): If True, generates precision-recall plots. Default is False.
        save_dir (str): Directory to save plots, if generated. Default is ".".
        names (tuple | list): Class names. Default is an empty tuple.

    Returns:
        dict: A dictionary containing AP metrics for both bounding boxes and masks with the following structure:
            - 'boxes': {
                'p' (np.ndarray): Precision values,
                'r' (np.ndarray): Recall values,
                'ap' (np.ndarray): Average precision values,
                'f1' (np.ndarray): F1 score values,
                'ap_class' (np.ndarray): Class-wise average precision.
            }
            - 'masks': {
                'p' (np.ndarray): Precision values,
                'r' (np.ndarray): Recall values,
                'ap' (np.ndarray): Average precision values,
                'f1' (np.ndarray): F1 score values,
                'ap_class' (np.ndarray): Class-wise average precision.
            }

    Notes:
        This function wraps around the `ap_per_class` function to separately compute metrics for boxes and masks.
        It provides an aggregated dictionary for ease of access and comparison.

    Examples:
        ```python
        tp_b = np.array([...])
        tp_m = np.array([...])
        conf = np.array([...])
        pred_cls = np.array([...])
        target_cls = np.array([...])

        results = ap_per_class_box_and_mask(tp_m, tp_b, conf, pred_cls, target_cls, plot=True, save_dir="./results")
        print(results)
        ```
    """
    results_boxes = ap_per_class(
        tp_b, conf, pred_cls, target_cls, plot=plot, save_dir=save_dir, names=names, prefix="Box"
    )[2:]
    results_masks = ap_per_class(
        tp_m, conf, pred_cls, target_cls, plot=plot, save_dir=save_dir, names=names, prefix="Mask"
    )[2:]

    return {
        "boxes": {
            "p": results_boxes[0],
            "r": results_boxes[1],
            "ap": results_boxes[3],
            "f1": results_boxes[2],
            "ap_class": results_boxes[4],
        },
        "masks": {
            "p": results_masks[0],
            "r": results_masks[1],
            "ap": results_masks[3],
            "f1": results_masks[2],
            "ap_class": results_masks[4],
        },
    }


class Metric:
    def __init__(self) -> None:
        """
        Initializes Metric class attributes for precision, recall, F1 score, AP values, and AP class indices.

        Attributes:
            p (list[float]): List to store precision values for each class.
            r (list[float]): List to store recall values for each class.
            f1 (list[float]): List to store F1 score values for each class.
            all_ap (list[list[float]]): Nested list to store AP values for each class and corresponding thresholds.

        Returns:
            None: This constructor does not return any value.

        Examples:
            ```python
            metric = Metric()
            # Now, metric.p, metric.r, metric.f1, and metric.all_ap are initialized as empty lists.
            ```
        """
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """
        AP@0.5 of all classes.

        Returns:
            np.ndarray | list: Average Precision at IoU=0.5 for all classes. Returns a list if no AP values are present.

        Notes:
            This property is particularly useful for evaluating object detection models where IoU threshold
            for considering a positive detection is set to 0.5.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Provides the average precision (AP) across all classes at different IoU thresholds.

        Returns:
            np.ndarray | list: The average precision values for all classes. If no data is available, an empty
            list is returned.

        Notes:
            The average precision is computed over IoU thresholds ranging from 0.5 to 0.95 with a step size of 0.05.
            This metric is useful for evaluating the performance of object detection models across different
            levels of localization precision.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Calculates the mean precision (mp) across all classes.

        Returns:
            float: The mean precision value across all classes. If there are no precision values available, returns 0.0.

        Notes:
            Precision is calculated for each class, and then the mean value is derived by averaging across all classes. This
            metric is used to evaluate the detection performance of the model by taking into account how many of the predicted
            objects are true positive detections.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Mean recall of all classes.

        Returns:
            float: The mean recall across all classes, providing a single metric of device performance.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Mean AP@0.5 of all classes.

        Returns:
            float: The mean Average Precision at IoU threshold 0.5 for all classes. Returns 0.0 if no AP values
            are available.

        Notes:
            AP@0.5 is computed using the `all_ap` attribute, which stores AP metrics for each class. If `all_ap`
            is not populated, the function returns 0.0. This metric provides a measure of the model's precision
            across all classes at a specific IoU threshold (0.5).
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Calculates the mean Average Precision (mAP) for all classes at IoU thresholds from 0.5 to 0.95.

        Returns:
            float: The mean Average Precision (mAP) for all classes, averaged over IoU thresholds from 0.5 to 0.95.

        Examples:
            ```python
            metric = Metric()
            metric.all_ap = np.random.rand(80, 10)  # Assuming 80 classes and 10 IoU thresholds for example
            mean_ap = metric.map
            print("Mean AP@0.5:0.95:", mean_ap)
            ```
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """
        Mean of results, return mean precision, recall, AP@0.5, and AP@0.5:0.95.

        Returns:
            tuple: A tuple containing four floats:
                - Mean precision (float): The average precision across all classes.
                - Mean recall (float): The average recall across all classes.
                - Mean AP@0.5 (float): The mean Average Precision at IoU threshold of 0.5 across all classes.
                - Mean AP@0.5:0.95 (float): The mean Average Precision at IoU thresholds from 0.5 to 0.95 in increments of 0.05.
        """
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """
        Class-aware result, return precision, recall, AP@0.5, and AP@0.5:0.95 for a specified class.

        Args:
            i (int): The index of the class for which to fetch results.

        Returns:
            tuple[float, float, float, float]: A tuple containing:
                - Precision for the class (float).
                - Recall for the class (float).
                - AP@0.5 for the class (float).
                - AP@0.5:0.95 for the class (float).

        Example:
            ```python
            metric = Metric()
            class_idx = 0
            precision, recall, ap50, ap = metric.class_result(class_idx)
            ```
        """
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        """
        Fetches and computes mean Average Precision (mAP) values for each class.

        Args:
            nc (int): Number of classes in the dataset.

        Returns:
            np.ndarray: An array containing the mAP values for each class. The array has a length of `nc`.

        Note:
            This function assumes that the Metric instance has been populated with relevant precision, recall, and AP values
            prior to calling this method. Ensure that the `ap_class_index` attribute is properly set to map class indices.

        Example:
            ```python
            metric = Metric()
            # Populate metric with precision, recall, and AP values...
            maps = metric.get_maps(nc=80)
            print(maps)
            ```
        """
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Updates the current metric attributes using the provided results tuple.

        Args:
            results (tuple(list[float], list[float], list[list[float]], list[float], list[int])):
                A tuple containing:
                - p: List of precision values for each class.
                - r: List of recall values for each class.
                - all_ap: List of lists containing AP values for each class at different IoU thresholds.
                - f1: List of F1 scores for each class.
                - ap_class_index: List of class indices corresponding to the AP values.

        Returns:
            None
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Metric for boxes and masks."""

    def __init__(self) -> None:
        """
        Initializes the Metrics class with separate Metric instances for precision, recall, and AP metrics of boxes and
        masks.

        Attributes:
          metric_box (Metric): An instance of the Metric class to track metrics for bounding boxes.
          metric_mask (Metric): An instance of the Metric class to track metrics for object masks.

        Examples:
          ```python
          metrics = Metrics()
          metrics.metric_box.update(results_box)
          metrics.metric_mask.update(results_mask)
          ```
        """
        self.metric_box = Metric()
        self.metric_mask = Metric()

    def update(self, results):
        """
        Update the metrics for boxes and masks with the provided results.

        Args:
            results (dict): A dictionary containing evaluation metrics for both 'boxes' and 'masks'.
                            The dictionary should have the following structure:
                            {
                                'boxes': {
                                    'p': ndarray,
                                    'r': ndarray,
                                    'ap': ndarray,
                                    'f1': ndarray,
                                    'ap_class': ndarray
                                },
                                'masks': {
                                    'p': ndarray,
                                    'r': ndarray,
                                    'ap': ndarray,
                                    'f1': ndarray,
                                    'ap_class': ndarray
                                }
                            }

        Returns:
            None

        Notes:
            This function updates the internal metrics for both boxes and masks by extracting and setting each metric from the
            provided results dictionary.

        Example:
            ```python
            metric_results = {
                'boxes': {
                    'p': np.array([0.9, 0.8]),
                    'r': np.array([0.7, 0.6]),
                    'ap': np.array([0.85, 0.75]),
                    'f1': np.array([0.8, 0.7]),
                    'ap_class': np.array([1, 2])
                },
                'masks': {
                    'p': np.array([0.9, 0.85]),
                    'r': np.array([0.65, 0.7]),
                    'ap': np.array([0.82, 0.78]),
                    'f1': np.array([0.75, 0.75]),
                    'ap_class': np.array([1, 2])
                }
            }
            metrics = Metrics()
            metrics.update(metric_results)
            ```
        """
        self.metric_box.update(list(results["boxes"].values()))
        self.metric_mask.update(list(results["masks"].values()))

    def mean_results(self):
        """
        Calculates and returns the combined mean results for metrics of both boxes and masks.

        Returns:
            tuple[float, float, float, float]: A tuple containing:
                - Mean precision of all classes across boxes and masks.
                - Mean recall of all classes across boxes and masks.
                - Mean AP@0.5 of all classes across boxes and masks.
                - Mean AP@0.5:0.95 of all classes across boxes and masks.

        Example:
            ```python
            metrics = Metrics()
            mean_results = metrics.mean_results()
            print(mean_results)  # Outputs a tuple (mp, mr, map50, map)
            ```
        """
        return self.metric_box.mean_results() + self.metric_mask.mean_results()

    def class_result(self, i):
        """
        Combines and returns class-specific results from 'metric_box' and 'metric_mask' for a given class index.

        Args:
            i (int): The class index for which the results are to be retrieved.

        Returns:
            tuple(float, float, float, float, float, float, float, float): A tuple containing:
                - `p_box` (float): Precision for boxes of the specified class.
                - `r_box` (float): Recall for boxes of the specified class.
                - `ap50_box` (float): Average Precision at IoU 0.5 for boxes of the specified class.
                - `ap_box` (float): Average Precision at IoU 0.5:0.95 for boxes of the specified class.
                - `p_mask` (float): Precision for masks of the specified class.
                - `r_mask` (float): Recall for masks of the specified class.
                - `ap50_mask` (float): Average Precision at IoU 0.5 for masks of the specified class.
                - `ap_mask` (float): Average Precision at IoU 0.5:0.95 for masks of the specified class.

        Example:
            ```python
            metrics = Metrics()
            precision, recall, ap50, ap, precision_mask, recall_mask, ap50_mask, ap_mask = metrics.class_result(0)
            ```
        """
        return self.metric_box.class_result(i) + self.metric_mask.class_result(i)

    def get_maps(self, nc):
        """
        Calculates and returns the combined mean Average Precision (mAP) scores for bounding boxes and masks.

        Args:
            nc (int): The number of classes for which to calculate mAP scores.

        Returns:
            np.ndarray: An array containing the mAP scores for each class.

        Notes:
            This method combines the individual mAP scores computed for bounding boxes and masks into a unified array. The results are essential for evaluating the performance of object detection and instance segmentation models.

        Examples:
            ```python
            metrics = Metrics()
            nc = 5  # Suppose there are 5 classes
            maps = metrics.get_maps(nc)
            print(maps)
            ```
        """
        return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)

    @property
    def ap_class_index(self):
        """
        Returns the AP class index, which is identical for both boxes and masks in this implementation.

        Returns:
            list[int]: A list of class indices used for AP calculation for both bounding boxes and masks.

        Notes:
            The class index is shared between box and mask metrics to maintain consistency during evaluation.

        Examples:
            To access the AP class index for a `Metrics` instance:

            ```python
            metrics = Metrics()
            ap_class_index = metrics.ap_class_index
            print(ap_class_index)
            ```
        """
        return self.metric_box.ap_class_index


KEYS = [
    "train/box_loss",
    "train/seg_loss",  # train loss
    "train/obj_loss",
    "train/cls_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP_0.5(B)",
    "metrics/mAP_0.5:0.95(B)",  # metrics
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP_0.5(M)",
    "metrics/mAP_0.5:0.95(M)",  # metrics
    "val/box_loss",
    "val/seg_loss",  # val loss
    "val/obj_loss",
    "val/cls_loss",
    "x/lr0",
    "x/lr1",
    "x/lr2",
]

BEST_KEYS = [
    "best/epoch",
    "best/precision(B)",
    "best/recall(B)",
    "best/mAP_0.5(B)",
    "best/mAP_0.5:0.95(B)",
    "best/precision(M)",
    "best/recall(M)",
    "best/mAP_0.5(M)",
    "best/mAP_0.5:0.95(M)",
]
