# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    Applies label smoothing to Binary Cross Entropy (BCE) targets, modifying positive and negative label values to
    reduce overfitting.

    Args:
        eps (float, optional): The smoothing factor, typically between 0 and 1. A higher value means more smoothing.
            Default is 0.1.

    Returns:
        (tuple[torch.Tensor, torch.Tensor]): A tuple containing the smoothed positive and negative label tensors.

    Examples:
        ```python
        pos, neg = smooth_BCE(eps=0.1)
        criterion = nn.BCELoss()
        loss = criterion(pred, pos)  # Use the smoothed positive label tensor in the loss computation
        ```
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        """
        Initializes the BCEBlurWithLogitsLoss module with a specified alpha value.

        Args:
            alpha (float, optional): A coefficient factor for reducing the impact of missing labels. Default is 0.05.

        Returns:
            None

        Examples:
            ```python
            # Initialize BCEBlurWithLogitsLoss with default alpha
            criterion = BCEBlurWithLogitsLoss()

            # Initialize BCEBlurWithLogitsLoss with a custom alpha
            criterion = BCEBlurWithLogitsLoss(alpha=0.1)
            ```

        Notes:
            This class extends nn.Module and internally uses nn.BCEWithLogitsLoss with 'reduction' set to 'none'.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """
        Calculates modified binary cross-entropy loss with logits, accounting for missing labels by using an alpha
        factor.

        Args:
            pred (torch.Tensor): Predicted logits from the model, with shape (N, *) where * means any number of additional dimensions.
            true (torch.Tensor): Ground truth binary labels for each element in `pred`, with the same shape as `pred`.

        Returns:
            torch.Tensor: The computed loss values for each element in the batch (same shape as `pred`).
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """
        Initializes FocalLoss with specified loss function, gamma, and alpha for enhanced training on imbalanced
        datasets.

        Args:
            loss_fcn (nn.Module): The base loss function to which Focal Loss will be applied. This should typically be
                `nn.BCEWithLogitsLoss()`.
            gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted. Default is 1.5.
            alpha (float): Balancing parameter that addresses the class imbalance. Default is 0.25.

        Returns:
            None

        Notes:
            The Focal Loss is designed to address the class imbalance by down-weighting the loss assigned to well-classified
            examples. This implementation wraps the focal loss around an existing base loss function like
            `nn.BCEWithLogitsLoss`. For further reading on Focal Loss, refer to the original paper:
            https://arxiv.org/abs/1708.02002
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """
        Computes the focal loss between predicted logits and true labels, incorporating modulating factors for
        imbalanced datasets.

        Args:
            pred (torch.Tensor): Predicted logits of shape (N, *), where N is the batch size and * represents additional dimensions.
            true (torch.Tensor): Ground truth labels of shape (N, *), with the same dimensions as `pred`.

        Returns:
            torch.Tensor: Calculated focal loss as a scalar if reduction is 'mean' or 'sum', otherwise per-element loss
            (torch.Tensor).

        Notes:
            Focal loss helps to focus training on hard examples and avoid the vast number of easy negatives, by down-weighting
            well-classified examples in the loss computation. For the TensorFlow implementation, see
            https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py

        Examples:
            ```python
            criterion = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5, alpha=0.25)
            loss = criterion(pred, true)
            ```
        """
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """
        Initializes QFocalLoss with a given loss function, gamma, and alpha for quality focal loss computation.

        Args:
            loss_fcn (nn.Module): The loss function to be wrapped, typically `nn.BCEWithLogitsLoss()`.
            gamma (float): Focusing parameter that controls the rate at which easy examples are down-weighted, default is 1.5.
            alpha (float): Balancing parameter to address class imbalance, default is 0.25.

        Returns:
            None

        Note:
            The specified `loss_fcn` must have its `reduction` attribute set to "none" to apply QFocalLoss correctly across each
            element. Ensure `loss_fcn` is an instance of `nn.BCEWithLogitsLoss`.

        Example:
            ```python
            import torch.nn as nn
            from your_module import QFocalLoss

            criterion = QFocalLoss(nn.BCEWithLogitsLoss(), gamma=2.0, alpha=0.5)
            ```
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """
        Calculates the Quality Focal Loss (QFL) between predicted outputs and true labels based on the specified loss
        function, gamma, and alpha.

        Args:
            pred (torch.Tensor): Predicted output logits from the model.
            true (torch.Tensor): Ground truth binary labels for each sample.

        Returns:
            torch.Tensor: Computed loss value, which can be a scalar if reduction is 'mean' or 'sum', or a tensor if reduction is 'none'.

        Note:
            The loss function must be of type `nn.BCEWithLogitsLoss` for proper compatibility with QFocalLoss.

        Example:
            ```python
            import torch
            import torch.nn as nn
            from your_loss_module import QFocalLoss

            loss_fn = QFocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5, alpha=0.25)
            preds = torch.randn((10,), requires_grad=True)
            truths = torch.randint(0, 2, (10,)).float()
            loss = loss_fn(preds, truths)
            ```
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """
        Initializes the ComputeLoss module, configuring the loss criteria for class and object prediction based on model
        hyperparameters and focal loss.

        Args:
            model (torch.nn.Module): The model for which the loss will be computed. It is used to extract device, hyperparameters,
                and specific model components like detection layers.
            autobalance (bool): Flag indicating whether to automatically balance the loss contributions from different output
                layers (default is False).

        Returns:
            None

        Notes:
            The method sets up the Binary Cross-Entropy Loss (BCE) with weighted loss components and optionally applies focal
            loss if specified in the model's hyperparameters. Class label smoothing is also configured based on model settings.
            The balancing weights for different layers and other required properties are initialized to facilitate loss
            computation. For details on label smoothing, refer to https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441

        Examples:
            ```python
            model = Model()
            compute_loss = ComputeLoss(model, autobalance=True)
            ```
        """
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """
        Computes the total loss by aggregating class, box, and object losses from predictions and targets.

        Args:
            p (list[torch.Tensor]): The predictions from the model for each scale, containing the bounding boxes,
                                    objectness scores, and class scores.
            targets (torch.Tensor): The ground truth labels with bounding box coordinates, class labels, and image indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the class loss, box loss, and object loss
                                                             (all of type torch.Tensor).

        Notes:
            This method computes the total loss by looping over each prediction scale, extracting the relevant
            subsets of predictions based on target information, and calculating intersection-over-union (IoU)
            for regression losses, binary cross-entropy (BCE) losses for objectness and class scores, and
            optionally applying focal loss modifications if configured.

                ```python
                # Example usage:
                loss_fn = ComputeLoss(model)
                class_loss, box_loss, obj_loss = loss_fn(predictions, targets)
                total_loss = class_loss + box_loss + obj_loss
                ```

            If using focal loss or label smoothing, the relevant modifications are automatically applied during
            the initialization of the ComputeLoss instance.
        """
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """
        Build matching anchor targets for the loss computation from the provided images and labels.

        Args:
            p (torch.Tensor): Predictions tensor of shape [batch_size, num_anchors, grid_height, grid_width, pred_components].
            targets (torch.Tensor): Targets tensor of shape [num_targets, 6], where each target consists of
                                    (image_index, class, x, y, width, height).

        Returns:
            tuple: Consisting of:
                - tcls (list of torch.Tensor): List of class targets for each layer.
                - tbox (list of torch.Tensor): List of box targets for each layer.
                - indices (list of tuples): List of tuples for each layer. Each tuple comprises (image indices, anchor indices,
                                           grid y indices, grid x indices).
                - anch (list of torch.Tensor): List of anchor boxes for each layer.

        Notes:
            - This function is a subroutine of the `ComputeLoss` class, providing essential targets for its loss calculation process.
            - The target tensors and predictions are dynamically adjusted to match the model's grid and anchor configurations.
            - Refer to the anchor-matching strategy and offset calculations [here](https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441).
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
