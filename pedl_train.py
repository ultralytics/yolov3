"""
This example demonstrates how to port the yolov3 model
based on PyTorch to PEDL client.
"""
import argparse
import pathlib
import warnings

from typing import Any, Dict, Sequence, Tuple, Union

# import original yolov3 model files
import torch
from torch import nn
from utils.datasets import LoadImagesAndLabels
from models import Darknet, attempt_download
from utils.utils import labels_to_class_weights, compute_loss
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

import pedl
from pedl.frameworks.pytorch import PyTorchTrial, LRScheduler
from pedl.frameworks.pytorch.data import DataLoader

import test

from collections import Counter

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


def download_data() -> str:
    # For a start we hard_code the dataset from bind_mounts
    # Currently, PEDL Client does not support a download_data_fn
    return "data"


class LazyModule(object):
    """
    Since the model definition requires information from the dataloader, we write this class that
    makes sure it is only initailzed once. This helper class creates a global singleton that lazily
    initializes a module instance to be used by the trial class and data
    loaders function.
    """

    __instance = None

    @staticmethod
    def get():
        if LazyModule.__instance is None:
            # Initializing the augmented dataset on first call
            train_dataset = LoadImagesAndLabels(
                train_path,
                img_size,
                batch_size,
                augment=True,
                hyp=hyp,  # augmentation hyperparameters
                rect=opt.rect,  # rectangular training
                cache_labels=True,
                cache_images=opt.cache_images,
                single_cls=opt.single_cls,
            )
            LazyModule.__instance = train_dataset

        # Return this dataset on second call
        return LazyModule.__instance


def make_data_loaders(
    experiment_config: Dict[str, Any], hparams: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:

    train_dataset = LazyModule.get()
    test_dataset = LoadImagesAndLabels(
        test_path,
        img_size_test,
        batch_size * 2,
        hyp=hyp,
        rect=True,
        cache_labels=True,
        cache_images=opt.cache_images,
        single_cls=opt.single_cls,
    )
    train_dataloader = DataLoader(  # torch.utils.data.DataLoader
        train_dataset,
        batch_size=batch_size,
        shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
        collate_fn=train_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size * 2, collate_fn=test_dataset.collate_fn,
    )

    return (
        train_dataloader,
        test_dataloader,
    )


class WarmupMultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Warmup part, sets the i-th component of lr to an early value for the first few
    epochs.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        threshold (int): The number of epochs to use the lower value
            Default: 3
        early_value (float): The value the learning rate should be set to for early epochs
            Default: 0.1
        index (int): The index of the learning rate array that should be changed. Changes all
                    values if index is set to -1.
            Default: -1

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.1      if epoch < 3
        >>> # lr = 0.05     if 3<= epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = WarmupMultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        last_epoch=-1,
        threshold=3,
        early_value=0.1,
        index=-1,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.threshold = threshold
        self.early_value = early_value
        self.index = index
        self.original_value = 0.0
        self.memory_set = False
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                DeprecationWarning,
            )
        if self.last_epoch < self.threshold:
            values = [group["lr"] for group in self.optimizer.param_groups]
            if not self.memory_set:
                if self.index == -1:
                    self.original_value = values
                else:
                    self.original_value = values[self.index]
                self.memory_set = True
            values[self.index] = self.early_value
            return values

        if self.last_epoch == self.threshold:
            values = [group["lr"] for group in self.optimizer.param_groups]
            if self.index == -1:
                values = self.original_value
            else:
                values[self.index] = self.original_value
            return values

        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


class YOLOv3Trial(PyTorchTrial):
    def build_model(self) -> nn.Module:

        # Initialize model
        model = Darknet(cfg, arc=opt.arc)  # .to(device)

        # Fetch starting weights
        attempt_download(weights)
        chkpt = torch.load(weights)
        # load model
        try:
            chkpt["model"] = {
                k: v
                for k, v in chkpt["model"].items()
                if model.state_dict()[k].numel() == v.numel()
            }
            model.load_state_dict(chkpt["model"], strict=False)
        except KeyError as e:
            s = (
                "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. "
                "See https://github.com/ultralytics/yolov3/issues/657"
                % (opt.weights, opt.cfg, opt.weights)
            )
            raise KeyError(s) from e

        del chkpt

        model.nc = nc  # attach number of classes to model
        model.arc = opt.arc  # attach yolo architecture
        model.hyp = hyp  # attach hyperparameters to model

        train_dataset = LazyModule.get()

        # The model class weights depend on the dataset labels
        model.class_weights = labels_to_class_weights(
            train_dataset.labels, nc
        )  # attach class weights

        return model

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        """
        Required Method to use a learning rate scheduler
        Returns: PEDL scheduler object
        PEDL will handle the learning rate scheduler update based on the PEDL LRScheduler parameters
        If step_every_batch or step_every_epoch is True, PEDL will handle the .step().
        If both are false, the user will be in charge of calling .step().
        """
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=[round(opt.epochs * x) for x in [0.8, 0.9]],
            gamma=0.1,
            index=2,
            early_value=0.1,
        )

        return LRScheduler(scheduler, step_every_epoch=True,)

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in dict(model.named_parameters()).items():
            if ".bias" in k:
                pg2 += [v]  # biases
            elif "Conv2d.weight" in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        # We use Adam
        optimizer = optim.Adam(pg0, lr=hyp["lr0"])

        # to use SGD we would need to change the optimizer dynamically
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": hyp["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)

        return optimizer

    def train_batch(
        self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        (imgs, targets, paths, _) = batch

        imgs = imgs.float() / 255.0

        pred = model(imgs)
        loss, loss_items = compute_loss(pred, targets, model, not prebias)

        loss *= batch_size / 64

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss_items)

        return {"loss": loss}

    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader, model: nn.Module
    ) -> Dict[str, Any]:

        batch_size = 16

        results, maps = test.test(
            cfg,
            data,
            batch_size=batch_size * 2,
            img_size=img_size_test,
            model=model,
            conf_thres=0.1,  # 0.1 for speed
            # 1e-3 if opt.evolve or (final_epoch and is_coco) else 0.1,
            iou_thres=0.6,
            single_cls=opt.single_cls,
            dataloader=data_loader,
        )
        keys = [
            "P",
            "R",
            "mAP_at_0.5",
            "F1",
            "Some_number_1",
            "Some_number_2",
            "Some_number_3",
        ]

        return dict(zip(keys[:4], results))


if __name__ == "__main__":

    # Hard code default arguments
    opt = argparse.Namespace(
        accumulate=4,
        adam=True,
        arc="default",
        batch_size=16,
        bucket="",
        cache_images=False,
        cfg="cfg/yolov3-spp.cfg",
        data="data/coco2017.data",
        device="",
        epochs=273,
        images=117263,
        evolve=False,
        img_size=[416],
        multi_scale=False,
        name="",
        nosave=True,  # default False
        notest=True,  # default False
        rect=False,
        resume=False,
        single_cls=False,
        var=None,
        weights="weights/ultralytics68.pt",
    )

    hyp = {
        "giou": 3.54,  # giou loss gain
        "cls": 37.4,  # cls loss gain
        "cls_pw": 1.0,  # cls BCELoss positive_weight
        "obj": 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
        "obj_pw": 1.0,  # obj BCELoss positive_weight
        "iou_t": 0.225,  # iou training threshold
        "lr0": 0.000579,  # initial learning rate (SGD=5E-3, Adam=5E-4) #0.00579
        "lrf": -4.0,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
        "momentum": 0.937,  # SGD momentum
        "weight_decay": 0.000484,  # optimizer weight decay
        "fl_gamma": 0.5,  # focal loss gamma
        "hsv_h": 0.0138,  # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.678,  # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.36,  # image HSV-Value augmentation (fraction)
        "degrees": 1.98,  # image rotation (+/- deg)
        "translate": 0.05,  # image translation (+/- fraction)
        "scale": 0.05,  # image scale (+/- gain)
        "shear": 0.641,
    }  # image shear (+/- deg)

    cfg = opt.cfg
    data = opt.data
    img_size, img_size_test = (
        opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2
    )  # train, test sizes
    images = opt.images
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    data_dict = {
        "classes": "80",
        "train": "data/coco/train2017.txt",
        "valid": "data/coco/val2017.txt",
        "names": "data/coco.names",
    }
    train_path = data_dict["train"]
    test_path = data_dict["valid"]

    nc = 1 if opt.single_cls else int(data_dict["classes"])

    # PEDL arguments
    # We make some of the default hyperparameters and turn them into PEDL hyperparameters.

    prebias = pedl.Constant(value=True, name="Prebias")
    batch_size = pedl.Constant(value=16, name="batch_size")
    init_lr = pedl.Constant(value=0.000579, name="init_lr")

    # Once defined, we update the original hyperparameters with the PEDL values
    opt.batch_size = batch_size
    hyp["lr0"] = init_lr

    config = {
        "description": "yolov3_pytorch",
        "searcher": {
            "name": "single",
            "metric": "mAP_at_0.5",
            "max_steps": int(epochs * images * batch_size * accumulate / 100),
            "smaller_is_better": False,
        },
        "max_restarts": 0,
        "bind_mounts": [
            {"host_path": "/home/anton/yolov3/data", "container_path": "data"}
        ],
        "optimizations": {"aggregation_frequency": accumulate},
        "min_validation_period": int(
            images * batch_size * accumulate / 100 / 4
        ),  # validate after each 1/4 epoch
    }

    exp = pedl.Experiment(
        context_directory=str(pathlib.Path.cwd()),
        trial_def=YOLOv3Trial,
        make_data_loaders_fn=make_data_loaders,
        configuration=config,
    )
    exp.create()
