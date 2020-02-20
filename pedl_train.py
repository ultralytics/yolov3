"""
This example demonstrates how to port the yolov3 model
based on PyTorch to PEDL client.
"""
import argparse
import pathlib


from typing import Any, Dict, Sequence, Tuple, Union

# import original yolov3 model files
import torch
from torch import nn
from utils.datasets import LoadImagesAndLabels
from models import Darknet, attempt_download
from utils.utils import labels_to_class_weights, compute_loss
from utils.warmuplr import WarmupMultiStepLR
import torch.optim as optim


import pedl
from pedl.frameworks.pytorch import PyTorchTrial, LRScheduler
from pedl.frameworks.pytorch.data import DataLoader

import test


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


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

        hyp = get_hyp(lr0=pedl_init_lr)

        opt = get_cli_args(
            batch_size=pedl_batch_size, prebias=pedl_prebias, accumulate=pedl_accumulate
        )
        data_dict = get_data_cfg()
        train_path = data_dict["train"]

        img_size, _ = (
            opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2
        )  # train, test sizes

        if LazyModule.__instance is None:
            # Initializing the augmented dataset on first call
            train_dataset = LoadImagesAndLabels(
                train_path,
                img_size,
                opt.batch_size,
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

    hyp = get_hyp(lr0=pedl_init_lr)
    opt = get_cli_args(
        batch_size=pedl_batch_size, prebias=pedl_prebias, accumulate=pedl_accumulate
    )
    data_dict = get_data_cfg()
    test_path = data_dict["valid"]

    train_dataset = LazyModule.get()

    _, img_size_test = opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2
    test_dataset = LoadImagesAndLabels(
        test_path,
        img_size_test,
        opt.batch_size * 2,
        hyp=hyp,
        rect=True,
        cache_labels=True,
        cache_images=opt.cache_images,
        single_cls=opt.single_cls,
    )
    train_dataloader = DataLoader(  # torch.utils.data.DataLoader
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
        collate_fn=train_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=opt.batch_size * 2, collate_fn=test_dataset.collate_fn,
    )

    return (
        train_dataloader,
        test_dataloader,
    )


class YOLOv3Trial(PyTorchTrial):
    def build_model(self) -> nn.Module:

        opt = get_cli_args(
            batch_size=pedl_batch_size, prebias=pedl_prebias, accumulate=pedl_accumulate
        )
        hyp = get_hyp(lr0=pedl_init_lr)

        # Initialize model
        model = Darknet(opt.cfg, arc=opt.arc)  # .to(device)

        # Fetch starting weights
        # TODO Once download_data_fn is implemented this should go into download_data
        attempt_download(opt.weights)
        chkpt = torch.load(opt.weights)

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

        data_dict = get_data_cfg()
        nc = 1 if opt.single_cls else int(data_dict["classes"])
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

        opt = get_cli_args(
            batch_size=pedl_batch_size, prebias=pedl_prebias, accumulate=pedl_accumulate
        )

        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=[round(opt.epochs * x) for x in [0.8, 0.9]],
            gamma=0.1,
            index=2,
            early_value=0.1,
        )

        return LRScheduler(scheduler, step_every_epoch=True)

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in dict(model.named_parameters()).items():
            if ".bias" in k:
                pg2 += [v]  # biases
            elif "Conv2d.weight" in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else

        hyp = get_hyp(lr0=pedl_init_lr)

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

        # opt = get_cli_args(
        #     batch_size=pedl_batch_size, prebias=pedl_prebias, accumulate=pedl_accumulate
        # )  # This seems to impact performance => replacing it with just the values
        (imgs, targets, paths, _) = batch

        imgs = imgs.float() / 255.0

        pred = model(imgs)
        loss, loss_items = compute_loss(pred, targets, model, not pedl_prebias)

        loss *= opt.batch_size / (pedl_batch_size * pedl_accumulate)

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss_items)

        return {"loss": loss}

    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader, model: nn.Module
    ) -> Dict[str, Any]:

        opt = get_cli_args(batch_size=pedl_batch_size)
        _, img_size_test = (
            opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2
        )  # train, test sizes

        results, maps = test.test(
            opt.cfg,
            opt.data,
            batch_size=opt.batch_size * 2,
            img_size=img_size_test,
            model=model,
            conf_thres=0.1,  # 0.1 for speed
            # 1e-3 if opt.evolve or (final_epoch and is_coco) else 0.1,
            iou_thres=0.6,
            single_cls=opt.single_cls,
            dataloader=data_loader,
        )
        keys = ["P", "R", "mAP_at_0.5", "F1"]

        return dict(zip(keys, results))


def get_cli_args(**new_args):
    """Returns the default command line arguments and the hyperparameters from the reference implementation

    Returns:
        Tuple[argparse.Namespace, dict] --
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=273
    )  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument(
        "--batch-size", type=int, default=16
    )  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument(
        "--accumulate",
        type=int,
        default=4,
        help="batches to accumulate before optimizing",
    )
    parser.add_argument(
        "--cfg", type=str, default="cfg/yolov3-spp.cfg", help="*.cfg path"
    )
    parser.add_argument(
        "--data", type=str, default="data/coco2017.data", help="*.data path"
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="adjust (67% - 150%) img_size every 10 batches",
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[416],
        help="train and test image-sizes",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume", action="store_true", help="resume training from last.pt"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/ultralytics68.pt",
        help="initial weights",
    )
    parser.add_argument(
        "--arc", type=str, default="default", help="yolo architecture"
    )  # default, uCE, uBCE
    parser.add_argument(
        "--name", default="", help="renames results.txt to results_name.txt if supplied"
    )
    parser.add_argument("--device", default="", help="device id (i.e. 0 or 0,1 or cpu)")
    parser.add_argument("--adam", action="store_true", help="use adam optimizer")
    parser.add_argument(
        "--single-cls", action="store_true", help="train as single-class dataset"
    )
    parser.add_argument("--var", type=float, help="debug variable")
    opt = parser.parse_args()

    opt.__dict__.update(new_args)
    return opt


def get_hyp(**new_hpars):
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
    }

    hyp.update(new_hpars)
    return hyp


def get_data_cfg():
    data_dict = {
        "classes": "80",
        "train": "data/coco/train2017.txt",
        "valid": "data/coco/val2017.txt",
        "names": "data/coco.names",
    }
    return data_dict


if __name__ == "__main__":

    # We turn some default values for cli args and hyperparameters into PEDL constants.

    pedl_prebias = pedl.Constant(value=True, name="prebias")
    pedl_batch_size = pedl.Constant(value=16, name="batch_size")
    pedl_accumulate = pedl.Constant(value=4, name="accumulate")
    pedl_init_lr = pedl.Constant(value=0.000579, name="init_lr")

    opt = get_cli_args(
        batch_size=pedl_batch_size, prebias=pedl_prebias, accumulate=pedl_accumulate
    )

    total_images = 117263  # number of coco images, TODO: replace by size of train set
    config = {
        "description": "yolov3_pytorch",
        "searcher": {
            "name": "single",
            "metric": "mAP_at_0.5",
            "max_steps": int(opt.epochs * total_images / opt.batch_size / 100),
            "smaller_is_better": False,
        },
        "bind_mounts": [
            {"host_path": "/home/anton/yolov3/data", "container_path": "data"}
        ],
        "optimizations": {"aggregation_frequency": opt.accumulate},
        "min_validation_period": int(
            total_images / opt.batch_size / 100 / 4
        ),  # validate after each 1/4 epoch
        "resources": {"slots_per_trial": 1},
    }

    exp = pedl.Experiment(
        context_directory=str(pathlib.Path.cwd()),
        trial_def=YOLOv3Trial,
        make_data_loaders_fn=make_data_loaders,
        configuration=config,
    )
    exp.create()
