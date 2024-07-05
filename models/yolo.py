# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *  # noqa
from models.experimental import *  # noqa
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv3 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """
        Initializes the YOLOv3 detection layer with the number of classes, anchors, channel dimensions, and settings.

        Args:
            nc (int): Number of detection classes. Defaults to 80.
            anchors (tuple): A tuple containing anchor box dimensions. Each anchor should be defined by a tuple
                of (width, height) pairs. Defaults to an empty tuple.
            ch (tuple): A tuple specifying the number of channels for each detection layer. Each entry corresponds
                to the input channel dimension for a detection layer. Defaults to an empty tuple.
            inplace (bool): If True, allows inplace operations to save memory. Defaults to True.

        Returns:
            None

        Notes:
            This function constructs the necessary layers and initializes parameters for object detection using YOLOv3
            architecture. It registers the anchors as a buffer, initializes device-specific grids, and creates
            convolutional layers for transforming feature maps into detection outputs.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """
        Processes input through convolutional layers, reshaping output for detection.

        Args:
            x (list[torch.Tensor]): A list of input tensors with shape (bs, C, H, W) from different detection layers.

        Returns:
            torch.Tensor: A tensor containing the processed output suitable for object detection, with shape depending
            on whether it is in training or inference mode. In training mode, it returns a list of reshaped tensors.
            In inference mode, it returns a concatenated tensor across all detection layers.

        Note:
            The function adapts its behavior based on the `self.training` attribute. During inference, it calculates
            the grid necessary for transforming bounding box coordinates.

        Example:
            ```python
            model = Detect(nc=80, anchors=[(10, 13), (16, 30)], ch=[1024, 512, 256])
            inputs = [torch.randn(1, 1024, 20, 20), torch.randn(1, 512, 40, 40), torch.randn(1, 256, 80, 80)]
            output = model.forward(inputs)
            ```
        """
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """
        Creates and returns a grid and corresponding anchor grid for the specified layer, used for indexing anchors.

        Args:
            nx (int): Number of grid columns. Default is 20.
            ny (int): Number of grid rows. Default is 20.
            i (int): Index of the detection layer. Default is 0.
            torch_1_10 (bool): Flag to check compatibility with torch>=1.10.0. Default checks the current torch version.

        Returns:
            tuple: Contains the following:
                - grid (torch.Tensor): A tensor with shape (1, num_anchors, ny, nx, 2) representing the grid.
                - anchor_grid (torch.Tensor): A tensor with shape (1, num_anchors, ny, nx, 2) representing the anchor grid.

        Example:
            ```python
            # Suppose 'detect' is an instance of Detect class
            grid, anchor_grid = detect._make_grid(20, 20, 0)
            ```

        Notes:
            The generated grid and anchor_grid are used to adjust coordinates during the YOLO detection process.
            The 'torch_1_10' flag ensures compatibility with different versions of torch, specifically torch>=1.10.0.
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv3 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """
        Initializes the YOLOv3 segment head with customizable class count, anchors, masks, protos, channels, and inplace
        option.

        Args:
            nc (int, optional): Number of classes. Defaults to 80.
            anchors (tuple, optional): Tuple of anchor boxes. Defaults to empty tuple.
            nm (int, optional): Number of masks. Defaults to 32.
            npr (int, optional): Number of protos. Defaults to 256.
            ch (tuple, optional): Tuple of input channels. Defaults to empty tuple.
            inplace (bool, optional): Use in-place operations for layers. Defaults to True.

        Returns:
            None
        """
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """
        Processes input data through segmentation head layers and returns predictions and prototype masks.

        Args:
            x (list[Tensor]): List of input feature maps from the backbone convolutional neural network. Each element in
                              the list is a Tensor of shape (batch_size, channels, height, width).

        Returns:
            tuple[Tensor] | list[Tensor]: If in training mode, returns list of processed feature maps. If in export
                                          mode, returns tuple containing concatenated predictions and segmented masks.
                                          Otherwise, returns tuple of concatenated predictions and list of processed feature maps.

        Note:
            The function behaves differently depending on the state of `self.training` and `self.export`. For inference,
            modifications are applied to coordinates and confidence scores, and results are concatenated.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv3 base model
    def forward(self, x, profile=False, visualize=False):
        """
        Performs a forward pass through the YOLOv3 model for inference or training, with optional profiling and
        visualization.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
            profile (bool, optional): If True, profiles the model's layer-wise execution time. Default is False.
            visualize (bool, optional): If True, generates feature visualizations for the input tensor. Default is False.

        Returns:
            torch.Tensor | tuple: If in training mode, returns the network output tensor. If in export mode, returns a tuple
            containing the concatenated inference results. If in default inference mode, returns a tuple containing the
            concatenated inference results and intermediate feature maps.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """
        Executes a single forward pass through the model, supporting layer profiling and feature visualization.

        Args:
            x (torch.Tensor | list[torch.Tensor]): Input tensor or a list of tensors to be passed through the model.
            profile (bool, optional): Flag for profiling individual layers during the forward pass. Default is False.
            visualize (bool | str, optional): Flag or directory path for visualizing feature maps. If True or a valid path is
            provided, visualizations are saved. Default is False.

        Returns:
            torch.Tensor: Output tensor from the model's last layer.

        Notes:
            Profiling helps in understanding the time taken by each layer during the forward pass and can be enabled by setting
            `profile=True`.
            Visualization of feature maps from each layer can aid in debugging and interpretability, and it is controlled by
            the `visualize` flag.
            If `visualize` is set to a directory path, feature visualizations will be saved to that directory.

        Example:
            ```python
            model = BaseModel()
            input_tensor = torch.randn(1, 3, 640, 640)
            output = model._forward_once(input_tensor, profile=True, visualize='path/to/save')
            ```
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        Profiles a single layer of the model by measuring its execution time and computational cost.

        Args:
            m (nn.Module): The model layer to be profiled.
            x (torch.Tensor): Input tensor to the model layer.
            dt (list): List to which the execution time of the layer will be appended.

        Returns:
            None: This function appends the execution time results to `dt` directly and logs profiling information.

        Notes:
            - This function uses the `thop` library to calculate FLOPs. If `thop` is not installed, FLOPs calculation is skipped.
            - The profiling is performed 10 times to capture a more reliable average execution time.
            - Logs the execution time, GFLOPs, and number of parameters for the layer.

        Example:
            ```python
            model = BaseModel()
            input_tensor = torch.randn(1, 3, 512, 512)
            dt = []
            model._profile_one_layer(model.model[0], input_tensor, dt)
            ```
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """
        Fuses Conv2d() and BatchNorm2d() layers in the model to optimize inference speed.

        Fuses adjacent Conv2d and BatchNorm2d layers in all modules in the model to reduce memory usage and improve inference speed by combining two operations into one.

        Returns:
            None: This method modifies the model in place and does not return any value.

        Notes:
            This optimization combines convolution and batch normalization layers, reducing the overall computation by
            eliminating redundant operations. This is particularly beneficial for inference scenarios where speed is critical.

        Example:
            ```python
            model = ...  # Initialize model
            model.fuse()  # Fuse layers for optimized inference
            ```

        References:
            https://github.com/ultralytics/yolov3
            https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        """
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        """
        Provides detailed information about the model, including parameter count, layer details, and image size.

        Args:
            verbose (bool): If True, prints detailed layer information (default is False).
            img_size (int): Specifies the input image size for model profiling (default is 640).

        Returns:
            None

        Notes:
            - This function is useful for understanding the model architecture and debugging.

        Example:
            ```python
            model = BaseModel()
            model.info(verbose=True, img_size=640)
            ```
        """
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """
        _Apply a function to all model tensors, excluding parameters or registered buffers._

        Args:
            fn (Callable): Function to apply to each module.

        Returns:
            self (BaseModel): The model with the function applied to the appropriate tensors.

        Notes:
            This method overrides `nn.Module._apply` to handle specific YOLOv3 modules (`Detect`, `Segment`), ensuring
            their specific attributes like `stride`, `grid`, and `anchor_grid` are correctly processed. This is useful
            for operations such as moving the model to a different device or changing tensor data types (`to()`, `cpu()`,
            `cuda()`, `half()`).
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv3 detection model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
        Initializes the YOLOv3 detection model with configurable YAML config, input channels, number of classes, and
        anchors.

        Args:
            cfg (str | dict): Path to the YAML config file or a dictionary containing the model configuration.
            ch (int): Number of input channels. Default is 3.
            nc (int | None): Number of classes to detect. Overrides the YAML config if specified. Default is None.
            anchors (list | None): Custom anchor sizes. Overrides the YAML config if specified. Default is None.

        Returns:
            None

        Notes:
            The YAML config provides the model architecture details, including the number of input channels, detection layers, and other hyperparameters. If `nc` or `anchors` are provided, they override the corresponding values in the YAML config.

        Example:
            ```python
            from ultralytics.yolo import DetectionModel

            # Initialize using YAML config
            model = DetectionModel(cfg='yolov5s.yaml', ch=3, nc=80, anchors=[[10, 13, 16, 30, 33, 23], ...])

            # Initialize using a dictionary
            cfg_dict = {
                'nc': 80,
                'depth_multiple': 0.33,
                'width_multiple': 0.50,
                ...
            }
            model = DetectionModel(cfg=cfg_dict, ch=3)
            ```

        Attributes:
            yaml (dict): Loaded YAML config in dictionary form.
            yaml_file (str): Filename of the YAML config if loaded from a file.
            model (nn.Module): Constructed YOLO model based on the config.
            save (list): List of layers to save output from.
            names (list): Class names corresponding to indices.
            inplace (bool): Whether to use inplace operations.
            stride (torch.Tensor): Calculated strides for the detection layer.

        Raises:
            FileNotFoundError: If the YAML config file does not exist.
            KeyError: If essential keys are missing in the config dictionary.
        """
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Processes input data through the YOLOv3 detection model, generating predictions for object detection.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images with shape `(batch_size, channels, height, width)`.
            augment (bool): If True, performs augmented inference to enhance detection. Defaults to False.
            profile (bool): If True, profiles execution time and computational cost of layers. Defaults to False.
            visualize (bool): If True, visualizes intermediate feature maps. Defaults to False.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
                If the model is in training mode, returns output feature maps with shape
                `(batch_size, num_anchors, height/grid_size, width/grid_size, num_outputs)`.
                If the model is exporting, returns a tuple with concatenated predictions and the original feature maps.
                If the model is in inference mode, returns a tuple containing concatenated predictions across all strides
                with shape `(batch_size, num_detections, num_outputs)`.

        Examples:
            ```python
            model = DetectionModel('yolov5s.yaml')
            inputs = torch.randn(1, 3, 640, 640)
            outputs = model.forward(inputs)
            ```

            For more details on YOLOv3, refer to: https://github.com/ultralytics/ultralytics
        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """
        Performs augmented inference by scaling and flipping input images, returning concatenated predictions.

        Args:
          x (torch.Tensor): Input tensor with shape `(batch_size, channels, height, width)`.

        Returns:
          torch.Tensor: Concatenated predictions from augmented inferences.

        Notes:
          Augmentation involves three different scaling factors (1, 0.83, 0.67) and corresponding flip operations (None, flip
          left-right, None).

        Example:
          ```python
          model = DetectionModel(cfg="yolov5s.yaml")
          predictions = model._forward_augment(input_tensor)
          ```

        See Also:
          https://github.com/ultralytics/ultralytics for further details.
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """
        Descale predictions of augmented images by adjusting for scale and flips.

        Args:
            p (torch.Tensor): Predictions tensor with shape `(batch_size, num_anchors, 5 + num_classes)`.
            flips (int | None): Flip type applied during augmentation.
                `None` for no flip, `2` for vertical flip (up-down), `3` for horizontal flip (left-right).
            scale (float): Scaling factor used during augmentation.
            img_size (tuple[int, int]): Original image dimensions (height, width).

        Returns:
            torch.Tensor: Adjusted predictions tensor with corrections for scaling and flips applied during augmentation.
        """
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """
        Clip augmented inference tails from YOLOv3 predictions, focusing on the first and last detection layers.

        Args:
            y (list[torch.Tensor]): List of tensors containing YOLOv3 predictions.

        Returns:
            list[torch.Tensor]: Modified list of tensors with clipped first and last detection layers.

        Notes:
            This method helps to refine the predictions by removing unwanted augmented tails,
            which is essential for maintaining prediction accuracy after augmentation.

        Example:
            ```python
            model = DetectionModel(cfg='yolov5s.yaml')
            preds = model.forward(images, augment=True)
            clipped_preds = model._clip_augmented(preds)
            ```
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        """
        Initializes biases for the Detect() layer, adjusting objectness and class biases based on anchor strides.

        Args:
            cf (torch.Tensor | None): Class frequency tensor (default None). Adjusts class bias initialization if provided.

        Returns:
            None

        Notes:
            - The function sets the objectness bias using an approximation based on image size and average object count.
            - If frequency tensor `cf` is given, class biases are set proportionally; otherwise, they follow a uniform distribution.

        Example:
            ```python
            model = DetectionModel(cfg='yolov5s.yaml')
            model._initialize_biases(cf=torch.Tensor([10, 20, 30]))
            ```
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv3 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv3 segmentation model
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """
        Initializes a segmentation model based on the YOLOv3 architecture with customizable configuration, channels,
        number of classes, and anchors.

        Args:
            cfg (str | dict): Path to the YAML configuration file or dictionary containing model configuration.
            ch (int): Number of input channels. Default is 3 (RGB images).
            nc (int | None): Number of classes. If provided, will override the value set in the YAML configuration.
                Default is None.
            anchors (list[float] | None): Anchor values. If provided, will override the values set in the YAML
                configuration. Default is None.

        Returns:
            None

        Example:
            ```python
            model = SegmentationModel(cfg='yolov5s-seg.yaml', ch=3, nc=80, anchors=[10, 13, 16, 30, 33, 23])
            model_info(model)
            ```

        Notes:
            Ensure that the YAML configuration file format is consistent with the expected structure defined by the `parse_model`
            function.

        See Also:
            https://github.com/ultralytics/ultralytics for further details and usage examples of the Ultralytics library.
        """
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv3 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        """
        Initializes a ClassificationModel for YOLOv3 with an optional configuration YAML or model, number of classes,
        and cutoff index.

        Args:
            cfg (str | dict, optional): Path to the YAML configuration file or configuration dictionary. If not provided,
                a default configuration is used.
            model (DetectionModel | None, optional): Pretrained detection model to convert for classification purposes.
                If `None`, a new model will be created based on the `cfg`. Default is `None`.
            nc (int, optional): The number of classification categories. Default is 1000.
            cutoff (int, optional): Index up to which the detection model layers are retained for feature extraction.
                Default is 10.

        Returns:
            None: This constructor method initializes the architecture of the classification model.

        Notes:
            - If a detection model is passed, the backbone portion of the model is retained up to the cutoff.
            - The `cfg` parameter can override configuration keys in the dictionary or YAML file.
            - This method registers the layers and modules necessary for classification tasks.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """
        Initializes a classification model from a YOLOv3 detection model, configuring classes and cutoff.

        Args:
            model (nn.Module): A YOLOv3 detection model from which the classification model will be derived.
            nc (int): The number of classes for the classification model. Default is 1000.
            cutoff (int): The index layer up to which the detection model layers are used for the classification model.
                Default is 10.

        Returns:
            None

        Notes:
            This method modifies the input detection model's layers up to the specified `cutoff` to form the backbone
            of the classification model and appends a new classifier head. Ensure the input `model` is compatible with
            the YOLOv3 architecture.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """
        Creates a YOLOv3 classification model from a YAML file configuration.

        Args:
            cfg (str | dict): Path to the YAML configuration file or a dictionary containing the model configuration.

        Returns:
            None: This method initializes the model attributes based on the configuration but does not return any value.

        Raises:
            FileNotFoundError: If the provided YAML file path does not exist.

        Notes:
            The function reads a YAML configuration to initialize the attributes of the ClassificationModel. If a string path is
            provided, it opens the file and loads the content. If a dictionary is provided, it uses the dictionary directly. This
            allows flexibility in passing configurations either as a file path or directly as a dictionary. The function is commonly
            used during model initialization or reinitialization from pre-specified configurations.
        """
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    """
    Parses a YOLOv3 model configuration from a dictionary and constructs the model.

    Args:
        d (dict): A dictionary containing the model configuration, including the backbone and head definitions,
            anchors, number of classes (nc), depth multiple (gd), width multiple (gw), and activation function (act).
        ch (list[int]): A list of input channels, typically beginning with one integer (e.g., [3] for RGB images).

    Returns:
        tuple[list[nn.Module], list[int]]: A tuple where the first element is a list of PyTorch modules representing the
            model layers, and the second element is a list of integers indicating which layers' outputs should be saved.

    Notes:
        The function logs detailed information about each layer's configuration, including its index, from, number of
        parameters, module type, and arguments.

    Example:
        ```python
        model_dict = {
            'anchors': [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
            'nc': 80,
            'depth_multiple': 0.33,
            'width_multiple': 0.50,
            'backbone': [
                [-1, 1, 'Conv', [32, 3, 1]],
                [0, 1, 'C3', [64, 3]],
            ],
            'head': [
                [-1, 1, 'Conv', [128, 3, 1]],
                [1, 1, 'Detect', [5, 5, [2]]]
            ]
        }
        input_channels = [3]
        model_layers, save_list = parse_model(model_dict, input_channels)
        ```
    """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"], d.get("activation")
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
