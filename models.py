import os
from collections import defaultdict

import torch.nn as nn

from utils.parse_config import *
from utils.utils import *

ONNX_EXPORT = False


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, img_dim, anchor_idxs, cfg):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.bbox_attrs = 5 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        if cfg.endswith('yolov3-tiny.cfg'):
            stride *= 2

        # Build anchor grids
        nG = int(self.img_dim / stride)  # number grid points
        self.grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
        self.grid_y = torch.arange(nG).repeat((nG, 1)).t().view((1, 1, nG, nG)).float()
        self.anchor_wh = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])  # scale anchors
        self.anchor_w = self.anchor_wh[:, 0].view((1, nA, 1, 1))
        self.anchor_h = self.anchor_wh[:, 1].view((1, nA, 1, 1))
        self.weights = class_weights()

        self.loss_means = torch.ones(6)
        self.yolo_layer = anchor_idxs[0] / nA  # 2, 1, 0
        self.stride = stride
        self.nG = nG

        if ONNX_EXPORT:  # use fully populated and reshaped tensors
            self.anchor_w = self.anchor_w.repeat((1, 1, nG, nG)).view(1, -1, 1)
            self.anchor_h = self.anchor_h.repeat((1, 1, nG, nG)).view(1, -1, 1)
            self.grid_x = self.grid_x.repeat(1, nA, 1, 1).view(1, -1, 1)
            self.grid_y = self.grid_y.repeat(1, nA, 1, 1).view(1, -1, 1)
            self.grid_xy = torch.cat((self.grid_x, self.grid_y), 2)
            self.anchor_wh = torch.cat((self.anchor_w, self.anchor_h), 2) / nG

    def forward(self, p, targets=None, var=None):
        bs = 1 if ONNX_EXPORT else p.shape[0]  # batch size
        nG = self.nG  # number of grid points

        if p.is_cuda and not self.weights.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
            self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Training
        if targets is not None:
            MSELoss = nn.MSELoss()
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()

            # Get outputs
            x = torch.sigmoid(p[..., 0])  # Center x
            y = torch.sigmoid(p[..., 1])  # Center y
            p_conf = p[..., 4]  # Conf
            p_cls = p[..., 5:]  # Class

            # Width and height (yolo method)
            w = p[..., 2]  # Width
            h = p[..., 3]  # Height
            # width = torch.exp(w.data) * self.anchor_w
            # height = torch.exp(h.data) * self.anchor_h

            # Width and height (power method)
            # w = torch.sigmoid(p[..., 2])  # Width
            # h = torch.sigmoid(p[..., 3])  # Height
            # width = ((w.data * 2) ** 2) * self.anchor_w
            # height = ((h.data * 2) ** 2) * self.anchor_h

            tx, ty, tw, th, mask, tcls = build_targets(targets, self.anchor_wh, self.nA, self.nC, nG)

            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nB = len(targets)  # batch size
            k = nM / nB
            if nM > 0:
                lx = k * MSELoss(x[mask], tx[mask])
                ly = k * MSELoss(y[mask], ty[mask])
                lw = k * MSELoss(w[mask], tw[mask])
                lh = k * MSELoss(h[mask], th[mask])

                lcls = (k / 4) * CrossEntropyLoss(p_cls[mask], torch.argmax(tcls, 1))
                # lcls = (k * 10) * BCEWithLogitsLoss(p_cls[mask], tcls.float())
            else:
                FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

            lconf = (k * 64) * BCEWithLogitsLoss(p_conf, mask.float())

            # Sum loss components
            loss = lx + ly + lw + lh + lconf + lcls

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), nT

        else:
            if ONNX_EXPORT:
                p = p.view(-1, 85)
                xy = torch.sigmoid(p[:, 0:2]) + self.grid_xy[0]  # x, y
                wh = torch.exp(p[:, 2:4]) * self.anchor_wh[0]  # width, height
                p_conf = torch.sigmoid(p[:, 4:5])  # Conf
                p_cls = F.softmax(p[:, 5:85], 1) * p_conf  # SSD-like conf
                return torch.cat((xy / nG, wh, p_conf, p_cls), 1)

                # p = p.view(1, -1, 85)
                # xy = torch.sigmoid(p[..., 0:2]) + self.grid_xy  # x, y
                # wh = torch.exp(p[..., 2:4]) * self.anchor_wh  # width, height
                # p_conf = torch.sigmoid(p[..., 4:5])  # Conf
                # p_cls = p[..., 5:85]
                #
                # # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
                # # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
                # p_cls = torch.exp(p_cls).permute((2, 1, 0))
                # p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
                # p_cls = p_cls.permute(2, 1, 0)
                #
                # return torch.cat((xy / nG, wh, p_conf, p_cls), 2).squeeze().t()

            p[..., 0] = torch.sigmoid(p[..., 0]) + self.grid_x  # x
            p[..., 1] = torch.sigmoid(p[..., 1]) + self.grid_y  # y
            p[..., 2] = torch.exp(p[..., 2]) * self.anchor_w  # width
            p[..., 3] = torch.exp(p[..., 3]) * self.anchor_h  # height
            p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf
            p[..., :4] *= self.stride

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return p.view(bs, -1, 5 + self.nC)


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nT']
        self.losses = []

    def forward(self, x, targets=None, var=0):
        self.losses = defaultdict(float)
        is_training = targets is not None
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets, var)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nT'] /= 3

        if ONNX_EXPORT:
            output = torch.cat(output, 0)  # merge the 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            return output[:, 5:85], output[:, :4]  # ONNX scores, boxes

        return sum(output) if is_training else torch.cat(output, 1)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 16

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
