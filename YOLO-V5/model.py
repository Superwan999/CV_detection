import torch
import torch.nn as nn
from copy import deepcopy
import logging
import argparse
import sys
import math
from common import *
import thop
from utils.general import LOGGER, make_divisible, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

modules = [Conv, GhostConv, BottleNeck, SPP, SPPF, Focus, BottleNeckCSP,
                 C3, ShuffleBlock, conv_bn_relu_maxpool, DWConvBlock, MBConvBlock, LC3,
                 SEBlock, MobileV3Block, Hswish, SELayer, Stem, CBH, LCBlock, Dense,
                 GhostConv, ESBlottleNeck, ESSEModule]



class Detect(nn.Module):
    stride = None
    dynamic = False
    export  = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super(Detect, self).__init__()
        self.nc = nc
        self.n_out = nc + 5
        self.n_layer = len(anchors)
        self.n_anch = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.n_layer)]  #  initialize grid, yolo layer has an grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.n_layer)]
        self.register_buffer('anchor', torch.tensor(anchors).float().view(self.n_layer, -1, 2)) # shape (n_layer, n_anchor, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anch, 1) for x in ch) # output conv
        self.inplace = inplace

    def forward(self, x):
        z = [] # inference output
        for i in range(self.n_layer):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.n_anch, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.n_out - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else: # Detection
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.n_anch * nx * ny, self.n_out))
        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        dev = self.anchors[i].device
        typ = self.anchors[i].dtype
        shape = 1, self.n_anch, ny, nx, 2
        y, x = torch.arange(ny, device=dev, dtype=typ), torch.arange(nx, device=dev, dtype=typ)
        yv, xv = torch.meshgrid(y, x, indexing='ij')

        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.n_anch, 1, 1, 2).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])




def parse_model(cfg_dict, ch):
    """
    parse the model dict which saves the config parameters of the model, and build the torch model
    :param cfg_dict: config parameters of the model
    :param ch: numbers of channels, initialized with input channels numbers
    :return: torch model (type: nn.Module)
    """
    LOGGER.info("\n%3s%18s%3s%10s  %-40s%-30s' %"
                " ('', 'from', 'n', 'params', 'module', 'arguments')")
    anchors, nc, gd, gw = cfg_dict['anchors'], \
                          cfg_dict['nc'],\
                          cfg_dict['depth_multiple'],\
                          cfg_dict['width_multiple']
    n_anc = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors # number of anchors
    n_out = n_anc * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c_2 = [], [], ch[-1]  # layers, savelist, output channels

    for i, (f, n, m, args) in enumerate(cfg_dict['backbone'] + cfg_dict['head']): # from, numbers, module, args
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n # depth gain
        if m in modules:
            c_1, c_2 = ch[f], args[0]
            if c_2 != n_out:  # if not output
                c_2 = make_divisible(c_2 * gw, 8)
            args = [c_1, c_2, *args[1:]]
            if m in [BottleNeckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c_2 = sum(ch[x] for x in f)
        elif m is ADD:
            c_2 = sum(ch[x] for x in f) // 2
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int): # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c_2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c_2 = ch[f] // args[0] ** 2
        else:
            c_2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        m_type = str(m)[8:-2].replace('__main__', '') # model type
        n_params = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.n_params = i, f, m_type, n_params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, n_params, m_type, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c_2)
    return nn.Sequential(*layers), sorted(save)

class BaseModel(nn.Module):
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]
        o = thop.profile(m, inputs=(x.copy() if c else x, ),
                         verbose=False)[0] / 1e9 * 2
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConvBlock)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        """
        :param cfg: model parameters config, type: dict or yaml file
        :param ch: input channels
        :param nc: number of classes
        :param anchors: anchor
        """
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.cfg_dict = cfg
        else:
            import yaml
            self.yaml_file = cfg
            with open(cfg) as f:
                self.cfg_dict = yaml.load(f, Loader=yaml.SafeLoader) # load yaml file to dict

        # Define model
        # input channel
        ch = self.cfg_dict['chs'] = self.cfg_dict.get('ch', ch)
        if nc and nc != self.cfg_dict['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.cfg_dict['nc']} with nc={nc}")
            self.cfg_dict['nc'] = nc
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
        self.model, self.save = parse_model(deepcopy(self.cfg_dict), ch=[ch])
        self.names = [str(i) for i in range(self.cfg_dict['nc'])]
        self.inplace = self.cfg_dict.get('inplace', True)
         # build strides, anchors

        m = self.model[-1] # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchor /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()

        initialize_weight(self)
        self.info()
        LOGGER.info()

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x):
        img_size = x.shape[-2:] # height, width
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, 1), None

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat([x, y, wh, p[..., 4:]], -1)
        return p

    def _clip_augmented(self, y):
        n_layer = self.model[-1].n_layer
        g = sum(4 ** x for x in range(n_layer))
        e = 1
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))
        y[0] = y[0][:, :-1]
        i = (y[-1].shape[1] // g) * sum(4 ** (n_layer - 1 - x) for x in range(e))
        y[-1] = y[-1][:, i:]
        return y
