import torch
import torch.nn as nn
from copy import deepcopy
import logging
import argparse
import sys
from common import *
from utils.general import make_divisible, check_file, set_logging
modules = [Conv, GhostConv, BottleNeck, SPP, SPPF, Focus, BottleNeckCSP,
                 C3, ShuffleBlock, conv_bn_relu_maxpool, DWConvBlock, MBConvBlock, LC3,
                 SEBlock, MobileV3Block, Hswish, SELayer, Stem, CBH, LCBlock, Dense,
                 GhostConv, ESBlottleNeck, ESSEModule]


def parse_model(cfg_dict, ch):
    """
    parse the model dict which saves the config parameters of the model, and build the torch model
    :param cfg_dict: config parameters of the model
    :param ch: numbers of channels, initialized with input channels numbers
    :return: torch model (type: nn.Module)
    """
    logger.info("\n%3s%18s%3s%10s  %-40s%-30s' %"
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
            if m in [BottleneckCSP, C3, C3TR]:
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
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, n_params, m_type, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c_2)
    return nn.Sequential(*layers), sorted(save)


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
            logger.info(f"Overriding model.yaml nc={self.cfg_dict['nc']} with nc={nc}")
            self.cfg_dict['nc'] = nc
        if anchors:
            logger.info(f"Overriding model.yaml anchors with anchors={anchors}")
        self.model, self.save = parser_model(deepcopy(self.cfg_dict), ch=[ch])
