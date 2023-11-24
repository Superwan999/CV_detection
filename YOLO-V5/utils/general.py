import glob
import contextlib
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import sys
import time
import signal
import subprocess
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml

LOGGING_NAME = 'yolov5'
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

def set_logging(name=LOGGING_NAME, verbose=True):
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                name: {'format': '%(message)s'}
            },
            'handlers': {
                name: {
                'class': 'logging.StreamHandler',
                'format': name,
                'level': level,
                }
            },
            'loggers': {
                name: {
                    'level': level,
                    'handlers': [name],
                    'propagate': False
                }
            }
        }
    )

set_logging(LOGGING_NAME)
LOGGER = logging.getLogger(LOGGING_NAME)

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and
            all(x not in k for x in exclude) and v.shape == db[k].shape}

def yaml_save(file:Path, data:dict):
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()},
                       f, sort_keys=False)

def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


def check_suffix(file='yolov5.pt', suffix=('.pt', ), msg=''):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]

        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'

def check_img_size(imgsz, s=32, floor=0):

    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:
        imgsz = list(imgsz)
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f"WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}")
    return new_size

def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    # Check device
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f"{prefix} Computing optimal batch size for --imgsz {imgsz}")
    device = next(model.parameters()).device
    if device.type == 'cpu':
        LOGGER.info(f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f'{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}')
        return batch_size

def check_train_batch_size(model, imgsz=640, amp=True):
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)

def check_dataset(data, autodownload=True):
    extract_dir = ''
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        pass


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])

def xywhn2xyxy(in_box, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    out_box = in_box.clone() if isinstance(in_box, torch.Tensor) else np.copy(in_box)
    out_box[..., 0] = w * (in_box[..., 0] - in_box[..., 2] / 2) + padw # top left x
    out_box[..., 1] = h * (in_box[..., 1] - in_box[..., 3] / 2) + padh # top left y
    out_box[..., 2] = w * (in_box[..., 0] + in_box[..., 2] / 2) + padw # bottom right x
    out_box[..., 3] = h * (in_box[..., 1] + in_box[..., 3] / 2) + padh # bottom right y
    return out_box

def xyxy2xywhn(in_box, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(in_box, (h - eps, w - eps))
    out_box = in_box.clone() if isinstance(in_box, torch.Tensor) else np.copy(in_box)
    out_box[..., 0] = ((in_box[..., 0] + in_box[..., 2]) / 2) / w
    out_box[..., 1] = ((in_box[..., 1] + in_box[..., 3]) / 2) / h
    out_box[..., 2] = (in_box[..., 2] - in_box[..., 0]) / w
    out_box[..., 3] = (in_box[..., 3] - in_box[..., 1]) / h
    return out_box

def labels_to_class_weights(labels, nc=80):
    # Get class weights from training labels
    if labels[0] is None:
        return torch.Tensor()

    labels = np.concatenate(labels, 0) # labels.shape = (866643, 5)
    classes = labels[:, 0].astype(int) # labels = [class, xywh]
    weights = np.bincount(classes, minlength=nc) # occurrences per class

    weights[weights == 0] = 1 # replace empty bins with 1
    weights = 1 / weights # numbers of targets per class
    weights /= weights.sum()
    return torch.from_numpy(weights).float()
