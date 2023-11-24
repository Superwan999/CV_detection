import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from augmentations import *
from tqdm import tqdm


from general import *
from torch_utils import *

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def img2label_path(img_paths):
    img_dir, label_dir = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return [label_dir.join(x.rsplit(img_dir, 1)).rsplit('.', 1)[0] + 'txt' for x in img_paths]

class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers
    uses same syntax as vanilla DataLoader
    """
    def __init__(self, *args, **kwargs):
        super(InfiniteDataLoader, self).__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)



class LoadImagesAndLabels(Dataset):
    # train_loader/valid_loader, loads images and labels for training and validation
    cache_version = 0.6
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix=''
                 ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p) as fp:
                        lines = fp.read().strip().splitlines()
                        fp.close()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in lines] # to global path
                else:
                    raise FileNotFoundError(f"{prefix}{p} does not exist")
            self.im_file = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert self.im_file, f'{prefix}No image found'
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}") from e

        # Check cache
        self.label_files = img2label_path(self.im_file) # labels
        # cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        self.indices = range(len(self))

    def __len__(self):
        return len(self.im_file)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, len(self) - 1)))
        else:
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        n_labels = len(labels)
        if n_labels:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        if self.augment:
            img, labels = self.albumentations(img, labels)
            n_labels = len(labels)

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.fliplr(img)
                if n_labels:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((n_labels, 6))
        if n_labels:
            labels_out[:, 1:] =  torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1] #HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.im_file[index], shapes


    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        f = self.im_file[i]
        im = cv2.imread(f)
        assert im is not None, f"Image Not Found {f}"
        h0, w0 = im.shape[:2] # orig hw
        r = self.img_size / max(h0, w0) # ratio
        if r != 1: # if sizes are not equal
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
        return im, (h0, w0), im.shape[:2]

    def load_mosaic(self, index):
        # Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        size = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * size + x)) for x in self.mosaic_border) # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        img4 = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
        for i, index in enumerate(indices):
            # load image
            img, _, (h, w) = self.load_image(index)
            # place img in img4
            if i == 0: # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h # xmin, ymin, xmax, ymax (small image)
            elif i == 1: # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2: # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else: # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, size * 2), min(size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh) # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * size, out=labels4[:, 1:])

        return img4, labels4

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)
        n = len(shapes) // 4
        img4, label4, path4, shape4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([0.0, 0, 0, 1, 0, 0])
        wo = torch.tensor([0.0, 0, 1, 0, 0, 0])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])
        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                img1 = F.interpolate(img[i].unsqueeze(0).float(),
                                     scale_factor=2.0, mode='bilinear',
                                     align_corners=False)[0].type(img[i].type())
                lb = label[i]
            else:
                img1 = torch.cat([torch.cat([img[i], img[i + 1]], dim=1),
                                  torch.cat([img[i + 2], img[i + 3]], dim=1)], dim=2)
                lb = torch.cat([label[i], label[i + 1] + ho,
                                label[i + 2] + wo, label[i + 3] + ho + wo], dim=0) * s

            img4.append(img1)
            label4.append(lb)
        for i, lb in enumerate(label4):
            lb[:, 0] = i
        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shape4













def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      seed=0):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,    # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix
        )

    batch_size = min(batch_size, len(dataset))
    n_cuda = torch.cuda.device_count()
    n_worker = min([os.cpu_count() // max(n_cuda, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader

    generator = torch.Generator()
    generator.manual_seed(10243443252432 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is not None,
                  num_workers=n_worker,
                  sampler=sampler,
                  pin_memory=PIN_MEMROY,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset
