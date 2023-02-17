import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from augment import *
from random import *


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class DetectionData(Dataset):
    def __init__(self, file_list, img_size):
        self.files = file_list
        self.img_size = img_size
        # self.transform = T.Compose([T.ToPILImage(), T.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index][0]
        label_path = self.files[index][1]
        img = np.asarray(Image.open(image_path))
        img_h, img_w, img_c = img.shape
        boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        img_tensor = T.ToTensor()(img)

        xmin = (boxes[:, 1] - boxes[:, 3] / 2) * img_w
        ymin = (boxes[:, 2] - boxes[:, 4] / 2) * img_h
        xmax = (boxes[:, 1] + boxes[:, 3] / 2) * img_w
        ymax = (boxes[:, 2] + boxes[:, 4] / 2) * img_h

        img_padded, pads = pad_to_square(img_tensor, 0)
        _, h_padded, w_padded = img_padded.shape

        xmin += pads[0]
        ymin += pads[2]
        xmax += pads[1]
        ymax += pads[3]

        boxes[:, 1] = ((xmin + xmax) / 2) / w_padded
        boxes[:, 2] = ((ymin + ymax) / 2) / h_padded
        boxes[:, 3] = (xmax - xmin) / w_padded
        boxes[:, 4] = (ymax - ymin) / h_padded
        target = torch.zeros((len(boxes), 6))
        target[:, 1:] = boxes
        return img_tensor, target, image_path

    def collate_fn(self, batch):
        imgs, targets, img_path = list(zip(*batch))
        targets = [target for target in targets if target is not None]
        for i, target in enumerate(targets):
            target[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        return imgs, targets, img_path



class DetectionDataLoader(Dataset):
    def __init__(self, args):
        name_lists = os.listdir(args.train_img_path)
        imgs_path = [os.path.join(args.train_img_path, name) for name in name_lists]
        labels_path = [os.path.join(args.train_label_path, name.replace('jpg', 'txt')) for name in name_lists]
        self.input_files = [(x, y) for x, y in zip(imgs_path, labels_path)]
        shuffle(self.input_files)
        self.train_data_loader = DetectionData(self.input_files, args.img_size)
        self.val_data_loader = DetectionData(sample(self.input_files, int(len(self.input_files) * 0.2)), args.img_size)
        if args.val_rate > 0:
            train_size = int(len(self.input_files) * (1 - args.val_rate))
            self.train_data_loader = DetectionData(self.input_files[:train_size], args.img_size)
            self.val_data_loader = DetectionData(self.input_files[train_size:], args.img_size)


if __name__ == "__main__":
    image_path = "./train_data/sample/JPEGImages/000005.jpg"
    label_path = "./train_data/sample/labels/000005.txt"
    img = np.asarray(Image.open(image_path))
    img_h, img_w, img_c = img.shape
    print(f"img_c {img_c}, img_h {img_h}, img_w {img_w}")
    boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
    print(f"boxes:{boxes}")
    img_tensor = T.ToTensor()(img)
    print(f"img_tensor.shape{img_tensor.shape}")
