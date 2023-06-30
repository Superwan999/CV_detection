import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
from .transforms import *


CLASSES = ('face')

class FaceDataSet(data.Dataset):
    def __init__(self,
                 img_size=640,
                 input_path=None,
                 transform=None,
                 color_augment=None,
                 mosaic=False):
        self.input_path = input_path
        self.img_size = img_size
        self.transform = transform
        self.color_augment = color_augment
        self.mosaic = mosaic

        self.file_list = self.get_input_list(self.input_path)
        self.ids = [i for i in range(len(self.file_list))]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, target = self.pull_item(index)
        return image, target

    def load_image_target(self, index):
        # load an image
        image = cv2.imread(self.file_list[index][0])
        h, w = image.shape[:2]
        txt_ = np.loadtxt(self.file_list[index][1]).reshape(-1, 5)
        bboxes = txt_[:, 1:].copy()
        bboxes[:, 0] = bboxes[:, 0] - (bboxes[:, 2] / 2)
        bboxes[:, 1] = bboxes[:, 1] - (bboxes[:, 3] / 2)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        bboxes[:, 0] *= w
        bboxes[:, 1] *= h
        bboxes[:, 2] *= w
        bboxes[:, 3] *= h
        target = {
            "boxes": bboxes,
            "labels": txt_[:, 0]
        }
        return image, target

    def load_mosaic(self, index):
        ids_list = self.ids[:index] + self.ids[index + 1:]
        id1 = self.ids[index]
        id2, id3, id4 = random.sample(ids_list, 3)
        new_ids = [id1, id2, id3, id4]
        img_list = []
        tg_list = []

        for id_ in new_ids:
            img_i, target_i = self.load_image_target(id_)
            img_list.append(img_i)
            tg_list.append(target_i)

        mosaic_img, mosaic_target = mosaic_augment(img_list, tg_list, self.img_size)

        return mosaic_img, mosaic_target

    def pull_item(self, index):
        if self.mosaic and np.random.randint(2):

            image, target = self.load_mosaic(index)

            # augment
            image, target = self.color_augment(image, target)
        else:
            image, target = self.load_image_target(index)
            image, target = self.transform(image, target)
        return image, target

    def pull_image(self, index):
        img_path = self.file_list[index][0]
        return cv2.imread(img_path)

    @staticmethod
    def get_input_list(input_path):
        with open(input_path, 'r') as f:
            img_list = f.readlines()
            img_list = [item.strip('\n') for item in img_list]
            f.close()
        file_list = [(img_path, img_path.replace("images", "labels").
                      replace('jpg', 'txt')) for img_path in img_list]
        return file_list


if __name__ == "__main__":
    path_list = []
    input_path = r"D:\Personals\detection\pytorch-yolov3_CHANGE\data\custom\valid.txt"

    from transforms import TrainTransforms, ValTransforms, BaseTransforms

    format = 'RGB'
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]

    trans_config = [{'name': 'DistortTransform',
                     'hue': 0.1,
                     'saturation': 1.5,
                     'exposure': 1.5},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'RandomShift', 'max_shift': 32},
                    {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                    {'name': 'ToTensor'},
                    {'name': 'Resize'},
                    {'name': 'Normalize'}]
    min_size = 512
    max_size = 736
    random_size = [320, 512, 640]
    min_box_size = 4

    transform = TrainTransforms(
        trans_config=trans_config,
        min_size=min_size,
        max_size=max_size,
        random_size=random_size,
        min_box_size=min_box_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        format=format)

    color_augment = BaseTransforms(
        min_size=min_size,
        max_size=max_size,
        random_size=random_size,
        min_box_size=min_box_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        format=format)

    pixel_mean = np.array(pixel_mean, dtype=np.float32)
    pixel_std = np.array(pixel_std, dtype=np.float32)

    dataset = FaceDataSet(img_size=640,
                          input_path=input_path,
                          transform=transform,
                          color_augment=color_augment,
                          mosaic=True)

    np.random.seed(0)
    class_colors = [[255, 255, 0]]
    print('Data length: ', len(dataset))

    for i in range(10):
        image, target = dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to BGR format
        if format == 'RGB':
            # denormalize
            image = image * pixel_std + pixel_mean
            image = image[:, :, (2, 1, 0)].astype(np.uint8)
        elif format == 'BGR':
            image = image * pixel_std + pixel_mean
            image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            print(f"scaled bbox value: {x1}, {y1}, {x2}, {y2}")
            # x1, y1, x2, y2 = int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(f"bbox value: {x1}, {y1}, {x2}, {y2}")

            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            label = CLASSES[cls_id]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # put the text on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, (0, 0, 255), 1, 3)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
