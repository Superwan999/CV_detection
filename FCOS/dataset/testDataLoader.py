import numpy as np
from torch.utils.data import Dataset
import cv2


class TestDataSet(Dataset):
    def __init__(self, input_path=None):
        super(TestDataSet, self).__init__()

        self.file_list = self.get_input_list(input_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path, label_path = self.file_list[index]
        image = cv2.imread(img_path)
        txt_ = np.loadtxt(label_path).reshape(-1, 5)
        labels = txt_[:, 0]
        bboxes = txt_[:, 1:].copy()
        bboxes[:, 0] = bboxes[:, 0] - (bboxes[:, 2] / 2)
        bboxes[:, 1] = bboxes[:, 1] - (bboxes[:, 3] / 2)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        target = {
            "bboxes": bboxes,
            "labels": labels
        }
        return image, target



    @staticmethod
    def get_input_list(input_path):

        with open(input_path, 'r') as f:
            img_list = f.readlines()
            f.close()
        img_list = [img_path.strip('\n') for img_path in img_list]
        file_list = [(img_path, img_path.replace("images", "labels").
                      replace("jpg", "txt")) for img_path in img_list]
        return file_list


if __name__ == "__main__":
    input_path = r"D:\Personals\detection\pytorch-yolov3_CHANGE\data\custom\sample.txt"
    test_dataset = TestDataSet(input_path=input_path)
    image, target = test_dataset[0]

    labels = target["labels"]
    bboxes = target["bboxes"]
    # image = image.permute(1, 2, 0).numpy()
    h, w = image.shape[:2]
    for i, (box, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = box
        print(f"scaled box {i}: x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        print(f"box {i}: x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, (0, 0, 255), 1, 3)

    cv2.imshow('gt', image)
    cv2.waitKey(0)
