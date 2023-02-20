import datetime
import gc
import time

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from config import load_config
from model import MiniDarkNet
from data_loader import *
from logger import *
import tensorflow as tf
from terminaltables import AsciiTable
from data_loader import *
from utils import *
import shutil


class Trainer:
    def __init__(self, args, logger, anchors):
        self.args = args
        self.logger = logger
        self.anchors = anchors
        self.detection_dataSet = DetectionDataLoader(args)
        self.train_dataloader = DataLoader(self.detection_dataSet.train_data_loader,
                                           batch_size=self.args.batch_size,
                                           collate_fn=self.detection_dataSet.train_data_loader.collate_fn,
                                           shuffle=True)
        self.val_dataloader = DataLoader(self.detection_dataSet.val_data_loader,
                                         batch_size=self.args.batch_size,
                                         collate_fn=self.detection_dataSet.val_data_loader.collate_fn,
                                         shuffle=True)
        self.model = MiniDarkNet(self.args, self.anchors)
        self.yolo_layers = [self.model.yolo_layer0, self.model.yolo_layer1, self.model.yolo_layer2]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.5, 0.9))
        self.metrics = [
                        "loss",
                        "x",
                        "y",
                        "w",
                        "h",
                        "conf",
                        "cls",
                        "cls_acc",
                        "recall50",
                        "recall75",
                        "precision",
                        "conf_obj",
                        "conf_noobj",
                        "grid_size"
                        ]
        self.class_names = (    # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor')


    def train(self, epoch_index):
        self.model.train()
        epoch_loss = 0
        for iteration, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            batches_done = len(self.train_dataloader) * epoch_index + iteration
            imgs, targets, img_path = batch[0], batch[1], batch[2]
            preds, losses = self.model(imgs, targets)
            epoch_loss += losses.item()
            losses.backward()
            if batches_done % self.args.gradient_accumulations:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if iteration % self.args.iter_log_step == 0:
                log_str = f"\n ------ [epoch {epoch_index}/{self.args.num_epochs} iteration" \
                          f" {iteration} / {len(self.train_dataloader)}------\n ]"
                metrics_table = [["Metrics", *[f"YOLO Layer{i}" for i in range(3)]]]
                formats = {m: "%.4f" for m in self.metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                for i, metric in enumerate(self.metrics):
                    # print(formats)
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in self.yolo_layers]
                    metrics_table += [[metric, *row_metrics]]

                    tensorboard_log = []
                    for j, yolo in enumerate(self.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                    tensorboard_log += [("loss", losses.item())]
                    self.logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metrics_table).table
                log_str += f"\nTotal loss {losses.item()}"

                epoch_batches_left = len(self.train_dataloader) - (iteration + 1)
                time_end = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (iteration + 1))
                log_str += f"\n--- ETA {time_end}"
                print(log_str)
        train_epoch_logs = f"===> train: epoch: {epoch_index}, loss: {epoch_loss / len(self.train_dataloader)}"
        print(train_epoch_logs)



    def validate(self, epoch_index):
        labels = []
        sample_metrics = []
        self.model.eval()
        epoch_loss = 0
        val_len = len(self.val_dataloader)
        for iteration, batch in enumerate(self.val_dataloader):
            imgs, targets, img_path = batch[0], batch[1], batch[2]
            targets0 = targets
            labels += targets[:, 1].tolist()

            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= self.args.img_size
            imgs = torch.autograd.Variable(imgs.type(torch.FloatTensor), requires_grad=False)
            with torch.no_grad():
                outputs, val_losses = self.model(imgs, targets0)
                epoch_loss += val_losses.item() / val_len
                outputs = non_max_suppression(outputs,
                                              conf_thres=self.args.conf_thres,
                                              nms_thres=self.args.nms_thres)
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=self.args.iou_thres)
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        evaluation_metrics = [
                              ("val_precision", precision.mean()),
                              ("val_recall", recall.mean()),
                              ("val_mAP", AP.mean()),
                              ("val_f1", f1.mean())
                             ]
        self.logger.list_of_scalars_summary(evaluation_metrics, epoch_index)

        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, self.class_names[c], "%.4f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"----mAP {AP.mean()}")
        print(f"epoch {epoch_index} / {self.args.num_epochs}: val_losses {val_losses}")
        return epoch_loss

    def save_checkpoint(self, epoch_id, state, is_best):
        os.makedirs(self.args.save_path, exist_ok=True)
        filename = os.path.join(self.args.save_path, 'epoch_{}.pt'.format(epoch_id))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.args.save_path + '/' + 'epoch_{}'.format(epoch_id) + '_best.pt')
        print('Checkpoint saved to {}'.format(filename))

if __name__ == "__main__":
     args = load_config()
     logger = Logger('logs')
     anchors = [[[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]]

     if not os.path.exists(args.save_path):
         os.mkdir(args.save_path)

     cur_valid_loss = float("inf")
     is_best = False
     trainer = Trainer(args, logger, anchors)
     for epoch_index in range(1, args.num_epochs + 1):
         trainer.train(epoch_index)
         if epoch_index % args.valid_epoch == 0:
             valid_loss = trainer.validate(epoch_index)
             if valid_loss < cur_valid_loss:
                 cur_valid_loss = valid_loss
                 is_best = True

                 trainer.save_checkpoint(epoch_index,
                                         {
                                             "epoch": epoch_index,
                                             "start_dict": trainer.model.state_dict(),
                                             "optimizer": trainer.optimizer.state_dict()
                                         },
                                         is_best)
             gc.collect()

         # save last checkpoint
         trainer.save_checkpoint(args.num_epochs,
                                 {
                                     'epoch': args.num_epochs,
                                     'state_dict': trainer.model.state_dict(),
                                     'optimizer': trainer.optimizer.state_dict()
                                 },
                                 is_best)

