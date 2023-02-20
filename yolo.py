import torch
import torch.nn as nn
from utils import *

class YoloLayer(nn.Module):
    def __init__(self, anchors, args):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = args.num_classes
        self.ignore_thred = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = args.img_size
        self.grid_size = 0
        self.cuda = args.cuda

    def compute_grid_offset(self, grid_size):
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_dim / self.grid_size
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for
                                           a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, pred,  target=None):
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        num_samples = pred.size(0)
        grid_size = pred.size(2)
        prediction = (pred.view(num_samples, self.num_anchors, self.num_classes + 5,
                                grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous())
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        if self.grid_size != grid_size:
            self.compute_grid_offset(grid_size)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat([pred_boxes.view(num_samples, -1, 4) * self.stride,
                            pred_conf.view(num_samples, -1, 1),
                            pred_cls.view(num_samples, -1, self.num_classes)], -1)

        if target == None:
            return output, 0

        iou_scores, class_mask, obj_mask, noobj_mask,\
        tx, ty, tw, th, tcls, tconf = build_target(pred_boxes=pred_boxes,
                                                   pred_cls=pred_cls,
                                                   target=target,
                                                   anchors=self.scaled_anchors,
                                                   ignore_thred=self.ignore_thred)

        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()     # class_mask/obj_mask(b, 3, 13, 13) # 正确率
        conf_obj = pred_conf[obj_mask].mean()           # 有物体的平均置信度
        conf_noobj = pred_conf[noobj_mask].mean()       # 无物体的平均置信度
        conf50 = (pred_conf > 0.5).float()              # 置信度大于0.5的位置 (b, num_anchor, 13, 13)
        iou50 = (iou_scores > 0.5).float()              # iou大于0.5的位置 (b, num_anchor, 13, 13)
        iou75 = (iou_scores > 0.75).float()             # iou大于0.75的位置 (b, num_anchor, 13, 13)
        detected_mask = conf50 * class_mask * tconf     # tconf=obj_mask, 即：既是预测的置信度>0.5，又class也对，又是obj
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            "loss": total_loss.item(),
            "x": loss_x.item(),
            "y": loss_y.item(),
            "w": loss_w.item(),
            "h": loss_h.item(),
            "conf": loss_conf.item(),
            "cls": loss_cls.item(),
            "cls_acc": cls_acc.item(),
            "recall50": recall50.item(),
            "recall75": recall75.item(),
            "precision": precision.item(),
            "conf_obj": conf_obj.item(),
            "conf_noobj": conf_noobj.item(),
            "grid_size": grid_size,
        }

        return output, total_loss
