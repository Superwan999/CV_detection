import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from build_backbone import BackBone
from build_neck import Neck
from build_head import FCOSHead


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """

        :param init_value: initial value for the scalar
        """
        super(Scale, self).__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        return x * self.scale


class FCOS(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 num_classes=1,
                 conf_thresh=0.05,
                 nms_thresh=0.6,
                 trainable=False,
                 topk=1000):
        super(FCOS, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.topk = topk

        self.backbone, bk_dim = BackBone(cfg=cfg)
        self.fpn = Neck(cfg, bk_dim)
        self.head = FCOSHead(cfg['head_dims'],
                             head_dim=64,
                             num_classes=1,
                             num_head=3)
        self.scales = nn.ModuleList([Scale() for _ in range(len(self.stride))])

    def forward(self, x, mask=None):
        if not self.trainable:
            return self.detection(x)
        else:
            feats = self.backbone(x)
            pyramid_feats = self.fpn(feats)

            all_cls_preds = []
            all_reg_preds = []
            all_ctn_preds = []
            all_masks = []

            for level, feat in enumerate(pyramid_feats):
                cls_pred, reg_pred, ctn_pred = self.head(feat)  # classification, regression, center-ness
                B, C, H, W = cls_pred.size()
                fmp_size = [H, W]

                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                reg_pred = F.relu(self.scales[level](reg_pred)) * self.stride[level]
                ctn_pred = ctn_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)

                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)
                all_ctn_preds.append(ctn_pred)

                if mask is not None:

                    mask_i = F.interpolate(mask[None], size=[H, W]).bool()[0]
                    mask_i = mask_i.flatten(1)
                    all_masks.append(mask_i)

            outputs = {
                "cls_pred": all_cls_preds,
                "reg_pred": all_reg_preds,
                "ctn_pred": all_ctn_preds,
                "masks": all_masks,
                "strides": self.stride
            }
            return outputs
