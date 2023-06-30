import torch
import torch.nn as nn
from layers import *


class FCOSHead(nn.Module):
    def __init__(self,
                 in_dims,
                 head_dim=64,
                 num_classes=1,
                 num_head=3):
        super(FCOSHead, self).__init__()
        if not isinstance(in_dims, list):
            in_dims = [head_dim for _ in range(num_head)]
        self.cls_feats = nn.Sequential(*[CBS(in_dims[i], head_dim, k=3, s=1, p=1)
                                         for i in range(num_head)])

        self.reg_feat = nn.Sequential(*[CBS(in_dims[i], head_dim, k=3, s=1, p=1)
                                         for i in range(num_head)])
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1, padding=0)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1, padding=0)
        self.ctn_pred = nn.Conv2d(head_dim, 1, kernel_size=1, padding=0)
        self._init_weight()

    def _init_weight(self):
        for m in [self.cls_feats, self.reg_feat]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # init cls pred
        nn.init.normal_(self.cls_pred.weight, mean=0, std=0.01)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
        # init reg pred
        nn.init.normal_(self.reg_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)
        # init ctn pred
        nn.init.normal_(self.ctn_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def forward(self, x):
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feat(x)
        cls_pred = self.cls_pred(cls_feats)
        reg_pred = self.reg_pred(reg_feats)
        ctn_pred = self.ctn_pred(reg_feats)
        return cls_pred, reg_pred, ctn_pred


def Head(params):
    head = FCOSHead(*params)
    return head
