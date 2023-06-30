from matcher import *


class Criterion(object):
    def __init__(self, cfg, device, num_classes=1):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_reg_weight = cfg['loss_reg_weight']
        self.loss_ctn_weight = cfg['loss_ctn_weight']
        if cfg['match'] == 'matcher':
            print('============================')
            print('Matcher: FCOS Matcher')
            self.matcher = Matcher(cfg,
                                   num_classes=num_classes,
                                   box_weights=[1., 1., 1., 1.])

    def cls_loss(self, pred_cls, tgt_cls, num_boxes=1.0):
        """
        :param self:
        :param pred_cls: prediction classes, Tensor, [N, C]
        :param tgt_cls: target classes, Tensor, [N, C]
        :param num_boxes: box numbers
        :return: classes loss
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma, reduction='none')
        return loss_cls.sum() / num_boxes

    def bbox_loss(self, pred_delta, tgt_delta, bbox_quality=None, num_boxes=1.0):
        """
        :param pred_delta: Tensor, [N, 4]
        :param tgt_delta: Tensor, [N, 4]
        :param bbox_quality:
        :param num_boxes: box numbers
        :return: box loss
        """
        pred_delta = torch.cat((-pred_delta[..., :2], pred_delta[..., 2:]), dim=-1)
        tgt_delta = torch.cat((-tgt_delta[..., :2], tgt_delta[..., 2:]), dim=-1)
        eps = torch.finfo(torch.float32).eps
        pred_area = (pred_delta[..., 2] - pred_delta[..., 0]).clamp_(min=0) * \
                    (pred_delta[..., 3] - pred_delta[..., 1]).clamp_(min=0)
        tgt_delta = (tgt_delta[..., 2] - tgt_delta[..., 0]).clamp_(min=0) * \
                    (tgt_delta[..., 3] - tgt_delta[..., 1]).clamp_(min=0)
        w_intersect = (torch.min(pred_delta[..., 2], tgt_delta[..., 2]) -
                       torch.max(pred_delta[..., 0], tgt_delta[..., 0])).clamp_(min=0)
        h_intersect = (torch.min(pred_delta[..., 3], tgt_delta[..., 3]) -
                       torch.max(pred_delta[..., 1], tgt_delta[..., 1])).clamp_(min=0)
        area_intersect = w_intersect * h_intersect
        area_union = tgt_delta + pred_area - area_intersect
        ious = area_intersect / area_union.clamp(min=eps)

        # giou
        g_w_intersect = torch.max(pred_delta[..., 2], tgt_delta[..., 2]) - \
                        torch.min(pred_delta[..., 0], tgt_delta[..., 0])
        g_h_intersect = torch.max(pred_delta[..., 3], tgt_delta[..., 3]) - \
                        torch.min(pred_delta[..., 1], tgt_delta[..., 1])
        ac_union = g_w_intersect * g_h_intersect
        gious = ious - (ac_union - area_union) / ac_union.clamp(min=eps)
        loss_box = 1 - gious

        if bbox_quality is not None:
            loss_box = loss_box * bbox_quality.view(loss_box.size())

        return loss_box.sum() / num_boxes

    # the origin loss of FCOS
    def basic_losses(self, outputs, targets):
        """

        :param outputs: ['pred_cls] : Tensor, [B, M, C]
                       ['pred_reg']: Tensor, [B, M, 4]
                       ['pred_ctn']: Tensor, [B, M, 1]
                       ['stride']: list [8, 16, 32,...] stride of the model output
        :param targets: (list) [dict{'boxes': [...],
                                     'labels': [...],
                                     'orig_size': ...},...]
        :return: loss
        """

        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']

        # matcher
        (
            gt_classes,
            gt_shifts_deltas,
            gt_centerness
        ) = self.matcher(fpn_strides, anchors, targets)

        # list [B, M, C] -> [B, M, C] -> [BM, C]
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        pred_ctn = torch.cat(outputs['pred_ctn'], dim=1).view(-1, 1)
        masks = torch.cat(outputs['mask'], dim=1).view(-1)

        gt_classes = gt_classes.flatten().to(device)
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4).to(device)
        gt_centerness = gt_centerness.view(-1, 1).to(device)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()
        num_foreground = torch.clamp(num_foreground, min=1).item()

        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        num_targets = torch.clamp(num_foreground_centerness, min=1).item()

        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs. gt_classes[foreground_idxs]] = 1

        # cls loss
        valid_idxs = (gt_classes >= 0) & masks
        cls_loss = self.cls_loss(pred_cls[valid_idxs],
                                 gt_classes_target[valid_idxs],
                                 num_foreground)
        # bbox loss
        bbox_loss = self.bbox_loss(
            pred_delta[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            num_targets)

        # center-ness loss
        centerness_loss = F.binary_cross_entropy_with_logits(
            pred_ctn[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction='none')
        centerness_loss = centerness_loss.sum() / num_foreground

        # total loss
        total_loss = self.loss_cls_weight * cls_loss + \
                     self.loss_ctn_weight * centerness_loss + \
                     self.loss_reg_weight * bbox_loss
        loss_dict = dict(
            cls_loss=cls_loss,
            bbox_loss=bbox_loss,
            centerness_loss=centerness_loss,
            total_loss=total_loss
        )
        return loss_dict

    def __call__(self, outputs, targets):
        return self.basic_losses(outputs, targets)


# build criterion
def build_criterion(cfg, device, num_classes=1):
    criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)
    return criterion


if __name__ == "__main__":
    pass
