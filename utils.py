import torch
import numpy as np
from tqdm import tqdm


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def xywh2xyxy(box):
    y = box.new(box.shape)
    y[..., 0] = box[..., 0] - box[..., 2] / 2
    y[..., 1] = box[..., 1] - box[..., 3] / 2
    y[..., 2] = box[..., 0] + box[..., 2] / 2
    y[..., 3] = box[..., 1] + box[..., 3] / 2
    return y


def bbox_iou(box1, box2, x1y1x2y2=False):
    if not x1y1x2y2:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)
    b1_x1, b1_x2, b1_y1, b1_y2 = box1[:, 0], box1[:, 2], box1[:, 1], box1[:, 3]
    b2_x1, b2_x2, b2_y1, b2_y2 = box2[:, 0], box2[:, 2], box2[:, 1], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1,
                             min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    box1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    box2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-16)
    return iou


def wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh2[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w2, w1) * torch.min(h2 - h1)
    union_area = w1 * h1 + w2 * h2 + 1e-16 - inter_area
    iou = inter_area / union_area
    return iou


def build_target(pred_boxes, pred_cls, target, anchors, ignore_thred):
    """
    :param pred_boxes: 预测的bbox （b, num_anchors, grid_size, grid_size, 4)
    :param pred_cls: 预测类别概率(0, 1) (b, num_anchors, grid_size, grid_size, n_classes)
    :param target: (n_box6, 6) 6 ==> (batch_index, class_index, xc, yc,w, h) (scaled)
    :param anchors: (num_anchor, 2) 2==>(aw, ah)
    :param ignore_thred: hard code, 0.5
    :return: mask & t
    """
    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(4)
    nG = pred_boxes.size(2)

    #Output Tensor (batch的每张图， 每个grid，上的每个anchor会有个mask)
    obj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = torch.ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    th = torch.FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    #convert to position relative to target box
    target_boxes = target[:, 2:6] * nG  # 所有boxes, b_index, class_id, (tx, ty, tw, th) ==> nG 把尺度回到grid size 大小
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # 哪个anchor和target中box的iou最大，那么这个anchor就负责预测这个box中的物体
    # shape (n_boxes, len(anchors)), 每个anchor会和每个bbox计算一个iou
    ious = torch.stack([wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)  # 每个box，如box1，和哪个anchor的iou最大，那么这个anchor就负责预测这个box1

    b, target_labels = target[:, :2].long().t()  # n_boxes变成了列向量的个数，每一列就是一个box的b_index, class_id
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()  # .long是取整操作， 决定了box落在哪个cell上
    # print(f"gj:{gj}, gi: {gi}")
    # print(f"gxy: {gxy}")

    obj_mask[b, best_n, gj, gi] = 1  # 哪张图上的哪个anchor，落在哪个cell来预测物体
    noobj_mask[b, best_n, gj, gi] = 0  # 对应noobj_mask的哪些需要预测具体物体的cell，需要设置为0
    # set noobj_mask to zero where iou exceed ignore threshold

    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thred, gj[i], gi[i]] = 0

    # coordinate
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    tcls[b, best_n, gj, gi, target_labels] = 1

    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    tconf = obj_mask.float()  # target confidence
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
