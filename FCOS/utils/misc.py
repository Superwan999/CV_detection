from torch.utils.data import DataLoader

from dataset.faceData import *
from dataset.transforms import *
from dataset.getdataset import *
from evaluators.face_evaluator import *

def vis_data(images, targets, masks):
    """
    :param images: tensor [B, 3, H, W]
    :param targets: list a list of targets
    :param masks: [B, H, W]
    :return:
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean = [123.675, 116.28, 103.53]
    rgb_std = [58.395, 57.12, 57.375]
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(91)]
    for bi in range(batch_size):
        # mask
        mask = masks[bi].bool()
        image_tensor = images[bi]
        index = torch.nonzero(mask)

        # pad image
        # to numpy
        pad_image = image_tensor.permute(1, 2, 0).cpu().numpy()

        # denormalize
        pad_image = (pad_image * rgb_std + rgb_mean).astype(np.uint8)

        # to BGR
        pad_image = pad_image[..., (2, 1, 0)]

        # valid image without pad
        valid_image = image_tensor[:, :index[-1, 0] + 1, :index[-1, 1] + 1]
        valid_image = valid_image.permute(1, 2, 0).cpu().numpy()
        valid_image = (valid_image * rgb_std + rgb_mean).astype(np.uint8)
        valid_image = valid_image[..., (2, 1, 0)]
        valid_image = valid_image.copy()
        targets_i = targets[bi]
        tgt_boxes = targets_i['boxes']
        tgt_labels = targets_i['labels']

        # to numpy
        mask = mask.cpu().numpy() * 255
        mask = mask.astype(np.uint8)

        for box, label in zip(tgt_boxes, tgt_labels):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            cls_id = int(labels)
            color = class_colors[cls_id]

            valid_image = cv2.rectangle(valid_image, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("pad image", pad_image)
        cv2.waitKey(0)

        cv2.imshow('valid image', valid_image)
        cv2.waitKey(0)

        cv2.imshow('mask', mask)
        cv2.waitKey(0)


def build_face_dataset(cfg, args, device):
    # dataset
    data_set, evaluator = build_dataset(cfg, args, device)
    num_classes = args.num_classes
    return data_set, evaluator, num_classes


def build_dataloader(dataset, batch_size, collate_fn=None):
    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler_train,
                             collate_fn=collate_fn, pin_memory=True)
    return data_loader


def nms(dets, scores, nms_thresh=0.4):
    # get xmin, ymin, xmax, ymax
    x1, y1, x2, y2 = [dets[:, i] for i in range(4)]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        over = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return keep


class CollateFunc(object):
    def _max_by_axis(self, list_):
        maxes = list_[0]
        for sublist in list_[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def __call__(self, batch):
        batch = list(zip(*batch))
        image_list = batch[0]
        target_list = batch[1]

        if image_list[0].ndim == 3:

            max_size = self._max_by_axis([list(img.shape) for img in image_list])
            batch_shape = [len(image_list)] + max_size
            b, c, h, w = batch_shape
            dtype = image_list[0].dtype
            device = image_list[0].device
            batch_tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            batch_mask = torch.zeros((b, h, w), dtype=dtype, device=device)

            for img, pad_img, m in zip(image_list, batch_tensor, batch_mask):
                pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = 1.0
        else:
            raise ValueError('not support image dim != 3')
        return batch_tensor, target_list, batch_mask


def sigmoid_focal_loss(logits,
                       targets,
                       alpha=0.25,
                       gamma=2.0,
                       reduction='none'):
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                 target=targets,
                                                 reduction=reduction)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device)
                                         for p in parameters]))
    return total_norm
