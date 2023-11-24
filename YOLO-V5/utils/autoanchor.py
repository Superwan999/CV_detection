import random
import numpy as np
import torch
import yaml
from tqdm import tqdm

from general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr('AutoAnchor: ')

def check_anchor_order(m):
    # Check anchor order against stride order for model, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)
    delta_a = a[-1] - a[0]
    delta_s = m.stride[-1] - m.stride[0]
    if delta_a and (delta_a.sign() != delta_s.sign()):
        LOGGER.info(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flips(0)

def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit data, recompute if necessary
    m = model.module.mode[-1] if hasattr(model, 'module') else model.model[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1 / thr).float().sum(1).mean() # anchors above threshold
        bpr = (best > 1 / thr).float().mean() # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)
    anchors = m.anchors.clone() * stride # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f'{s}Current anchors are a good fit to dataset ✅')
    else:
        LOGGER.info(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...')

        na = m.anchors.numel() // 2 # number of anchors
        anchors = kmeans_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)
            m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(s)


def kmeans_anchors(dataset, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    Create kmeans-evolved anchors from training dataset
    :param dataset: path of dataset
    :param n: number of anchors
    :param img_size: image size used for training
    :param thr: anchor-label wh ratio threshold hyperparameter hyp['anchor-t'] used for training, default=4.0
    :param gen: generation to evolve anchors using genetic algorithm
    :param verbose: print all results
    :return: k, k-means evolved anchors
    """
    from scipy.cluster.vq import kmeans
    npr = np.random
    thr = 1 / thr

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0] # ratio metric
        return x, x.max(1)[0]

    def anchor_fitness(k, wh):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

    def print_results(k, wh, verbose=True):
        k = k[np.argsort(k.prod(1))] # sort small to large
        x, best = metric(k, wh)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        from dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shape / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)

    # Kmeans init
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)
        s = wh.std(0)
        k = kmeans(wh / s, n, iter=30)[0] * s
        assert n == len(k)
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size

    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, wh0, verbose=False)

    # Evolve
    f, sh, mp, s = anchor_fitness(k, wh), k.shape, 0.9, 0.1 # fitness, generation, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 0.3)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg, wh)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)
    return print_results(k).astype(np.float32)
