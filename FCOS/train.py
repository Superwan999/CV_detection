from __future__ import division

import os
import argparse
import time
from copy import deepcopy

import torch
import torch.distributed as dist
from utils.misc import *
from fcos import *
from build import *
from utils.optimizer import build_optimizer
from utils.warmup import build_warmup
from utils.com_flops_params import flops_and_params
from criterion import build_criterion
from config import build_config


def parse_args():
    parser = argparse.ArgumentParser(description='FCOS Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('-bs', '--batch_size', default=16, type=int,
                        help='Batch size on single GPU for training')
    parser.add_argument('--schedule', type=str, default='1x',
                        help='training schedule: 1x, 2x, 3x, ...')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', default=1, type=int,
                        help='interval between evaluations')
    parser.add_argument('--grad_clip_norm', default=-1., type=float,
                        help='grad clip.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='path to save weight')
    parser.add_argument('--vis', dest="vis", action="store_true", default=False,
                        help="visualize input data.")

    # model
    parser.add_argument('-v', '--version', default='fcos',
                        help='build object detector')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')

    # train trick
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False,
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments...:", args)
    print("---------------------------------------------------")
    cfg = build_config(args)

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)

    # cuda
    if args.cuda:
        print("use cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset, evaluator, num_classes = build_face_dataset(cfg, args, device)

    # dataloader
    dataloader = build_dataloader(dataset, args.batch_size, CollateFunc())

    # build model and criterion
    model = FCOS(cfg,
                 device,
                 num_classes=num_classes,
                 conf_thresh=cfg['conf_thresh'],
                 nms_thresh=cfg['nms_thresh'],
                 trainable=False,
                 topk=args.topk)
    criterion = build_criterion(cfg, device, num_classes)

    # optimizer
    base_lr = cfg['base_lr'] * args.batch_size
    backbone_lr = base_lr * cfg['bk_lr_ratio']
    optimizer = build_optimizer(cfg, model, base_lr, backbone_lr)

    # lr scheduler
    lr_epoch = cfg['epoch'][args.schedule]['lr_epoch']
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_epoch)

    # warmup scheduler
    warmup_scheduler = build_warmup(cfg, base_lr)

    # training configuration
    max_epoch = cfg['epoch'][args.scheduler]['max_epoch']
    epoch_size = len(dataloader)
    best_map = -1.
    warmup = not args.no_warmp

    # compute FLOPS and Params
    model_ = deepcopy(model)
    flops_and_params(model=model_,
                     min_size=cfg['test_min_size'],
                     max_size=cfg['test_max_size'],
                     device=device)
    del model_

    t0 = time.time()

    # start training loop
    for epoch in range(max_epoch):

        # train one epoch
        for iter_i, (images, targets, masks) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size

            # warmup
            if ni < cfg['wp_iter'] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == cfg['wp_iter'] and warmup:
                # warmup is over
                print('warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, base_lr, base_lr)

            # to device
            images = images.to(device)
            masks = masks.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # visualize input data
            if args.vis:
                vis_data(images, targets, masks)
                continue

            # inference
            outputs = model(images, mask=masks)

            # compute loss
            loss_dict = criterion(outputs, targets)
            losses = loss_dict['total_loss']

            # check loss
            if torch.isnan(losses):
                print("loss is NAN !")
                continue

            # backward and optimize
            losses.backward()
            if args.grad_clip_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm)
            else:
                total_norm = get_total_grad_norm(model.parameters())

            optimizer.step()
            optimizer.zero_grad()

            # display
            if iter_i % 20 == 0:
                t1 = time.time()

                cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
                cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1]}
                # basic information
                log = '[Epoch: {} / {}]'.format(epoch + 1, max_epoch)
                log += '[Iter: {} / {}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}][lr_bk: {:.6f}]'.format(cur_lr_dict['lr'], cur_lr_dict['lr_bk'])

                # loss information
                for k in loss_dict.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

                # other information
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[g-norm: {:.2f}]'.format(total_norm)
                log += '[size: [{}, {}]]'.format(cfg['train_min_size'], cfg['train_max_size'])

                # print log information
                print(log)
                t0 = time.time()
        lr_scheduler.step()

        # evaluation
        if epoch % args.eval_epoch == 0 or (1 + epoch) == max_epoch:
            # check evaluator
            if evaluator is None:
                print('No evaluator...save model and go on training.')
                print('Saving state, epoch: {}'.format(epoch + 1))
                weight_name = 'fcos_epoch_{}.pth'.format(epoch + 1)
                checkpiont_path = os.path.join(path_to_save, weight_name)
                torch.save({'model': model.state_dict()}, checkpiont_path)

            else:
                print('eval ... ')
                model_eval = model

                # set eval mode
                model_eval.trainable = False
                model_eval.eval()

                # evaluate
                evaluator.evaluate(model_eval)
                cur_map = evaluator.map
                if cur_map > best_map:
                    best_map = cur_map

                    # save model
                    print(f"Saving state, epoch: {epoch + 1}")
                    weight_name = f"fcos_epoch_{epoch + 1}_{best_map * 100:.2f}.pth"
                    checkpiont_path = os.path.join(path_to_save, weight_name)
                    torch.save({'model': model.state_dict()}, checkpiont_path)

                # set train mode
                model_eval.trainable = True
                model_eval.train()

        # close mosaic augmentation
        if cfg['mosaic'] and max_epoch - epoch == 5:
            print("close Mosaic Augmentation")
            dataloader.dataset.mosaic = False


if __name__ == "__main__":
    train()
