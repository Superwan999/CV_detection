import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import ToneDataLoader
from criterion import *
from model import HDRPointWiseNN
from hiunet import HINet
from config import load_config

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 schedule_lr: CosineAnnealingLR,
                 gpu_id: int,
                 args,
                 loss_fnc)->None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.schedule_lr = schedule_lr
        self.args = args
        self.loss_fnc = loss_fnc.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        if args.is_resume:
            if os.path.isfile(args.resume):
                print(f"===>loading checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint, strict=True)
            else:
                print(f"==> no checkpiont at: {args.resume}")
        
        #self.model = DDP(self.model, device_ids=[gpu_id])

    def _run_batch(self, low, full, target, epoch_id):
        self.optimizer.zero_grad()
        output = self.model(low, full)
        loss = self.loss_fnc(output, target, epoch_id)
        total_loss = 0
        if isinstance(loss, tuple):
            for l in loss:
                if l == 0:
                    continue
                total_loss += l
        else:
            total_loss = loss

        total_loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(name)
        self.optimizer.step()
        return loss


    def _run_epoch(self, epoch_id):
        pixel_loss = 0
        ssim_loss = 0
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch_id} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch_id)
        for lows, fulls, targets in self.train_data:
        # for fulls, targets in self.train_data:
            lows = lows.to(self.gpu_id)
            fulls = fulls.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(lows, fulls, targets, epoch_id)

            pixel_loss += loss[0].item() if loss[0] != 0 else 0
            ssim_loss += loss[1].item() if loss[1] != 0 else 0
        self.schedule_lr.step()

        print(f"==>[GPU]: {self.gpu_id} Epoch: {epoch_id} "
              f"lr:{self.optimizer.param_groups[-1]['lr']}"
              f"pixel_loss: {pixel_loss / len(self.train_data)},"
              f" ssim_loss: {ssim_loss / len(self.train_data)}")

    def _save_chekpoint(self, epoch_id):
        ckp = self.model.state_dict()
        os.makedirs(self.args.save_path, exist_ok=True)
        PATH = os.path.join(self.args.save_path, f"epoch_{epoch_id}.pth")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch_id} | Training checkpoint saved at {PATH}")

    def train(self):
        for epoch_id in range(1, self.args.epochs + 1):
            self._run_epoch(epoch_id)
            if self.gpu_id == 0:
                self._save_chekpoint(epoch_id)


def prepare_dataloader(dataset:Dataset, batch_size:int):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      pin_memory=True,
                      shuffle=False,
                      sampler=DistributedSampler(dataset))

def load_train_objs(args):
    train_set = ToneDataLoader(args)
    loss_fn = Loss(args)
    model = HDRPointWiseNN(params=args)
    # model = HINet()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    schedule_lr = CosineAnnealingLR(optimizer=optimizer, T_max=1000, eta_min=1e-7)
    return train_set, loss_fn, model, optimizer, schedule_lr

def main(rank:int, world_size:int, args):
    ddp_setup(rank, world_size)
    train_set,  loss_fnc, model, optimizer, schedule_lr = load_train_objs(args)
    train_data = prepare_dataloader(train_set, args.batch_size)
    trainer = Trainer(model,
                      train_data,
                      optimizer,
                      schedule_lr,
                      rank,
                      args,
                      loss_fnc)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = load_config()

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE)
