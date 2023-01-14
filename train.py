import torch.nn as nn
import torch
from torch.utils.data import dataloader
from config import Config
from model import DarkNet
from data_loader import LoadDataSet



class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = DarkNet()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_data = LoadDataSet.trainData(self.args.train_path)
        self.val_data = LoadDataSet.valData(self.args.train_path)

    def train(self, epoch_index):
        self.model.train()
        total_loss = 0
        for i, batch in enumerate(self.train_data):
            imgs, targets = batch[0], batch[1]
            preds, losses = self.model(imgs)
            total_loss += losses.item()
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

