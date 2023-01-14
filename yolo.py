import torch
import torch.nn as nn
from model import DarkNet


class YoloLayer(nn.Module):
    def __init__(self, predicts, targets):
        super(YoloLayer, self).__init__()

