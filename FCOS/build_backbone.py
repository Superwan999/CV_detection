from layers import *
import torch
import torch.nn as nn


class BackBone(nn.Module):
    def __init__(self, cfg):
        super(BackBone, self).__init__()
        self.layer_config = cfg
        self.backbone, self.b_out_pos = self.make_layers('backbone')

    def make_layers(self, module_name):
        module = self.layer_config[module_name]
        layers = []
        out_pos = []
        cnt = 0
        for i, (block, cfg, n, out) in enumerate(module):
            block = eval(block) if isinstance(block, str) else block
            cur_layer = [block(*cfg) for _ in range(n)]
            cnt += n

            if out > -1:
                out_pos.append(cnt - 1)

            layers += cur_layer
        return nn.ModuleList(layers), out_pos

    def forward(self, x):
        feat1, feat2 = x, x
        for i, block in enumerate(self.backbone):
            x = block(x)
            if i == self.b_out_pos[0]:
                feat1 = x
            if i == self.b_out_pos[1]:
                feat2 = x
        feat3 = x
        feats = [feat3, feat2, feat1]

        return feats, [feat.size(1) for feat in feats]


if __name__ == "__main__":
    cfg = {
    'backbone': [
                    ['CBS', [3, 16, 7, 2, 3], 1, -1],
                    ['conv_bn_relu_maxpool', [16, 16], 1, -1],
                    ['CBS', [16,  32, 3, 1, 1], 1, -1],
                    ['C3', [32,  32, 1, True, 1], 1, -1],
                    ['CBS', [32,  64, 3, 1, 1], 1, -1],
                    ['BottleneckCSP', [64, 64, 2, True, 1], 1, 0],

                    ['CBS', [64, 64, 3, 2, 1], 1, -1],
                    ['BottleneckCSP', [64, 64, 2, True, 1], 1, 1],

                    ['CBS', [64, 64, 3, 2, 1], 1, -1],
                    ['BottleneckCSP', [64, 64, 2, True, 1], 1, -1],

                    ['SPP', [64, 64, (5, 9, 13)], 1, 2]
    ]
    }
    backBone = BackBone(cfg)
    x = torch.randn(1, 3, 640, 640)
    feats, bk_dims = backBone(x)
    from torchstat import stat
    input_size = (3, 640, 640)
    stat(backBone, input_size)
    print([f" feat{i} shape: {feats[i].shape}" for i in range(3)])
