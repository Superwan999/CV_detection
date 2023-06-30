from layers import *
import torch
import torch.nn as nn


class Neck(nn.Module):
    def __init__(self, cfg, in_dims):
        super(Neck, self).__init__()
        self.cfg = cfg
        self.in_dims = in_dims
        self.layers = self.make_layers()

    def make_layers(self):
        layers = []
        for layer_def in self.cfg['neck']:
            block = layer_def[0]
            block = eval(block) if isinstance(block, str) else block
            params = layer_def[1]
            # if isinstance(block, Concat):
            if params[0] == -1:
                in_dim = self.in_dims.pop(0)
                params[0] = in_dim

            if len(layer_def) > 2:
                num_r = layer_def[2]
                layers += [block(*params) for i in range(num_r)]
            else:
                layers.append(block(*params))
        return nn.ModuleList(layers)

    def forward(self, feats):
        x = feats.pop(0)
        output = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Concat):
                y = feats.pop(0)
                # print(f"y shape: {y.shape}, x shape: {x.shape}")
                x = layer(x, y)
                output.append(x)
            else:
                x = layer(x)
            if i == 0:
                output.append(x)
        return output


if __name__ == "__main__":
    feats = [torch.randn(1, 64, 20 * 2 ** i, 20 * 2 ** i) for i in range(3)]
    in_dims = [feat.size(1) for feat in feats]

    cfg = {
        "neck": [['CBS', [-1, 64, 3, 1, 1], 2, -1],
                 ['UpSample', [64, 64, 1, 1, 0], 1, -1],
                 ['Concat', [-1, 64, 64]],
                 ['C3', [64, 64, 3, True, 1], 2, -1],
                 ['CBS', [64, 64, 3, 1, 1], 1, -1],
                 ['UpSample', [64, 64, 1, 1, 0], 1, -1],
                 ['Concat', [-1, 64, 64]]
                 ]
    }

    neck = Neck(cfg, in_dims)
    output = neck(feats)
    # input_size = [(64, 20, 20), (64, 40, 40), (64, 80, 80)]
    # from torchstat import stat
    # stat(neck, input_size)
    print([f" feat{i} shape: {output[i].shape}" for i in range(3)])
