from pathlib import Path
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
import math
from general import LOGGER


def feature_visualization(x, module_type, stage, n=32,
                          save_dir=Path('./runs/detect/exp')):
    """

    :param x: Features to be visualized
    :param module_type:  Module type
    :param stage: Module stage within model
    :param n: Maximum number of feature maps to plot
    :param save_dir: Directory to save results
    :return:
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)
            n = min(n, channels)
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())
                ax[i].axis('off')

            LOGGER.info(f"Saving {f}... ({n} / {channels})")
            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())
