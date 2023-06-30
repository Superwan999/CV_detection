from torch import optim


def build_optimizer(cfg,
                    model,
                    base_lr=0.0,
                    backbone_lr=0.0):
    param_dict = [
        {
            'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
        },

        {
            'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad],
            'lr': backbone_lr,
        }
    ]

    if cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            params=param_dict,
            lr=base_lr,
            weight_decay=cfg['weight_decay']
        )
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            params=param_dict,
            lr=base_lr,
            weight_decay=cfg['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            params=param_dict,
            lr=base_lr,
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay']
        )

    return optimizer
