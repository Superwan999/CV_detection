config = {
    "train_min_size": 800,
    "train_max_size": 1333,
    "test_min_size": 800,
    "test_max_size": 1333,
    "format": 'RGB',
    "pixel_mean": [123.675, 116.28, 103.53],
    "pixel_std": [58.395, 57.12, 57.375],
    "min_box_size": 8,
    "mosaic": False,
    "transforms": [
        {'name': "RandomHorizontalFlip"},
        {'name': 'ToTensor'},
        {'name': 'Resize'},
        {'name': 'Normalize'}
    ],

    # ----------------- PostProcess --------------------
    'conf_thresh': 0.1,
    'nms_thresh': 0.5,
    'conf_thresh_val': 0.05,
    'nms_thresh_val': 0.6,

    # ----------------Label Assignment -----------------
    ## FCOS Matcher
    'matcher': "matcher",
    'center_sampling_radius': 1.5,
    'object_sizes_of_interest': [[-1, 64], [64, 128], [128, 256], [256, 512], [512, float('inf')]],

    # --------------- Loss Configuration -----------------
    ## loss hyper-parameters
    'alpha': 0.25,
    'gamma': 2.0,
    'loss_cls_weight': 1.0,
    'loss_reg_weight': 1.0,
    'loss_ctn_weight': 1.0,

    # --------------- Network Parameters ------------------
    'backbone': [
        ['CBS', [3, 16, 7, 2, 3], 1, -1],
        ['conv_bn_relu_maxpool', [16, 16], 1, -1],
        ['CBS', [16, 32, 3, 1, 1], 1, -1],
        ['C3', [32, 32, 1, True, 1], 1, -1],
        ['CBS', [32, 64, 3, 1, 1], 1, -1],
        ['BottleneckCSP', [64, 64, 2, True, 1], 1, 0],

        ['CBS', [64, 64, 3, 2, 1], 1, -1],
        ['BottleneckCSP', [64, 64, 2, True, 1], 1, 1],

        ['CBS', [64, 64, 3, 2, 1], 1, -1],
        ['BottleneckCSP', [64, 64, 2, True, 1], 1, -1],

        ['SPP', [64, 64, (5, 9, 13)], 1, 2]
    ],

    "neck": [['CBS', [-1, 64, 3, 1, 1], 2, -1],
             ['UpSample', [64, 64, 1, 1, 0], 1, -1],
             ['Concat', [-1, 64, 64]],
             ['C3', [64, 64, 3, True, 1], 2, -1],
             ['CBS', [64, 64, 3, 1, 1], 1, -1],
             ['UpSample', [64, 64, 1, 1, 0], 1, -1],
             ['Concat', [-1, 64, 64]]
             ],

    # --------------- HEAD ---------------------
    'head_dim': 256,
    'num_cls_heads': 4,
    'num_reg_heads': 4,
    # --------------- Training Configuration ------------------

    # optimizer
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'base_lr': 0.01 / 16.,
    'bk_lr_ratio': 1.0,

    ## Warmup
    'warmup': 'linear',
    'wp_iter': 1000,
    'warmup_factor': 0.00066667,

    # Epoch
    'epoch': {
        '1x': {
            'max_epoch': 12,
            'lr_epoch': [8, 11],
            'multi_scale': None
        },
        '2x': {
            'max_epoch': 24,
            'lr_epoch': [16, 22],
            'multi_scale': [640, 672, 704, 736, 768, 800]
        },
        '3x': {
            'max_epoch': 36,
            'lr_epoch': [24, 33],
            'multi_scale': [640, 672, 704, 736, 768, 800]
        }
    }
}


def build_config():
    return config
