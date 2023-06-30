from evaluators.face_evaluator import *
from .faceData import FaceDataSet
from .transforms import *


def build_dataset(cfg, args, device):
    trans_config = cfg['transforms']
    print("==============================")
    print(f"TrainTransforms: {trans_config}")
    train_transform = TrainTransforms(
        trans_config=trans_config,
        min_size=cfg['train_min_size'],
        max_size=cfg['train_max_size'],
        random_size=cfg['epoch'][args.schedule]['multi_scale'],
        min_box_size=cfg['min_box_size'],
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        format=cfg['format']
    )
    val_transform = ValTransforms(
        min_size=cfg['test_min_size'],
        max_size=cfg['test_max_size'],
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        format=cfg['format']
        )
    color_augment = BaseTransforms(
        min_size=cfg['train_min_size'],
        max_size=cfg['train_max_size'],
        random_size=cfg['epoch'][args.schedule]['multi_scale'],
        min_box_size=cfg['min_box_size'],
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        format=cfg['format']
        )

    dataset = FaceDataSet(img_size=cfg['train_min_size'],
                          input_path=args.input_path,
                          transform=train_transform,
                          color_augment=color_augment,
                          mosaic=True)
    evaluator = FaceAPIEvaluator(input_path=args.input_path,
                                 output_path=args.output_path,
                                 device=device,
                                 transform=val_transform)
    return dataset, evaluator
