import argparse

def load_config():
    parsers = argparse.ArgumentParser()
    parsers.add_argument('-timg',
                         '--train_img_path',
                         type=str,
                         default="./train_data/JPEGImages",
                         help='the path of images in train data')
    parsers.add_argument('-tla',
                        '--train_label_path',
                        type=str,
                        default="./train_data/labels",
                        help="the path of labels in train data")
    parsers.add_argument('-isize',
                         '--img_size',
                         type=int,
                         default=416,
                         help="the size of the input size for the detection model")
    parsers.add_argument('-bs',
                         '--batch_size',
                         type=int,
                         default=8,
                         help="batch size")
    parsers.add_argument('-l',
                         '--lr',
                         type=float,
                         default=0.0005,
                         help='initial learning rate')
    parsers.add_argument('--save_path',
                         type=str,
                         default='./model_save',
                         help="the path for saved model")
    parsers.add_argument('--valid_epoch',
                        type=int,
                        default=4,
                        help='check and save model every \"valid_epoch\" epochs')

    parsers.add_argument('--gradient_accumulations',
                         type=int,
                         default=2,
                         help="number of gradient accums before step")

    parsers.add_argument('--num_epochs',
                         type=int,
                         default=150,
                         help="the total training epoches")

    parsers.add_argument('--conf_thres',
                         type=float,
                         default=0.5,
                         help="the confidence threshold for no object")
    parsers.add_argument('--nms_thres',
                         type=float,
                         default=0.5,
                         help="the threshold for nms")
    parsers.add_argument('--iou_thres',
                         type=float,
                         default=0.5,
                         help="the threshold for iou")
    parsers.add_argument('--num_classes',
                         type=int,
                         default=20,
                         help="the number of classes")
    parsers.add_argument('--val_rate',
                         type=float,
                         default=0.1,
                         help="the validation data ratio of the total data")
    parsers.add_argument('--cuda',
                         type=bool,
                         default=False,
                         help="if the device has cuda")
    parsers.add_argument('--iter_log_step',
                         type=int,
                         default=50,
                         help="number of iterations to print training logs")
    args = parsers.parse_args()
    return args
