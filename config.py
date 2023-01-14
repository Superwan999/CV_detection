import argparse

def Config():
    parsers = argparse.ArgumentParser()
    parsers.add_argument('-tr',
                         '--train_path',
                         type=str,
                         default="./train_data",
                         help='the path of train data')
    parsers.add_argument('te',
                        '--test_path',
                        type=str,
                        default="./test_data",
                        help="the path of test data")
    args = parsers.parse_args()
    return args