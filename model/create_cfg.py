import argparse
import re
from pathlib import Path


def create_cfg(args):
    cfg_file = Path(args.cfg)
    max_batch = args.b

    num_classes = 14

    # calculate the 2 steps values:
    step1 = 0.8 * max_batch
    step2 = 0.9 * max_batch

    num_filters = (num_classes + 5) * 3

    with open(cfg_file) as f:
        s = f.read()

    s = re.sub('max_batches = \d*', 'max_batches = '+str(max_batch), s)
    s = re.sub('steps=\d*,\d*', 'steps=' +
               "{:.0f}".format(step1)+','+"{:.0f}".format(step2), s)
    s = re.sub('classes=\d*', 'classes='+str(num_classes), s)
    s = re.sub('pad=1\nfilters=\d*', 'pad=1\nfilters=' +
               "{:.0f}".format(num_filters), s)

    with open(cfg_file, 'w') as f:
        f.write(s)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocessing Pipeline')
    parser.add_argument('-b', metavar='max batch', type=int, default=3000,
                        help='Max batch')
    parser.add_argument('-cfg', metavar='darknet config file', required=True, type=str,
                        help='darknet.data path')

    args = parser.parse_args()
    create_cfg(args)
