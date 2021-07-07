import argparse
from util import create_transform
from functions import augment, create_darknet_data, create_yolo_cfg, resize_images, readImages, split, kfold
from Dataclasses import PipeConfig
from Pipeline import Pipeline


def main(args):
    config = PipeConfig(
        name=args.n,
        input_folder=args.i,
        output_folder=args.o,
        resized_img_size=600,
        final_img_size=416,
        number_of_augmentations=10,
        color=args.c,
        transform=create_transform(),
        classes_txt=args.cls,
        yolo_cfg=args.yolo_cfg,
        max_batch_size=args.batch_size,
        folds=args.f
    )
    # Initialize Pipeline
    pipe = Pipeline(config=config)

    # 1. Red the images
    pipe.add(readImages)

    # 2. Resize images (if done once comment out)
    # pipe.add(resize_images)

    # 3. Split the images in train & test images
    if pipe.config.folds == 1:
        pipe.add(split)
    else:
        pipe.add(kfold)

    # 4. Augment the images
    pipe.add(augment)

    # # 5. Create the darknet data file
    pipe.add(create_darknet_data)

    # # 6. Create the yolo config
    pipe.add(create_yolo_cfg)

    # Execute Pipeline
    pipe.execute()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocessing Pipeline')
    parser.add_argument('-f', metavar='number of folds', type=int, default=1,
                        help='Amount of folds to be created')
    parser.add_argument('-n', metavar='name', required=True, type=str,
                        help='Name of this pipeline run.')
    parser.add_argument('-i', metavar='input folder', type=str, default='./data',
                        help='Path where the original images are stored. Default to "./data"')
    parser.add_argument('-cls', metavar='classes.txt', required=False, type=str, default="./data/classes.txt",
                        help='Classes txt file for darknet')
    parser.add_argument('-o', metavar='output folder', type=str, default='./output',
                        help='Path where the results of this pipeline run are stored. Default to "./output"')
    parser.add_argument('-c', metavar='color', type=bool, default=False,
                        help='Whether the images are colored or greyscaled')
    parser.add_argument('-yolo_cfg', metavar='yolo cfg file', type=str, default='../model/darknet_cfgs/yolov4-tiny-custom.cfg',
                        help='Original yolovX config file that is beeing modified')
    parser.add_argument('-batch_size', metavar='max batch size', type=int, default=3000,
                        help='Max batch size of the yolovX.cfg file')

    args = parser.parse_args()
    main(args)
