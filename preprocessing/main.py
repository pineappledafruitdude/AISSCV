import argparse
from util import create_transform
from functions import augment, resize_images, readImages, split
from Dataclasses import PipeConfig
from Pipeline import Pipeline


def main(args):
    config = PipeConfig(
        name=args.n,
        inputFolder=args.i,
        outputFolder=args.o,
        imgSubFolderName="obj",
        resizedImgSize=600,
        finalImgSize=416,
        numberOfAugmentations=10,
        color=args.c,
        transform=create_transform()
    )
    # Initialize Pipeline
    pipe = Pipeline(config=config)

    # 1. Red the images
    pipe.add(readImages)

    # 2. Resize images (if done once comment out)
    pipe.add(resize_images)

    # 3. Split the images in train & test images
    pipe.add(split)

    # 4. Augment the images
    pipe.add(augment)

    # Execute Pipeline
    pipe.execute()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocessing Pipeline')
    parser.add_argument('-n', metavar='name', required=True, type=str,
                        help='Name of this pipeline run.')
    parser.add_argument('-i', metavar='input folder', type=str, default='./data',
                        help='Path where the original images are stored. Default to "./data"')
    parser.add_argument('-o', metavar='output folder', type=str, default='./output',
                        help='Path where the results of this pipeline run are stored. Default to "./output"')
    parser.add_argument('-c', metavar='color', type=bool, default=False,
                        help='Whether the images are colored or greyscaled')

    args = parser.parse_args()
    main(args)
