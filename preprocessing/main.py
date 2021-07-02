from pathlib import Path
from util import create_transform
from functions import augment, clear_output_folder, resize_images, readImages, split, test
from Dataclasses import PipeConfig
from Pipeline import Pipeline


config = PipeConfig(
    inputFolder="./data",
    outputFolder="./output/augmentation_1/",
    imgSubFolderName="obj",
    resizedImgSize=600,
    finalImgSize=416,
    numberOfAugmentations=10,
    color=False,
    transform=create_transform()
)


transform = None

if __name__ == '__main__':
    # Initialize Pipeline
    pipe = Pipeline(config=config)

    # Add Steps

    # 1. Red the images
    pipe.add(readImages)

    # 2. Resize images (if done once comment out)
    # pipe.add(resize_images)

    # 3. Split the images in train & test images
    pipe.add(split)

    # 4. Augment the images
    pipe.add(augment)

    # Execute Pipeline
    pipe.execute()
