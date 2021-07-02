from __future__ import annotations
from util import add_to_df, augmentImage, clear_folder, get_labels, pipePrint, mkdir
from pathlib import Path
import shutil
from typing import TYPE_CHECKING
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

if TYPE_CHECKING:
    from Dataclasses import PipeConfig

# Template function


def test(config: PipeConfig, input: pd.DataFrame, a):
    """Test function for demonstration purposes"""
    print("Test: %s. Input: %s" % (a, input))
    # logic here
    return "Ciao"


def clear_output_folder(config: PipeConfig, input):
    """Clears the output folders"""
    clear_folder(config.outputFolder)
    pipePrint("Output Folder cleared")


def create_output_directories(config: PipeConfig, input):
    """Create the output directories (output and img output)"""
    mkdir(Path(config.outputFolder))
    mkdir(Path(config.outputImgSubFolder))


def readImages(config: PipeConfig, input: Path, testSize=0.2):
    """
        Read all images of the given input path and stores them in a DataFrame.
        The DataFrame has two columns: ["stem", "img_file", "txt_file", "class"]. 
        The class refers to the first part of the images, NOT the label class (see below).
        [Example]
        Mensa_1.jpg  -->    stem: 'Mensa_1', 
                            img_file: '/Absoulte/Path/To/Mensa_1.jpg', 
                            txt_file: '/Absoulte/Path/To/Mensa_1.txt',
                            class: 'Mensa'
    """
    # logic here
    classes = [
        "AKK_ASTA",
        # "Alte_Bib",
        # "Audimax",
        # "Gruenderschmiede",
        # "Haber_Bosch",
        # "Kolben",
        # "Kopf",
        # "Lernzentrum",
        # "Mathebau",
        # "Mensa",
        # "Neue_Bib",
        # "Soldat",
        # "Studierendenwerk",
        # "Waermflasche",
    ]
    df = pd.DataFrame(columns=["stem", "img_file", "txt_file", "class"])

    for c in classes:

        # Rename jpeg images
        jpeg_images = config.inputFolder.glob("%s_*.jpeg" % c)
        for img in jpeg_images:
            pipePrint("Renaming JPEG %s" % img.name)
            img.rename(img.stem+".jpg")

        images = config.inputFolder.glob("%s_*.jpg" % c)
        for img in images:
            if not img.is_file:
                continue
            # get the differnt absolute file paths
            stem = img.stem
            absoluteImgPath = img.absolute()
            absoluteTxtPath = Path(img.parent, stem+".txt").absolute()

            # add the row to df
            row = [stem, absoluteImgPath, absoluteTxtPath, c]
            add_to_df(df, row)

    return df


def resize_images(config: PipeConfig, input: pd.DataFrame):
    """
        The images stored in the DataFrame (input) are resize according to the config
        entry 'resizedImgSize'. If the images are already have the correct dimensions,
        no resize is performed.
    """

    # Resize images
    for i in input[["img_file"]].iterrows():
        fileName = i[1].values[0]
        filePath = Path(fileName)
        outputPath = filePath

        img = cv2.imread(str(filePath))

        if img.shape[0] == config.resizedImgSize and img.shape[0] == config.resizedImgSize:
            continue

        resized_img = cv2.resize(
            img, (config.resizedImgSize, config.resizedImgSize), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(outputPath), resized_img)
        pipePrint("%s resized" % fileName)
    return input


def split(config: PipeConfig, input: pd.DataFrame, test_size: float = 0.2):
    """
        Performs a stratified split of the images passen in the DataFrame column 'file' (input)
        according to the specified test_size.
        The full DataFrame will be passed on with an additional colum: 'is_test' (bool)
    """
    train_txt = config.train_txt
    test_txt = config.test_txt

    X = input.iloc[:, :-1]
    Y = input.iloc[:, -1]  # last colum "class"

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y)

    # Concat the X and Y to give the DataFrame it's original columns
    test_data: pd.DataFrame = pd.concat([X_test, Y_test], axis=1)
    train_data: pd.DataFrame = pd.concat([X_train, Y_train], axis=1)

    # Add the is_test boolean
    test_data['is_test'] = True
    train_data['is_test'] = False

    # Create the full DataFrame thats passed onwards
    full_data = pd.concat([test_data, train_data])
    full_data.to_csv(str(config.outputFolder)+"full_data.txt")

    return full_data


def augment(config: PipeConfig, input: pd.DataFrame):
    """
        Iterates over all rows in the input DataFrame.
        It is checked wether the images is used for testing via column 'is_test'

        Case 'is_test'=True:
            A copy of the original test image is created, resized to the 'finalImgSize' config 
            (and greyscaled depending on the 'color' config)
        Case 'is_test'=False:
            The augmentation is applied to the image. 
            Then a copy of the original train image is created, resized to the 'finalImgSize' config 
            (and greyscaled depending on the 'color' config).
        All images that are copied (original ones) and augmented will be stored in the DataFrame 'output_df'
        Images in this dataframe will either be saved to train.txt or test.txt depending on the 'is_test' flag 
        of the original image.
        The test and train images are saved with their absolute path in the txt file.
        The train.txt or test.txt will be used for model training.
    """
    # iterate in the df
    for i in input.iterrows():
        fileStem: str = i[1].values[0]
        img_file: str = i[1].values[1]
        txt_file: str = i[1].values[2]
        className: str = i[1].values[3]
        is_test: bool = i[1].values[4]

        # This dataframe holds the image(s) that will be outputed to either the test.txt or train.txt
        output_df = pd.DataFrame(
            columns=["stem", "img_file", "txt_file", "class"])

        input_path_img = Path(img_file)
        output_path_img = Path(config.outputImgSubFolder, fileStem + ".jpg")
        input_path_txt = Path(txt_file)
        output_path_txt = Path(config.outputImgSubFolder, fileStem + ".txt")

        # Read the image
        image = cv2.imread(str(input_path_img), cv2.IMREAD_GRAYSCALE)

        # Apply augmentation if the image is not a test image
        if not is_test:
            # Read in label
            labels = get_labels(input_path_txt)

            # Augment data with albumentations
            pipePrint("Augmenting: File: %s; #Labels %i; Class %s" %
                      (fileStem, len(labels), className))
            augmented_df = augmentImage(config, image, labels,
                                        config.outputImgSubFolder, fileStem, className)
            # Add the augmented images to the output df
            output_df = pd.concat([output_df, augmented_df])

        # Save original image (train & test) in appropriate scaling
        image = cv2.resize(image, (416, 416),
                           interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output_path_img), image)

        # Add the original image to the output df
        row = [fileStem, output_path_img.absolute(), output_path_txt.absolute(),
               className]
        add_to_df(output_df, row)

        # copy the original label file
        shutil.copy(input_path_txt, output_path_txt)

        output_txt = config.test_txt if is_test else config.train_txt
        # Add all images to the train.txt or test.txt
        with open(output_txt, "ab") as f:
            np.savetxt(f, output_df[['img_file']], fmt='%s')
