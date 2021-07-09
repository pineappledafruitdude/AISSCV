from __future__ import annotations
from ImageDataFrame import ImageDataFrame
import re
from util import augmentImage, clear_folder, count_lines, debugPrint, get_labels, infoPrint, pipePrint, mkdir
from pathlib import Path
import shutil
from typing import TYPE_CHECKING
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from datetime import datetime

if TYPE_CHECKING:
    from Pipeline import PipeConfig

# Template function


def test(config: PipeConfig, input: ImageDataFrame, a):
    """Test function for demonstration purposes"""
    print("Test: %s. Input: %s" % (a, input))
    # logic here
    return "Ciao"


def clear_output_folder(config: PipeConfig, input: ImageDataFrame):
    """Clears the output folders"""
    clear_folder(config.output_folder)
    pipePrint("Output Folder cleared")
    return input


def create_output_directories(config: PipeConfig, input: ImageDataFrame):
    """Create the output directories (output and img output)"""
    mkdir(Path(config.output_folder))
    # Check if folds should be created
    for run in config.runs:
        pipePrint("Createing folders for %s" % run.output_folder.name)
        mkdir(Path(run.output_folder))
        # create image & weights output folders  in the folds subfolder
        mkdir(run.weights_folder)
        mkdir(run.img_folder)

    return input


def create_darknet_data(config: PipeConfig, input: ImageDataFrame):
    """Creates the darknet data file and copy classes.txt file"""
    num_classes = count_lines(config.org_classes_txt)

    # create folds subfolders
    for run in config.runs:
        # Copy classes.txt to run folder
        shutil.copy(config.org_classes_txt, run.classes_txt)
        # Create darknet.data file in the run folder
        with open(run.darknet_data, 'a') as f:
            f.write('classes = %i %s' % (num_classes, os.linesep))
            f.write('train = %s %s' % (run.train_txt, os.linesep))
            f.write('valid = %s %s' % (run.test_txt, os.linesep))
            f.write('names = %s %s' % (run.classes_txt, os.linesep))
            f.write('backup = %s %s' % (run.weights_folder, os.linesep))

    return input


def create_yolo_cfg(config: PipeConfig, input: ImageDataFrame):
    max_batch = config.max_batch_size

    num_classes = count_lines(config.org_classes_txt)

    # calculate the 2 steps values:
    step1 = 0.8 * max_batch
    step2 = 0.9 * max_batch

    num_filters = (num_classes + 5) * 3
    for run in config.runs:
        new_cfg_file = shutil.copy(
            config.org_yolo_cfg, run.yolo_cfg)
        with open(new_cfg_file) as f:
            s = f.read()

        s = re.sub('channels=\d*', 'channels=' +
                   str(3 if config.color else 1), s)
        s = re.sub('max_batches = \d*', 'max_batches = '+str(max_batch), s)
        s = re.sub('steps=\d*,\d*', 'steps=' +
                   "{:.0f}".format(step1)+','+"{:.0f}".format(step2), s)
        s = re.sub('classes=\d*', 'classes='+str(num_classes), s)
        s = re.sub('pad=1\nfilters=\d*', 'pad=1\nfilters=' +
                   "{:.0f}".format(num_filters), s)

        with open(new_cfg_file, 'w') as f:
            f.write(s)


def readImages(config: PipeConfig, input: ImageDataFrame, testSize=0.2) -> ImageDataFrame:
    """
        Read all images of the given input path and stores them in a DataFrame.
        The DataFrame has five columns: ["stem", "img_file", "label_file", "is_test", "class"].
        In this stage of the pipeline the column 'is_test' is not relevant.
        The class refers to the first part of the images, NOT the label class (see below).
        [Example]
        Mensa_1.jpg  -->    stem: 'Mensa_1',
                            img_file: '/Absoulte/Path/To/Mensa_1.jpg',
                            label_file: '/Absoulte/Path/To/Mensa_1.txt',
                            class: 'Mensa'
    """
    # logic here
    classes = [
        "AKK_ASTA",
        "Alte_Bib",
        "Audimax",
        "Gruenderschmiede",
        "Haber_Bosch",
        "Kolben",
        "Kopf",
        "Lernzentrum",
        "Mathebau",
        "Mensa",
        "Neue_Bib",
        "Soldat",
        "Studierendenwerk",
        "Waermflasche",
    ]
    df = ImageDataFrame()

    for img_class in classes:

        # Rename jpeg images
        jpeg_images = config.input_folder.glob("%s_*.jpeg" % img_class)
        for img in jpeg_images:
            pipePrint("Renaming JPEG %s" % img.name)
            img.rename(img.stem+".jpg")

        images = config.input_folder.glob("%s_*.jpg" % img_class)
        for img_path in images:
            if not img_path.is_file:
                continue
            # get the differnt absolute file paths
            stem = img_path.stem
            label_path = Path(img_path.parent, stem+".txt").absolute()

            # add the row to df
            df.addImg(
                img_path=img_path,
                label_path=label_path,
                img_class=img_class
            )

    return df


def resize_images(config: PipeConfig, input: ImageDataFrame) -> ImageDataFrame:
    """
        The images stored in the DataFrame (input) are resize according to the config
        entry 'resized_img_size'. If the images are already have the correct dimensions,
        no resize is performed.
    """

    # Resize images
    for i in input.frame[["img_file"]].iterrows():
        fileName = i[1].values[0]
        filePath = Path(fileName)
        outputPath = filePath

        img = cv2.imread(str(filePath))

        if img.shape[0] == config.resized_img_size and img.shape[0] == config.resized_img_size:
            continue

        resized_img = cv2.resize(
            img, (config.resized_img_size, config.resized_img_size), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(outputPath), resized_img)
        pipePrint("%s resized" % fileName)

    return input


def kfold(config: PipeConfig, input: ImageDataFrame) -> list[ImageDataFrame]:
    """
        Performs a stratified kfold of the images in the ImageDataFrame based on the column 'class' (Y).
        Each fold is added to a list of ImageDataFrames that will be passed on. 
        Each ImageDataFrame has  an additional colum: 'is_test' (bool).
    """
    X = input.frame.iloc[:, :-1]
    Y = input.frame.iloc[:, -1]  # last colum "class"

    # Create the k folds
    skf = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=1)
    skf.get_n_splits(X, Y)

    output: list[ImageDataFrame] = []
    i = 0
    for train_index, test_index in skf.split(X, Y):
        run = config.runs[i]
        name = config.runs[i].output_folder.name

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]

        # Concat the X and Y to give the DataFrame it's original columns
        test_data: pd.DataFrame = pd.concat([X_test, Y_test], axis=1)
        train_data: pd.DataFrame = pd.concat([X_train, Y_train], axis=1)

        # Add the is_test boolean
        test_data['is_test'] = True
        train_data['is_test'] = False

        # Create the full DataFrame thats passed onwards
        full_data = pd.concat([test_data, train_data])
        full_data.to_csv(str(Path(run.output_folder, name+".csv")))

        output.append(ImageDataFrame(full_data))
        i += 1

    return output


def split(config: PipeConfig, input: ImageDataFrame, test_size: float = 0.2) -> list[ImageDataFrame]:
    """
        Performs a stratified split of the images in the ImageDataFrame based on the column 'class' (Y)
        according to the specified test_size.
        The full ImageDataFrame will be passed on with an additional colum: 'is_test' (bool)
    """

    X = input.frame.iloc[:, :-1]
    Y = input.frame.iloc[:, -1]  # last colum "class"

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
    full_data.to_csv(str(Path(config.output_folder, "full_data.csv")))

    output = [ImageDataFrame(full_data)]

    return output


def augment(config: PipeConfig, input: list[ImageDataFrame]) -> list[ImageDataFrame]:
    """
        Iterates over all rows in the input ImageDataFrame.
        It is checked wether the images is used for testing via column 'is_test'

        Case 'is_test'=True:
            A copy of the original test image is created, resized to the 'final_img_size' config
            (and greyscaled depending on the 'color' config)
        Case 'is_test'=False:
            The augmentation is applied to the image.
            Then a copy of the original train image is created, resized to the 'final_img_size' config
            (and greyscaled depending on the 'color' config).
        All images that are copied (original ones) and augmented will be stored in the ImageDataFrame 'output'
        Images in this dataframe will either be saved to train.txt or test.txt depending on the 'is_test' flag
        of the original image.
        The test and train images are saved with their absolute path in the txt file.
        The train.txt or test.txt will be used for model training.
    """
    for i, run in enumerate(config.runs):
        input_data = input[i]

        number_of_images = len(input_data.frame[input_data.frame["is_test"] == False].index) * \
            (1+config.number_of_augmentations) + \
            len(input_data.frame[input_data.frame["is_test"] == True].index)

        pipePrint("%s: Creating %i images" %
                  (run.output_folder.name, number_of_images))

        # iterate in the df
        number_of_images_processed = 0
        for i, row in enumerate(input_data.frame.iterrows()):
            file_stem: str = row[1].values[0]
            img_file: str = row[1].values[1]
            label_file: str = row[1].values[2]
            is_test: bool = row[1].values[3]
            class_name: str = row[1].values[4]

            # This dataframe holds the image(s) that will be outputed to either the test.txt or train.txt
            output = ImageDataFrame()

            input_path_img = Path(img_file)
            output_path_img = Path(
                run.img_folder, file_stem + ".jpg")
            input_path_txt = Path(label_file)
            output_path_txt = Path(
                run.img_folder, file_stem + ".txt")

            # Read the image
            if config.color is True:
                image = cv2.imread(str(input_path_img), cv2.IMREAD_UNCHANGED)
            else:
                image = cv2.imread(str(input_path_img), cv2.IMREAD_GRAYSCALE)

            # Apply augmentation if the image is not a test image
            if not is_test:
                # Read in label
                labels = get_labels(input_path_txt)
                number_of_images_processed += config.number_of_augmentations
                # Augment data with albumentations
                if i % 50 == 0:
                    infoPrint("Fold %i: (%i/%i) Augmenting: File %s; #Labels %i; Class %s" %
                              (run.run, number_of_images_processed, number_of_images, file_stem, len(labels), class_name))
                else:
                    debugPrint("Fold %i: (%i/%i) Augmenting: File %s; #Labels %i; Class %s" %
                               (run.run, number_of_images_processed, number_of_images, file_stem, len(labels), class_name))
                augmented_df = augmentImage(config, image, labels,
                                            run.img_folder, file_stem, class_name)
                # Add the augmented images to the output df
                output.frame = pd.concat([output.frame, augmented_df.frame])

            # Save original image (train & test) in appropriate scaling
            number_of_images_processed += 1
            if i % 50 == 0:
                infoPrint("Fold %i: (%i/%i) Copying: Original %s File %s" %
                          (run.run, number_of_images_processed, number_of_images, "Test" if is_test else "", file_stem))
            else:
                debugPrint("Fold %i: (%i/%i) Copying: Original %s File %s" %
                           (run.run, number_of_images_processed, number_of_images, "Test" if is_test else "", file_stem))
            image = cv2.resize(image, (config.final_img_size, config.final_img_size),
                               interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(output_path_img), image)

            # Add the original image to the output df
            output.addImg(
                img_path=output_path_img,
                label_path=output_path_txt,
                img_class=class_name
            )

            # copy the original label file
            shutil.copy(input_path_txt, output_path_txt)

            output_txt = run.test_txt if is_test else run.train_txt
            # Add all images to the train.txt or test.txt
            with open(output_txt, "ab") as f:
                np.savetxt(f, output.frame[['img_file']], fmt='%s')

    return input
