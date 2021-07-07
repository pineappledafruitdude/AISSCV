from __future__ import annotations
import os
from pandas.core.frame import DataFrame
from Dataclasses import ImageDataFrame, PipeConfig
import sys
from pathlib import Path
from typing import List, Optional
from albumentations.core.composition import Compose
from rich.console import Console
import pandas as pd
import albumentations as A
import numpy as np
import cv2
import shutil
import argparse
import subprocess
import shlex

console = Console()

PRINT_ARROW = "------->"

DICT = {0: 'Mensa',
        1: 'AKK',
        2: 'Audimax',
        3: 'Neue Bib',
        4: 'Alte Bib',
        5: 'Studierendenwerk',
        6: 'Lernzentrum',
        7: 'Mathebau',
        8: 'Harber-Bosch-Reaktor',
        9: 'Statue am Ehrenhof',
        10: 'Heinrich-Hertz-Denkmal',
        11: 'Kolben',
        12: 'Wärmeflasche',
        13: 'Gründerschmiede'}


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        console.print(PRINT_ARROW + " " + question +
                      prompt, style="bold red", end=None)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            console.print(PRINT_ARROW +
                          " Please respond with 'yes' or 'no' " "(or 'y' or 'n').", style="bold red")


def mkdir(p: Path) -> Path:
    """Creates a directory if it doesn't exist yet. The directory Path is returned.
    You can pass as many chunks seperated by a comma, e.g. mkdir('this/is', 'a', 'folder')
    results in 'this/is/a/path'. You can also specify path in windows style with backslash """
    try:

        if p.exists() is False:
            pipePrint("Creating directory %s" % p)
            p.mkdir(parents=True)
        else:
            pipePrint("Directory %s exists" % p)
        return p

    except Exception as error:
        print("Error creating directory %s: %s" % (p, error,))
        sys.exit()


def clear_folder(folder: Path):
    """Clears the folder"""
    if not folder.exists():
        return
    data = folder.glob('**/*')
    clean = query_yes_no(
        "The folder '%s' is not empty. Do you want to clean it?" % folder)

    if clean:
        pipePrint("Cleaning Folder...")
        for i in data:
            i.unlink() if i.is_file() else shutil.rmtree(str(i), ignore_errors=True)
    else:
        pipePrint("Stopping pipeline as the folder is not empty")
        sys.exit()


def pipePrint(*args, style=None):
    console.print(PRINT_ARROW, *args, style=style)


def redPrint(*args):
    console.print(*args, style="red")


def greenPrint(*args):
    console.print(*args, style="green")


def bluePrint(*args, bold: bool = False):
    console.print(*args, style="blue %s" % ("bold" if bold else ""))


def stepPrint(i, total, name):
    bluePrint("\nStep %i/%i '%s': Started" % (i, total, name), bold=True)


def preprocess_label(line: str) -> List:
    line = line.split(" ")
    classes = pd.Series(int(line[0]))
    classes_m = classes.map(DICT)
    line = pd.Series(line[1:])
    line = line.astype('float32')
    line = line.tolist()
    line.append(classes_m[0])
    return line


def inverse_mapping(f):
    return f.__class__(map(reversed, f.items()))


def get_labels(filePath: Path) -> List[str]:
    labels = []
    # Open the file
    file = open(filePath, "r")
    lines = file.readlines()
    # Read lines
    for line in lines:
        line = line.strip()
        # Process txt file for input into albumentations
        line = preprocess_label(line)
        labels.append(line)
    file.close()

    return labels

# Image Augmentation


def augmentImage(config: PipeConfig, image, bboxes: List[str], output_path: Path, file_stem: str, class_name: str) -> ImageDataFrame:
    transform = config.transform

    output_df = ImageDataFrame()

    for j in range(config.number_of_augmentations):
        output_img = Path(output_path, file_stem +
                          "_transformed_"+str(j+1) + ".jpg")
        output_txt = Path(output_path, file_stem + "_transformed_" +
                          str(j+1) + ".txt")

        while True:
            transformed = transform(image=image, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_image = np.asarray(transformed_image)
            transformed_bboxes = transformed['bboxes'].copy()

            if(transformed_bboxes != []):
                break

        transformed_bboxes = transformed['bboxes'].copy()

        t_img = transformed_image.copy()
        transformed['bboxes'].clear()

        # Potentially visualize bbox in output image, disabled in final output as no

        # Uncomment to Get BBOXES and Text on Images when saving them

        # for bbox in transformed_bboxes:
        #   t_img=visualize_bbox(transformed_image, bbox[0:4], bbox[-1], color=BOX_COLOR, thickness=2)

        # Save augmented images and bounding boxes

        inv_d = inverse_mapping(DICT)

        transformed_df = pd.DataFrame(transformed_bboxes)
        # ensure transformed_bboxes is empty for next iteration
        transformed_bboxes = transformed_bboxes.clear()
        # prep DataFrame for saving labels in correct yolo format
        # print(transformed_df)
        transformed_df.iloc[:, 4] = transformed_df.iloc[:, 4].map(inv_d)
        transformed_df.insert(loc=0, column='ident', value=1)
        transformed_df.iloc[:, 0] = transformed_df.iloc[:, 5]
        transformed_df = transformed_df.drop(4, axis=1)
        transformed_df = transformed_df.astype('string')
        # Save image to file
        cv2.imwrite(str(output_img), transformed_image)
        # Save labels to file
        np.savetxt(str(output_txt), transformed_df.values, fmt='%s')
        # ensure df is cleared for next iteration
        transformed_df = transformed_df[0:0]

        # Add augmented img to df
        output_df.addImg(
            img_path=output_img,
            label_path=output_txt,
            img_class=class_name
        )
    return output_df


def create_transform():
    """Create the albumentation transform object"""
    transform = A.Compose(
        [

            A.RandomCrop(height=416, width=416, p=1),
            # A.FancyPCA (alpha=0.1, always_apply=False, p=0.5),
            # A.ColorJitter(brightness=0.8, contrast=0.6, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            # A.ToGray,
            A.ShiftScaleRotate(shift_limit=0.0325, scale_limit=0.05, rotate_limit=25, interpolation=1, border_mode=4,
                               value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(
                8, 8), always_apply=False, p=0.5),
            A.Equalize(mode='cv', by_channels=False, mask=None,
                       mask_params=(), always_apply=False, p=0.7),
            # A.Rotate(limit=40, p=0.8, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.3),
            A.Sharpen(alpha=(0.2, 0.3), lightness=(
                0.5, 0.7), always_apply=False, p=0.05),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1,  always_apply=False, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120),  always_apply=False, p=0.5),
            # A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=True, interpolation=1, always_apply=False, p=0.5),
            # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
            A.Blur(blur_limit=2, always_apply=False, p=0.3),
            # A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
            # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
            # A.ImageCompression(quality_lower=25, quality_upper=100, compression_type=ImageCompressionType.JPEG, always_apply=False, p=0.8),
            A.GaussNoise(var_limit=(1.0, 5.0), mean=0,
                         per_channel=False, always_apply=False, p=0.5),
            # A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),


        ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.2))

    return transform


def count_lines(txt_file: Path) -> int:
    """Count lines in a txt file"""
    lines = 0
    with open(txt_file, 'r') as f:
        for line in f:
            if line != "\n":
                lines += 1

    return lines


def add_pipe_args(parser: argparse.ArgumentParser):
    """Add the args for the pipeline to the provided parser"""

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
    parser.add_argument('-nbr_augment', metavar='number of augmentations', type=int, default=10,
                        help='Number of augmentations to perform per train image')


def add_train_args(parser: argparse.ArgumentParser):
    """Add the args for the trainig to the provided parser"""
    parser.add_argument('-darknet', metavar='darknet folder', type=str, required=True,
                        help='Path to the darknet executable')


def train(config: PipeConfig, darknet_path: Path):
    number_of_runs = len(config.runs)
    for i, run in enumerate(config.runs):
        yolo_conv = Path(darknet_path, 'yolov4-tiny.conv.29')

        # Go to darknet path
        os.chdir(darknet_path)

        # Train command
        run_darknet = './darknet detector train %s %s %s -dont_show' % (str(run.darknet_data),
                                                                        str(run.yolo_cfg),
                                                                        str(yolo_conv))

        execute_cmd("Training Darknet", run_darknet, i+1, number_of_runs)

        # Test/Map command

        final_weight = Path(run.weights_folder,
                            run.yolo_cfg.stem+"_final.weights")
        results_txt = Path(run.output_folder, "results.txt")

        run_map = './darknet detector map %s %s %s > %s' % (str(run.darknet_data),
                                                            str(run.yolo_cfg),
                                                            str(final_weight),
                                                            str(results_txt))

        execute_cmd("Map Darknet", run_map, i+1, number_of_runs, results_txt)

        # Remove image folder
        remove_img_folder = 'rm -rf %s' % run.img_folder
        execute_cmd("Removing image folder",
                    remove_img_folder, i+1, number_of_runs)

        # Copy the chart
        chart = Path(darknet_path, "chart.png")
        cp_chart = 'cp %s %s' % (
            str(chart), str(run.output_folder))
        execute_cmd("Copy chart", cp_chart, i+1, number_of_runs)

    # Copy the output folders of all runs
    destination = Path("/", config.output_folder.name)
    cp_output_folders = 'gsutil -m cp -r %s gs://aisscv%s' % (
        str(config.output_folder), str(destination))
    execute_cmd("Copy run folder", cp_output_folders,
                number_of_runs, number_of_runs)


def execute_cmd(descr: str, cmd: str, number: int, total: int, output_file: Optional[Path] = None):
    """Execute a commad. Command must be string including the complete command"""
    bluePrint("\nRUN %i/%i: %s (%s)" % (number, total, descr, cmd), bold=True)

    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print_cmd_output(process, output_file)


def print_cmd_output(process: subprocess.Popen, output_file: Optional[Path] = None):
    while True:
        output = process.stdout.readline()
        output_2 = process.stderr.readline()
        pipePrint(output.strip().decode('utf-8'))
        pipePrint(output_2.strip().decode('utf-8'))
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            save_output = isinstance(output_file, Path)
            pipePrint('RETURN CODE', return_code)
            # Process has finished, read or save rest of the output
            if save_output:
                f = open(output_file, 'a')

            for output in process.stdout.readlines():
                text = output.strip().decode('utf-8')
                pipePrint(text)
                if save_output:
                    f.write('%s %s' % (text, os.linesep))
            for output in process.stderr.readlines():
                text = output.strip().decode('utf-8')
                pipePrint(text)
                if save_output:
                    f.write('%s %s' % (text, os.linesep))
            if save_output:
                f.close()
            break
