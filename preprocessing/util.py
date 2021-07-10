from __future__ import annotations
import os
from pandas.core.frame import DataFrame
from ImageDataFrame import ImageDataFrame
import sys
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from albumentations.core.composition import Compose
import pandas as pd
import albumentations as A
import numpy as np
import cv2
import shutil
import argparse
import subprocess
import shlex
import logging
import imgaug
imgaug.random.seed(123)
if TYPE_CHECKING:
    from Pipeline import PipeConfig

logging.basicConfig(
    level=logging.DEBUG,  # allow DEBUG level messages to pass through the logger
    format='%(asctime)s - %(levelname)s: %(message)s'
)

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
        print(PRINT_ARROW + " " + question +
              prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print(PRINT_ARROW +
                  " Please respond with 'yes' or 'no' " "(or 'y' or 'n').")


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


def pipePrint(msg=""):
    logging.info("%s %s" % (PRINT_ARROW, msg))


def redPrint(msg=""):
    logging.error(msg)


def debugPrint(msg=""):
    logging.debug(msg)


def infoPrint(msg=""):
    logging.info(msg)


def stepPrint(i, total, name):
    logging.info("Step %i/%i '%s': Started" % (i, total, name))


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


def create_transform_1(config: PipeConfig) -> Compose:
    """Create the albumentation transform object"""
    transform = A.Compose(
        [

            A.RandomCrop(height=416, width=416, p=1),
            A.ShiftScaleRotate(shift_limit=0.0325, scale_limit=0.05, rotate_limit=25, interpolation=1, border_mode=4,
                               value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(
                8, 8), always_apply=False, p=0.5),
            A.Equalize(mode='cv', by_channels=False, mask=None,
                       mask_params=(), always_apply=False, p=0.7),
            A.HorizontalFlip(p=0.3),
            A.Sharpen(alpha=(0.2, 0.3), lightness=(
                0.5, 0.7), always_apply=False, p=0.05),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1,  always_apply=False, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120),  always_apply=False, p=0.5),
            A.Blur(blur_limit=2, always_apply=False, p=0.3),
            A.GaussNoise(var_limit=(1.0, 5.0), mean=0,
                         per_channel=False, always_apply=False, p=0.5),

        ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.2))

    return transform


def create_transform_2(config: PipeConfig) -> Compose:
    """Create the albumentation transform object"""
    if not config.color:
        # greyscale
        transform = A.Compose(
            [

                A.RandomCrop(height=config.final_img_size,
                             width=config.final_img_size, p=1),
                A.ShiftScaleRotate(shift_limit=0.0325, scale_limit=0.05, rotate_limit=25, interpolation=1, border_mode=4,
                                   value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(
                    8, 8), p=0.1),
                A.Equalize(mode='cv', by_channels=False, mask=None,
                           mask_params=(), p=0.7),
                A.HorizontalFlip(p=0.3),

                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.3), lightness=(
                        0.5, 0.7), p=0.05),
                    A.Blur(blur_limit=2, p=0.2),
                    A.MotionBlur(p=0.2)

                ], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1,  always_apply=False, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120),
                              p=0.5),
                A.GaussNoise(var_limit=(1.0, 5.0), mean=0,
                             per_channel=False, p=0.5)
            ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.2))
    else:
        # color
        transform = A.Compose(
            [
                A.RandomCrop(height=config.final_img_size,
                             width=config.final_img_size, p=1),
                A.FancyPCA(alpha=0.1, always_apply=False, p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.2,
                              saturation=0.15, hue=0.07, always_apply=False, p=0.75),
                A.ShiftScaleRotate(shift_limit=0.0325, scale_limit=0.05, rotate_limit=25, interpolation=1, border_mode=4,
                                   value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=0.75),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(
                    8, 8), always_apply=False, p=0.1),
                A.Equalize(mode='cv', by_channels=True, mask=None,
                           mask_params=(), p=0.7),
                A.HorizontalFlip(p=0.3),

                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.3), lightness=(
                        0.5, 0.7), p=0.05),
                    A.Blur(blur_limit=2, p=0.2),
                    A.MotionBlur(p=0.2)
                ], p=1),

                A.OneOf([
                    A.RandomRain(slant_lower=-1, slant_upper=1, drop_length=2, drop_width=1, drop_color=(200, 200, 200),
                                 blur_value=5, brightness_coefficient=0.7, rain_type='drizzle', p=1),
                    A.RandomSnow(snow_point_lower=0.05, snow_point_upper=0.3,
                                 brightness_coeff=2.5,  p=1),
                    A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6,
                                alpha_coef=0.08, p=1)
                ], p=0.15),

                A.RandomGamma(gamma_limit=(80, 120),
                              p=0.5),
                A.GaussNoise(var_limit=(1.0, 5.0), mean=0,
                             per_channel=False, p=0.3)


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
    parser.add_argument('-c', metavar='color', action=argparse.BooleanOptionalAction, type=bool, default=False,
                        help='Whether the images are colored or greyscaled')
    parser.add_argument('-yolo_cfg', metavar='yolo cfg file', type=str, default='../model/darknet_cfgs/yolov4-tiny-custom.cfg',
                        help='Original yolovX config file that is beeing modified')
    parser.add_argument('-batch_size', metavar='max batch size', type=int, default=3000,
                        help='Max batch size of the yolovX.cfg file')
    parser.add_argument('-nbr_augment', metavar='number of augmentations', type=int, default=10,
                        help='Number of augmentations to perform per train image')
    parser.add_argument('-t', metavar='transformation function', type=int, default=1,
                        help='Which transformation function should be used for the augmentation. Number 1 or 2')


def add_train_args(parser: argparse.ArgumentParser):
    """Add the args for the trainig to the provided parser"""
    parser.add_argument('-darknet', metavar='darknet folder', type=str, required=True,
                        help='Path to the darknet executable')


def train(config: PipeConfig, darknet_path: Path):
    number_of_runs = len(config.runs)
    for i, run in enumerate(config.runs):
        logging.info("Running Training for %i/%i" % (i+1, number_of_runs))

        yolo_conv = Path(darknet_path, 'yolov4-tiny.conv.29')

        # Go to darknet path
        os.chdir(darknet_path)

        # Train command
        run_darknet = './darknet detector train %s %s %s -dont_show' % (str(run.darknet_data),
                                                                        str(run.yolo_cfg),
                                                                        str(yolo_conv))

        execute_cmd("Training Darknet", run_darknet,
                    log_level="debug", show_error=False)

        # Test/Map command

        final_weight = Path(run.weights_folder,
                            run.yolo_cfg.stem+"_final.weights")
        results_txt = Path(run.output_folder, "results.txt")

        run_map = './darknet detector map %s %s %s' % (str(run.darknet_data),
                                                       str(run.yolo_cfg),
                                                       str(final_weight)
                                                       )

        execute_cmd("Map Darknet", run_map, log_level="info",
                    output_file=results_txt, show_error=False)

        # Remove image folder
        remove_img_folder = 'rm -rf %s' % run.img_folder
        execute_cmd("Removing image folder",
                    remove_img_folder, i+1, number_of_runs)

        # Copy the chart
        chart = Path(darknet_path, "chart.png")

        cp_chart = 'cp %s %s' % (
            str(chart), str(run.output_folder))

        execute_cmd("Copy chart", cp_chart, log_level="info")

    # Copy the output folders of all runs
    destination = Path("/", config.output_folder.name)

    cp_output_folders = 'gsutil -m cp -r %s gs://aisscv%s' % (
        str(config.output_folder), str(destination))

    execute_cmd("Copy run folder", cp_output_folders, log_level="info")


def execute_cmd(descr: str, cmd: str, log_level: str = "debug", output_file: Optional[Path] = None, show_error: bool = True):
    """Execute a command. Command must be string including the complete command"""
    infoPrint("%s (%s)" % (descr, cmd))

    process = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print_cmd_output(process=process, log_level=log_level,
                     output_file=output_file, show_error=show_error)
    pipePrint("Done %s" % descr)


def print_cmd_output(process: subprocess.Popen, log_level: str = "debug", output_file: Optional[Path] = None, show_error: bool = True):
    """Print the output of a command

    Args:
        process (subprocess.Popen): Process (command)
        log_level (str, optional): Log level. Defaults to "debug".
        output_file (Optional[Path], optional): If a file is specifed the output/errors are persisted in the file. Defaults to None.
        show_error (bool, optional): Should erros be shown. Defaults to True.
    """
    save_output = isinstance(output_file, Path)

    if save_output:
        f = open(output_file, 'a')
    while True:
        # Log output
        output = process.stdout.readline().strip().decode('utf-8')
        log(output, log_level)
        # Save output to file
        if save_output and output != "":
            f.write('%s %s' % (output, os.linesep))

        # Log Error
        error = process.stderr.readline().strip().decode('utf-8')
        if show_error is True:
            log(error, "error")

        # Save error to file
        if save_output and show_error and error != "":
            f.write('Error: %s %s' % (error, os.linesep))

        # Check return codes
        return_code = process.poll()
        if return_code is not None:
            pipePrint('RETURN CODE %s' % return_code)

            # Process has finished, read or save rest of the output
            for output in process.stdout.readlines():
                text = output.strip().decode('utf-8')
                log(text, log_level)
                # Save output
                if save_output and text != "":
                    f.write('%s %s' % (text, os.linesep))
            # Process has finished, read or save rest of the errors
            if show_error:
                for output in process.stderr.readlines():
                    error = output.strip().decode('utf-8')
                    log(error, "error")
                    # Save error
                    if save_output and error != "":
                        f.write('Error:%s %s' % (text, os.linesep))
            break
    if save_output:
        f.close()


def log(msg: str, log_level: str = "debug"):
    if msg == "":
        return
    if log_level == "error":
        logging.error(msg)
    if log_level == "info":
        logging.info(msg)
    else:
        logging.debug(msg)
