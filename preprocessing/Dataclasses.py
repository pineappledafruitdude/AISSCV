from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING
from pathlib import Path
import pandas as pd
from albumentations.core.composition import Compose

if TYPE_CHECKING:
    from Pipeline import Pipeline


@dataclass
class PipelineFunction:
    pipeline: Pipeline
    function: Callable
    kwargs: dict
    args: list

    def call(self, input):
        return self.function(
            config=self.pipeline.config,
            input=input,
            *self.kwargs,
            **self.args
        )


class RunConfig:
    output_folder: Path
    weights_folder: str
    img_folder: str
    train_txt: Path
    test_txt: Path
    classes_txt: Path
    yolo_cfg: Path
    darknet_data: Path
    run: int

    def __init__(self, output_folder: Path, run: int) -> None:
        """Pipeline config for a single run

        Args:
            output_folder (Path): The output folder of the container folder where all runs are stored
            run (int): Run Number
        """
        self.run = run
        self.output_folder = Path(
            output_folder, "run_"+str(run)).absolute()
        self.weights_folder = Path(self.output_folder, "weights").absolute()
        self.img_folder = Path(self.output_folder, "obj").absolute()
        self.train_txt = Path(self.output_folder, "train.txt").absolute()
        self.test_txt = Path(self.output_folder, "test.txt").absolute()
        self.classes_txt = Path(self.img_folder, "classes.txt").absolute()
        self.yolo_cfg = Path(self.output_folder, "yolo.cfg").absolute()
        self.darknet_data = Path(self.output_folder, "darknet.data").absolute()


class PipeConfig:
    input_folder: Path
    input_folder: Path
    runs: list[RunConfig]
    resized_img_size: int
    final_img_size: int
    number_of_augmentations: int
    color: bool
    transform: Compose
    org_classes_txt: Path
    org_yolo_cfg: Path
    max_batch_size: int
    folds: int

    def __init__(self, name: str, input_folder: str, output_folder: str, resized_img_size: int, final_img_size: int, number_of_augmentations: int, color: bool, transform: Compose, classes_txt: str, yolo_cfg: str, max_batch_size: int, folds: int) -> None:
        """
            Pipeline configuration class

            Args:
                name (str): The name of the pipeline run. In the output_folder a subfolder with this name will be created.
                input_folder (str): The folder where the original images are stored
                output_folder (str): The output folder of this pipeline run
                resized_img_size (int): The size of the images before passing it to the augmentation function
                final_img_size (int): The final size after the augmentation
                number_of_augmentations (int): The number of augmentations per image
                color (bool): Should color be included?
                transform (Compose): The albumentations transform variable
                classes_txt (str): Full path to the classes.txt file including the filename e.g. '/Path/classes.txt'
                yolo_cfg (str): Full path to the yolo.cfg file including the filename e.g. '/Path/yolov4.cfg'. The file is copied and modified.
                max_batch_size (int): Max batch size of the yolo.cfg file
                folds (int): Number of folds
        """
        self.input_folder = Path(input_folder).absolute()
        self.output_folder = Path(output_folder, name).absolute()
        self.runs = []
        # Create the run configs for each run
        for i in range(folds):
            fold_config = RunConfig(self.output_folder, i+1)
            self.runs.append(fold_config)

        self.resized_img_size = resized_img_size
        self.final_img_size = final_img_size
        self.number_of_augmentations = number_of_augmentations
        self.color = color
        self.transform = transform
        self.org_classes_txt = Path(classes_txt).absolute()
        self.org_yolo_cfg = Path(yolo_cfg).absolute()
        self.max_batch_size = max_batch_size
        self.folds = folds


class ImageDataFrame:
    """
        Wrapper around a dataframe to provide consistent data passing between functions.
        The DataFrame can be accessed via the frame variable.
        Columns: ["stem", "img_file", "label_file", "is_test", "class"]
    """
    frame: pd.DataFrame

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        if isinstance(df, pd.DataFrame):
            self.frame = df
        else:
            self.frame = pd.DataFrame(
                columns=["stem", "img_file", "label_file", "is_test", "class"])

    def addImg(self, img_path: Path, label_path: Path, img_class: str, is_test: Optional[bool] = None):
        """Add an image to the dataframe. Img and txt path are converted to absolutes

        Args:
            img_path (Path): Path to the image
            txt_path (Path): The path to the label
            img_class (str): The image class
            is_test (Optional[bool], optional): Optional is_test boolean. Defaults to None.
        """
        row = [img_path.stem, img_path.absolute(), label_path.absolute(), is_test,
               img_class]
        self.frame.loc[len(self.frame)] = row
