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


class PipeConfig:
    inputFolder: Path
    outputFolder: Path
    outputImgSubFolder: Path
    resizedImgSize: int
    finalImgSize: int
    numberOfAugmentations: int
    color: bool
    transform: Compose
    train_txt: Path
    test_txt: Path

    def __init__(self, name: str, inputFolder: str, outputFolder: str, imgSubFolderName: str, resizedImgSize: int, finalImgSize: int, numberOfAugmentations: int, color: bool, transform: Compose) -> None:
        """
            Pipeline configuration class

            Args:
                name (str): The name of the pipeline run. In the outputfolder a subfolder with this name will be created.
                inputFolder (str): The folder where the original images are stored
                outputFolder (str): The output folder of this pipeline run
                imgSubFolderName ([type]): The name of the subfolder created within the outputFolder where the training images are stored
                resizedImgSize (int): The size of the images before passing it to the augmentation function
                finalImgSize (int): The final size after the augmentation
                numberOfAugmentations (int): The number of augmentations per image
                color (bool): Should color be included?
                transform (Compose): The albumentations transform variable
        """
        self.inputFolder = Path(inputFolder)
        self.outputFolder = Path(outputFolder, name)
        self.outputImgSubFolder = Path(self.outputFolder, imgSubFolderName)
        self.resizedImgSize = resizedImgSize
        self.finalImgSize = finalImgSize
        self.numberOfAugmentations = numberOfAugmentations
        self.color = color
        self.transform = transform
        self.train_txt = Path(self.outputFolder, "train.txt")
        self.test_txt = Path(self.outputFolder, "test.txt")


class ImageDataFrame:
    """
        Wrapper around a dataframe to provide consistent data passing between functions.
        The DataFrame can be accessed via the frame variable.
        Columns: ["stem", "img_file", "label_file", "is_test", "class"]
    """
    frame: pd.Datafram

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
