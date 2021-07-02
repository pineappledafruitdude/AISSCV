from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING
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

    def __init__(self, inputFolder: str, outputFolder: str, imgSubFolderName: str, resizedImgSize: int, finalImgSize: int, numberOfAugmentations: int, color: bool, transform: Compose) -> None:
        """
            Pipeline configuration class

            Args:
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
        self.outputFolder = Path(outputFolder)
        self.outputImgSubFolder = Path(outputFolder, imgSubFolderName)
        self.resizedImgSize = resizedImgSize
        self.finalImgSize = finalImgSize
        self.numberOfAugmentations = numberOfAugmentations
        self.color = color
        self.transform = transform
        self.train_txt = Path(self.outputFolder, "train.txt")
        self.test_txt = Path(self.outputFolder, "test.txt")
