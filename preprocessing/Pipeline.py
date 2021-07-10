from __future__ import annotations
import sys
from pathlib import Path, PosixPath
from typing import Callable
from dataclasses import dataclass
from typing import Callable
import json

from albumentations.core.composition import Compose
from util import create_transform_1, create_transform_2, infoPrint, pipePrint, redPrint, stepPrint
from functions import clear_output_folder, create_output_directories, store_config


class Pipeline:
    config: PipeConfig
    steps: list[PipelineFunction]

    def __init__(self, config: PipeConfig):
        self.config = config
        self.steps = []

        self.setup()
        infoPrint("Running pipe with the following config: \n %s" %
                  (self.config))

    def setup(self):
        """Setup the pipeline"""
        # Create the necessary directories
        self.add(clear_output_folder)
        self.add(create_output_directories)
        # save the configuration
        self.add(store_config)

        # Check if input directory exists
        if not self.config.input_folder.exists():
            redPrint("The input folder '%s' doesn't exist" %
                     self.config.input_folder)
            sys.exit()
        if not self.config.org_classes_txt.exists():
            redPrint("The orginal darknet classes file '%s' doesn't exist" %
                     self.org_classes_txt)
            sys.exit()
        if not self.config.org_yolo_cfg.exists():
            redPrint("The orginal yolo.cfg file '%s' doesn't exist" %
                     self.org_yolo_cfg)
            sys.exit()
        if not self.config.folds > 0:
            redPrint("The number of folds must be > 0")
            sys.exit()

    def add(self, function: Callable, *kwargs, **args):
        self.steps.append(
            PipelineFunction(
                pipeline=self,
                function=function,
                kwargs=kwargs,
                args=args
            )
        )

    def execute(self):
        lastRestult = None
        numberOfSteps = len(self.steps)

        for i, step in enumerate(self.steps):
            stepPrint(i+1, numberOfSteps, step.function.__name__)
            # Exectute the steps function
            lastRestult = step.call(input=lastRestult)
            # Done
            pipePrint("Done")


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
    input_folder: Path
    input_folder: Path
    runs: list[RunConfig]
    resized_img_size: int
    final_img_size: int
    number_of_augmentations: int
    color: bool
    org_classes_txt: Path
    org_yolo_cfg: Path
    max_batch_size: int
    folds: int
    transform: Compose
    transform_nbr: int
    occlude: bool

    def __init__(self, name: str, input_folder: str, output_folder: str, resized_img_size: int, final_img_size: int, number_of_augmentations: int, color: bool, transform: int, classes_txt: str, yolo_cfg: str, max_batch_size: int, folds: int, occlude: bool) -> None:
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
                transform (int): Which albumentation transform method should be used (Number 1 or Number 2)
                classes_txt (str): Full path to the classes.txt file including the filename e.g. '/Path/classes.txt'
                yolo_cfg (str): Full path to the yolo.cfg file including the filename e.g. '/Path/yolov4.cfg'. The file is copied and modified.
                max_batch_size (int): Max batch size of the yolo.cfg file
                folds (int): Number of folds
                occlude (bool): Should the images be occluded?
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
        if transform == 1:
            self.transform_nbr = 1
            self.transform = create_transform_1(self)
        else:
            self.transform_nbr = 2
            self.transform = create_transform_2(self)
        self.org_classes_txt = Path(classes_txt).absolute()
        self.org_yolo_cfg = Path(yolo_cfg).absolute()
        self.max_batch_size = max_batch_size
        self.folds = folds
        self.occlude = occlude

    def to_dict(self) -> dict:
        """Dictionary represantation of the config"""
        repr = {}
        for key, value in self.__dict__.items():
            if key == "runs":
                continue
            elif type(value) == PosixPath or type(value) == Compose:
                repr[key] = str(value)
            else:
                repr[key] = value
        return repr

    def __str__(self) -> str:
        repr = self.to_dict()
        del repr["transform"]
        return json.dumps(repr, indent=4)


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
