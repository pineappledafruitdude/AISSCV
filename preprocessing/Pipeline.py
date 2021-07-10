from __future__ import annotations
from typing import Callable
import sys
from util import infoPrint, pipePrint, redPrint, stepPrint
from functions import clear_output_folder, create_output_directories, store_config
from Dataclasses import PipeConfig
from Dataclasses import PipelineFunction


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
