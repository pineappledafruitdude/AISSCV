# Pipeline Overview

<img src="./doc_img/pipeline.png" width=600 />

# Install

Python Version used: 3.9.5

<div style="color:red; margin-bottom: 20px; text-align:center; font-weight: bold; padding: 5px; background: #e5e5e5">
You MUST be in the preprocessing folder in your terminal
</div>

1. Create venv

```zsh
python3 -m venv ./env
```

1. Activate venv

```zsh
source env/bin/activate
```

2. Install requirements

```zsh
pip3 install -r requirements.txt
```

# Execute Pipeline

<div style="color:red; margin-bottom: 20px; text-align:center; font-weight: bold; padding: 5px; background: #e5e5e5">
Copy the original images in the preprocessing data directory<br>
The images in the input path provided are overwritten with the resized ones
</div>

Run the following minimal command to execute the pipeline:

```zsh
python3 run_pipe.py -n "The name of the run"
```

## Arguments

The following arguments can be passed when calling `python3 run_pipe.py -n "Name"`:

| arg            | Type   | Default                                        |                         | Description                                                                                                                                                                  |
| -------------- | ------ | ---------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -n             | `str`  | -                                              | Name                    | Name of this pipeline run. A subfolder with this name will be created in the output directory                                                                                |
| -i             | `str`  | `./data`                                       | Input Folder            | Path where the original images are stored. Default to "./data"                                                                                                               |
| -cls           | `str`  | `./data/classes.txt`                           | Darknet classes file    | Full path to the darknet classes file. E.g. "./data/classes.txt"                                                                                                             |
| -o             | `str`  | `./output`                                     | Output Folder           | Path where the results of this pipeline run are stored. Default to "./output"                                                                                                |
| -c             | `bool` |                                                | Color                   | Whether the images are colored or greyscaled. If the `-c`flag is added to the command, then color is used. If the flag is not added greyscal is used.                        |
| -occl          | `bool` |                                                | Occlude                 | Whether the images should be occluded or not. If the `-occl`flag is added to the command, then the occlude function is applied. If the flag is not added then not.           |
| -incl_no_label | `bool` |                                                | Include no label images | Add the no label images to the training dataset. If the `-incl_no_label`flag is added to the command, then the no label images are added. If the flag is not added then not. |
| -f             | `int`  | `1`                                            | Number of folds         | If f=1 then a train_test_split is performed (20%) if f>1 f-folds are created for training                                                                                    |
| -yolo_cfg      | `str`  | `../model/darknet_cfgs/yolov4-tiny-custom.cfg` | Yolo cfg file           | Original yolovX config file that is beeing modified. Default to '../model/darknet_cfgs/yolov4-tiny-custom.cfg'                                                               |
| -max_batches   | `int`  | `3000`                                         | Max. Batches            | Max. batches (iterations) that is saved to the yolovX.cfg file used for training                                                                                             |
| -nbr_augment   | `int`  | `10`                                           | Number of augmentations | Number of augmentations to perform per train image                                                                                                                           |
| -t             | `int`  | `1`                                            | Transform function      | Which type of transform function to be applied: Number 1 or Number 2                                                                                                         |
| -is_final      | `bool` |                                                | Final Model             | Add the is_final flag if you want all images to be used for training the final model                                                                                         |

## Other options

The following other options can be configured in the file `run_pipe.py`:

| Config           | Description                                                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| resized_img_size | The size of the images before the augmentation is applied. Actually the original images in the input folder are cropped in place before the augmentation starts |
| final_img_size   | The image size after the pipeline is done. Applies to all train & test images.                                                                                  |
