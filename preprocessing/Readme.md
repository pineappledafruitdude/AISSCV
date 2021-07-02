# Install

Python Version used: 3.9.5
You **MUST** be in the preprocessing folder in your terminal

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
python3 main.py -n "The name of the run"
```

## Arguments

The following arguments can be passed when calling `python3 main.py`:

| arg | Long          | Description                                                                                   |
| --- | ------------- | --------------------------------------------------------------------------------------------- |
| -n  | Name          | Name of this pipeline run. A subfolder with this name will be created in the output directory |
| -i  | Input Folder  | Path where the original images are stored. Default to "./data" $12                            |
| -o  | Output Folder | Path where the results of this pipeline run are stored. Default to "./output"                 |
| -c  | Color         | Whether the images are colored or greyscaled                                                  |

## Other options

The following other options can be configured in the file `main.py`:

| Config                | Description                                                                                                                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| imgSubFolderName      | The subfolder where the images are stored. Example for imgSubFolderName='obj: If your output path is './output' and your name is 'run_1' then the augmented images will be stored in './output/run_1/obj' |
| resizedImgSize        | The size of the images before the augmentation is applied. Actually the original images in the inputfolder are cropped in place before the augmentation starts centered                                   |
| finalImgSize          | The image size after the pipeline is done. Applies to all train & test images.                                                                                                                            |
| numberOfAugmentations | The number of augmentations per image                                                                                                                                                                     |
