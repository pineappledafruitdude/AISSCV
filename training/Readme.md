# Training

Training is done via google cloud AI Platform
<br>
Image: `eu.gcr.io/delta-basis-318917/aisscv-dynamic-cfg:1.0`

## Arguments for Training

| arg             |  Type  | Default | Description                                                                                   |
| --------------- | :----: | :-----: | --------------------------------------------------------------------------------------------- |
| --name          | `str`  | `test`  | Name of this pipeline run. A subfolder with this name will be created in the output directory |
| --color         | `bool` |   `0`   | Whether the images are colored or greyscaled. Value: 0 or 1                                   |
| --folds         | `int`  |   `1`   | If f=1 then a train_test_split is performed (20%) if f>1 f-folds are created for training     |
| --batch_size    | `int`  | `3000`  | Max batch size that is saved to the yolovX.cfg file used for training                         |
| --augmentations | `int`  |  `10`   | Number of augmentations to perform per train image                                            |

## Submitting a Job

1. Open google cloud console
2. Click on AI Platform -> Jobs
3. Add new job with custom code
4. Click on the select master image button
   <img src="./img/add_job_1.png" width=400 style="margin-top:20px; margin-bottom: 20px" />
5. Select the Image `eu.gcr.io/delta-basis-318917/aisscv-dynamic-cfg:1.0`
   <img src="./img/add_job_2.png" width=400 style="margin-top:20px; margin-bottom: 20px" />
6. Click on next
7. Add the parameters show above. Each in one line: e.g

    ```
    --name=run_1
    --color=1
    --folds=5
    ```

    <img src="./img/add_job_3.png" width=400 style="margin-top:20px; margin-bottom: 20px" />

8. Click on next
9. Enter a job name
10. Select `europe-west1` as region
11. Select `BASIC_GPU` as scale tier
12. Click on `Done`

# Results

Results will be saved to a google cloud bucket
