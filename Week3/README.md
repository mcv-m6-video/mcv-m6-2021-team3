# Week 3

## Introduction
In this project the main goal was to get used to different background estimation methods. The project contains several source code files that implement the required functionality.

* [main.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/main.py): Contains the pipeline to execute the different tasks.
* [ai_city.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/ai_city.py): Contains the class AICity, where the data is processed in order to obtain the background estimation.
* [metrics.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/metrics.py): Contains functions related to get quantitative results.
* [utils.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/utils.py): Contains other functions, such as the one to write a json or read one.
* [visualize.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/visualize.py): Contains the functinos related to plot the different (qualitative) results.
* [detect2.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/detect2.py): Contains the class Detect2, which contains the function to predict the Detectron2 models.
* [tf_models.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/tf_models.py): Contains the class TFModel, which contains the functions to predict and train the TensorFlows models.
* [yolov3.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/yolov3.py): Contains the class UltralyricsYolo, which contains the funtion to predict and train the Ultralytics Yolo model. 
* [sort.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/sort.py): Contains the class Sort, which is the one related to compute the Kalman tracking.


## Inference
### Available tasks
* **Task 1.1**: Off-the-shelf
* **Task 1.2**: Fine-tuning
* **Task 1.3**: K-fold Cross validation
* **Task 2.1**: Tracking by Overlap
* **Task 2.2**: Tracking with a Kalman Filter
* **Task 2.3**: IDF1 for Multiple Object Tracking


Run the command below, from Week2/src folder, to obtain the desired task results.

```bash
$ python main.py 
```
Any parameter could be modified in file [Config.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week2/src/config/config.py)

### Directory structure

```bash
├── datasets
│   ├── AICity
│   │   ├── train
│   │   │   ├── S03
│   │   │   │   ├── c010
│   │   │   │   │   ├── vdo
├── Week3
│   ├── src
│   │   ├── main.py
│   │   ├── utils
│   │   │   ├── ai_city.py
│   │   │   ├── metrics.py
│   │   │   ├── visualize.py
│   │   │   ├── utils.py
│   │   │   ├── detect2.py
│   │   │   ├── tf_models.py
│   │   │   ├── yolov3.py
│   │   │   ├── sort.py
│   │   ├── data
│   │   ├── models
│   │   ├── yolov3
│   │   ├── runs
│   │   ├── config
│   │   │   ├── config.py
```

## Results
### Task 1.1: Object Detection: Off-the-shelf

<table>
    <thead>
        <tr>
            <th>Network</th>
            <th>Framework</th>
            <th>mIoU</th>
            <th>mAP50</th>
            <th>mAP70</th>
            <th>#Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Yolo V3 640x640</td>
            <td rowspan=3>&nbsp;Ultralytics</td>
            <td>0.7456</td>
            <td>0.5247</td>
            <td>0.5188</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;61.9M</td>
        </tr>
        <tr>
            <td>Yolo V3 SPP 640x640</td>
            <td>0.7262</td>
            <td>0.5532</td>
            <td>0.5339</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;63.0M</td>
        </tr>
        <tr>
            <td>Yolo V3 Tiny 640x640</td>
            <td>0.7476</td>
            <td>0.5738</td>
            <td>0.5409</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.9M</td>
        </tr>
        <tr>
            <td>SSD MN V1 FPN 640x640</td>
            <td rowspan=8>TensorFlow</td>
            <td>0.3911</td>
            <td>0.5782</td>
            <td>0.5655</td>
            <td></td>
        </tr>
        <tr>
            <td>SSD MN V2 FPNLite 640x640</td>
            <td>0.3022</td>
            <td>0.6152</td>
            <td>0.5943</td>
            <td></td>
        </tr>
        <tr>
            <td>SSD RN101 V1 FPN 640x640</td>
            <td>0.2267</td>
            <td>0.3753</td>
            <td>0.3687</td>
            <td></td>
        </tr>
        <tr>
            <td>SSD RN152 V1 FPN 640x640</td>
            <td>0.2365</td>
            <td>0.4724</td>
            <td>0.4683</td>
            <td></td>
        </tr>
        <tr>
            <td>EfficientDet D1 640x640</td>
            <td>0.2184</td>
            <td>0.4995</td>
            <td>0.5069</td>
            <td></td>
        </tr>
        <tr>
            <td>FR-CNN RN101 V1 640x640</td>
            <td>0.256</td>
            <td>0.443</td>
            <td>0.4366</td>
            <td></td>
        </tr>
        <tr>
            <td>CN RN101 V1 FPN 512x512</td>
            <td>0.2412</td>
            <td>0.4377</td>
            <td>0.4254</td>
            <td></td>
        </tr>
        <tr>
            <td>MR-CNN Inception RN V2 1024x1024</td>
            <td>0.2943</td>
            <td>0.4405</td>
            <td>0.4499</td>
            <td></td>
        </tr>
    </tbody>
</table>

### Task 1.2: Object Detection: Fine-tuning

The results obtained after applying the fine-tuning

<table>
    <thead>
        <tr>
            <th>Network</th>
            <th>Framework</th>
            <th>mIoU</th>
            <th>mAP50</th>
            <th>mAP70</th>
            <th>#Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Yolo V3</td>
            <td rowspan=3>Ultralytics</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Yolo V3 - SPP</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Yolo V3 - Tiny</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>SSD MN V1 FPN 640x640</td>
            <td rowspan=3>TensorFlow</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>SSD RN101 V1 FPN 640x640</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>FR-CNN RN101 V1 640x640</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>


### Task 1.3: K-fold Cross validation

Results obtained with the best method when applying K-fold Cross Validation.

| AP50 | alpha | rho |
| :---: | :---: | :---: |
| 0.51 | 1.6 | 0.0025 | 

### Task 2.1: Tracking by Overlap

Tracking results when it is used the Overlapping method.

| Threshold | 0.3 | 0.5 | 0.7 |
| :---: | :---: | :---: | :---: |
| IDF1 |  |  |  | 


### Task 2.2: Tracking with a Kalman Filter

Tracking results when it is used the Kalman method.

| Threshold | 0.3 | 0.5 | 0.7 |
| :---: | :---: | :---: | :---: |
| IDF1 | 63.05 | 62.82 | 61.61 | 


## Report
The report for week 3 is available [here](https://docs.google.com/presentation/d/1M0Vw8quKhlRDudc1A5YYByKr7KclVOnSPspYGjmRxCo/edit?usp=sharing).
