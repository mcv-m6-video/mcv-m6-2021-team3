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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Yolo V3 640x640</td>
            <td rowspan=3>&nbsp;Ultralytics</td>
            <td>0.7456</td>
            <td>0.5247</td>
            <td>0.5188</td>
        </tr>
        <tr>
            <td>Yolo V3 SPP 640x640</td>
            <td>0.7262</td>
            <td>0.5532</td>
            <td>0.5339</td>
        </tr>
        <tr>
            <td>Yolo V3 Tiny 640x640</td>
            <td>0.7476</td>
            <td>0.5738</td>
            <td>0.5409</td>
        </tr>
        <tr>
            <td>SSD MN V1 FPN 640x640</td>
            <td rowspan=8>TensorFlow</td>
            <td>0.3911</td>
            <td>0.5782</td>
            <td>0.5655</td>
        </tr>
        <tr>
            <td>SSD MN V2 FPNLite 640x640</td>
            <td>0.3022</td>
            <td>0.6152</td>
            <td>0.5943</td>
        </tr>
        <tr>
            <td>SSD RN101 V1 FPN 640x640</td>
            <td>0.2267</td>
            <td>0.3753</td>
            <td>0.3687</td>
        </tr>
        <tr>
            <td>SSD RN152 V1 FPN 640x640</td>
            <td>0.2365</td>
            <td>0.4724</td>
            <td>0.4683</td>
        </tr>
        <tr>
            <td>EfficientDet D1 640x640</td>
            <td>0.2184</td>
            <td>0.4995</td>
            <td>0.5069</td>
        </tr>
        <tr>
            <td>FR-CNN RN101 V1 640x640</td>
            <td>0.256</td>
            <td>0.443</td>
            <td>0.4366</td>
        </tr>
        <tr>
            <td>CN RN101 V1 FPN 512x512</td>
            <td>0.2412</td>
            <td>0.4377</td>
            <td>0.4254</td>
        </tr>
        <tr>
            <td>MR-CNN Inception RN V2 1024x1024</td>
            <td>0.2943</td>
            <td>0.4405</td>
            <td>0.4499</td>
        </tr>
    </tbody>
</table>
 

![Image of Task 1.1](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/1a79886f4f0ec76c4f1315e6c3ca072040812fc8/Week3/Task11.jpg)

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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Yolo V3</td>
            <td rowspan=3>Ultralytics</td>
            <td>0.8866</td>
            <td>0.9445</td>
            <td>0.9065</td>
        </tr>
        <tr>
            <td>Yolo V3 - SPP</td>
            <td>0.8435</td>
            <td>0.8083</td>
            <td>0.8148</td>
        </tr>
        <tr>
            <td>Yolo V3 - Tiny</td>
            <td>0.8436</td>
            <td>0.8096</td>
            <td>0.8148</td>
        </tr>
        <tr>
            <td>SSD MN V1 FPN 640x640</td>
            <td rowspan=2>TensorFlow</td>
            <td>0.4929</td>
            <td>0.7842</td>
            <td>0.7585</td>
        </tr>
        <tr>
            <td>EfficientDet D1 640x640</td>
            <td>0.6914</td>
            <td>0.7965</td>
            <td>0.7899</td>
        </tr>
    </tbody>
</table>

![Image of Task 1.2](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/af963ca82b75a6332d701df067bba98a36594f96/Week3/Task12.jpg)

### Task 1.3: K-fold Cross validation

Results obtained with the best method when applying K-fold Cross Validation using as a model YOLOv3.

<table>
    <thead>
        <tr>
            <th colspan=4>Sorted</th>
            <th>mIoU</th>
            <th>mAP50</th>
            <th>mAP70</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>train</td>
            <td colspan=3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test</td>
            <td>0.8866</td>
            <td>0.9445</td>
            <td>0.9065</td>
        </tr>
        <tr>
            <td>test</td>
            <td>train</td>
            <td colspan=2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test</td>
            <td>0.9056</td>
            <td>0.9458</td>
            <td>0.9073</td>
        </tr>
        <tr>
            <td colspan=2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test</td>
            <td>train</td>
            <td>test</td>
            <td>0.8899</td>
            <td>0.9474</td>
            <td>0.9069</td>
        </tr>
        <tr>
            <td colspan=3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test</td>
            <td>train</td>
            <td>0.8921</td>
            <td>0.9558</td>
            <td>0.9075</td>
        </tr>
        <tr>
            <td colspan=4></td>
            <td>0.8935 <br> ±7e-3</td>
            <td>0.9484 <br> ±4e-3</td>
            <td>0.9070 <br> ±3e-4</td>
        </tr>
    </tbody>
</table>


<table>
    <thead>
        <tr>
            <th colspan=4>Shuffle</th>
            <th>mIoU</th>
            <th>mAP50</th>
            <th>mAP70</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>train</td>
            <td colspan=3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test</td>
            <td>0.9001</td>
            <td>0.9520</td>
            <td>0.9079</td>
        </tr>
        <tr>
            <td>test</td>
            <td>train</td>
            <td colspan=2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test</td>
            <td>0.9038</td>
            <td>0.9555</td>
            <td>0.9079</td>
        </tr>
        <tr>
            <td colspan=2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test</td>
            <td>train</td>
            <td>test</td>
            <td>0.9021</td>
            <td>0.9587</td>
            <td>0.9082</td>
        </tr>
        <tr>
            <td colspan=3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test</td>
            <td>train</td>
            <td>0.9038</td>
            <td>0.9512</td>
            <td>0.9081</td>
        </tr>
        <tr>
            <td colspan=4></td>
            <td>0.9025 <br> ±1e-3</td>
            <td>0.9543 <br> ±3e-3</td>
            <td>0.9080 <br> ±1e-4</td>
        </tr>
    </tbody>
</table>


### Task 2.1: Tracking by Overlap

Tracking results when it is used the Overlapping method using a Threshold of 0.5.

| Parameters| Interpolation = Off  Denoise = Off | Interpolation = True  Denoise = Off | Interpolation = Off  Denoise = True | Interpolation = True Denoise = True |
| :---: | :---: | :---: | :---: | :---: |
| IDF1 | 61.22 | 61.67 | 61.81 | 63.61 | 


### Task 2.2: Tracking with a Kalman Filter

Tracking results when it is used the Kalman method.

| Threshold | 0.3 | 0.5 | 0.7 |
| :---: | :---: | :---: | :---: |
| IDF1 | 63.05 | 62.82 | 61.61 | 

![Image of Task 2.2](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/be83276ec1d00bf487dac148199e9d530abbae13/Week3/Task22.jpg)


## Report
The report for week 3 is available [here](https://docs.google.com/presentation/d/1M0Vw8quKhlRDudc1A5YYByKr7KclVOnSPspYGjmRxCo/edit?usp=sharing).
