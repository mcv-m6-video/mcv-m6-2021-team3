# Week 4

## Introduction
In this project the main goal was to get used to different Optical Flow methods, use this method and others to stabilize a video and finally use OF to improve the tracking method. The project contains several source code files that implement the required functionality.

* [main.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/main.py): Contains the pipeline to execute the different tasks.
* [ai_city.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/datasets/ai_city.py): Contains the class AICity, where the data is processed in order to obtain the tracking and the detections.
* [kitti.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/datasets/kitti.py): Contains the class KITTI, where the data is processed in order to obtain the .
* [load_seq.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/datasets/load_seq.py): Contains the class LoadSeq, where the data is processed in order to obtain the video stabilization.
* [metrics.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/utils/metrics.py): Contains functions related to get quantitative results.
* [utils.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/utils/utils.py): Contains other functions, such as the one to write a json or read one.
* [visualize.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/utils/visualize.py): Contains the functinos related to plot the different (qualitative) results.
* [optical_flow.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/models/optical_flow.py): Contains the functions with the different methods of Optical Flow.
* [stabilize.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/models/stabilize.py): Contains the functions of the different methods of video stabilization.
* [tracking.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/models/tracking.py): Contains the functions related to the different methods of tracking.


## Inference
### Available tasks
* **Task 1.1**: Optical Flow with Block Matching
* **Task 1.2**: Off-the-shelf Optical Flow
* **Task 2.1**: Video stabilization with Block Matching
* **Task 2.2**: Off-the-shelf Stabilization
* **Task 3.1**: Object Tracking with Optical Flow


Run the command below, from Week4/src folder, to obtain the desired task results.

```bash
$ python main.py 
```
Any parameter could be modified in file [Config.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/config/config.py)
Any parameter when using MaskFlownet could be modified in file [Mask_flownet_config.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week4/src/config/mask_flownet_config.py)

### Directory structure

```bash
├── raw_data
│   ├── AICity
│   │   ├── train
│   │   │   ├── S03
│   │   │   │   ├── c010
│   │   │   │   │   ├── vdo
│   ├── data_stereo_flow
│   │   ├── training
│   │   │   ├── image_0
│   ├── video_stabilization
│   │   ├── flowers
│   │   │   ├── flowers_01
├── Week4
│   ├── src
│   │   ├── main.py
│   │   ├── utils
│   │   │   ├── metrics.py
│   │   │   ├── visualize.py
│   │   │   ├── utils.py
│   │   ├── datasets
│   │   │   ├── ai_city.py
│   │   │   ├── kitti.py
│   │   │   ├── load_seq.py
│   │   ├── models
│   │   │   ├── optical_flow.py
│   │   │   ├── stabilize.py
│   │   │   ├── tracking.py
│   │   │   ├── yolov3.py
│   │   │   ├── sort.py
│   │   ├── pyflow
│   │   ├── MaskFlownet
│   │   ├── SelFlow
│   │   ├── yolov3
│   │   ├── DUTCode
│   │   ├── config
│   │   │   ├── config.py
│   │   │   ├── mask_flownet_config.py
```

## Results
### Task 1.1: Optical Flow with Block Matching

Quantitative results obtained after applying inference on different models.

<table>
    <thead>
        <tr>
            <th colspan=5>Explored values</th>
            <th></th>
            <th colspan=2>Best Configuration</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>FWD/BWD compensation</td>
            <td>Area of search</td>
            <td>Block size</td>
            <td>Step size</td>
            <td>Error function</td>
            <td>Others</td>
            <td>PEPN</td>
            <td>MSEN</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SSD</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NCC</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SAD</td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>

<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/af963ca82b75a6332d701df067bba98a36594f96/Week3/Task11.jpg" width="700">

### Task 1.2: Off-the-shelf Optical Flow

Quantitative results obtained after doing fine-tuning on differents models.

<table>
    <thead>
        <tr>
            <th colspan=2>MSEN</th>
            <th colspan=2>PEPN</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Pyflow</td>
            <td>Best method</td>
            <td>Pyflow</td>
            <td>Best method</td>
        </tr>
        <tr>
            <td>0.9746</td>
            <td></td>
            <td>0.0799</td>
            <td></td>
        </tr>
    </tbody>
</table>

<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/af963ca82b75a6332d701df067bba98a36594f96/Week3/Task12.jpg" width="700">

### Task 2.1: Video stabilization with Block Matching


<p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/4c2e0306c7a1e52557fe0b9abf984e0885bd8eef/Week3/m6w3.gif" width="700">
</p>

### Task 2.2: Off-the-shelf Stabilization

<p align="center">
    <img align="center" src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/be83276ec1d00bf487dac148199e9d530abbae13/Week3/Task22.jpg" width="500">
    <img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/4c2e0306c7a1e52557fe0b9abf984e0885bd8eef/Week3/m6w3_clara.gif" width="700">
</p>

### Task 3.1: Object Tracking with Optical Flow

<table>
    <thead>
        <tr>
            <th>Method</th>
            <th>Mask Flownet</th>
            <th>Block Matching</th>
            <th>Block Matching</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>IDF10.5</td>
            <td rowspan=3>&nbsp;&nbsp;&nbsp;63.8698%</td>
            <td>OpenCV: 61.7942% </td>
            <td rowspan=3>&nbsp;&nbsp;&nbsp;62.6187%</td>
        </tr>
        <tr>
            <td>SSD:</td>
        </tr>
        <tr>
            <td>SAD:</td>
        </tr>
    </tbody>
</table>

## Report
The report for week 4 is available [here](https://docs.google.com/presentation/d/1JYmlzbrf8hvug4VpYjFmhxA6A4ILqHioQ-ZunkLy9V8/edit?usp=sharing).
