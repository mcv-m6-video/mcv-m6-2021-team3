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

Quantitative results obtained by applying Block Matching.

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

### Task 1.2: Off-the-shelf Optical Flow

Quantitative results obtained for the Off-the-shelf method compared with our best result.

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
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3029</td>
            <td>0.0799</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.1580</td>
        </tr>
    </tbody>
</table>

#### MaskFlownet Optical Flow
<p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/0c7ed3be7a713cf4af0af22dfbc8ae4e66f42eb0/Week4/MFnet12.png" width="500">
</p>

#### Pyflow Optical Flow
<p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/0c7ed3be7a713cf4af0af22dfbc8ae4e66f42eb0/Week4/Pyflow12.png" width="500">
</p>

### Task 2.1: Video stabilization with Block Matching

#### Stabilization using Block Matching
<p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/f289b74aa219dd975a1961436b3d5ec58136c422/Week4/src/task2.gif" width="700">
</p>

### Task 2.2: Off-the-shelf Stabilization

#### Stabilization using DUT
<p align="center">
    <img align="center" src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/d50850ce3a7e2bf67517876ff809686364c0915c/Week4/stabilization_dut.gif" width="700">
</p>

#### Stabilization using OpenCV
<p align="center">
    <img align="center" src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/d50850ce3a7e2bf67517876ff809686364c0915c/Week4/opencv2.gif" width="700">
</p>

### Task 3.1: Object Tracking with Optical Flow

<table>
    <thead>
        <tr>
            <th>Method</th>
            <th>Mask Flownet</th>
            <th>Block Matching</th>
            <th>Pyflow</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>IDF10.5</td>
            <td>&nbsp;&nbsp;&nbsp;65.8698%</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;63.3578%</td>
            <td>64.6187%</td>
        </tr>
    </tbody>
</table>

#### Tracking with Block Matching

<p align="center">
    <img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/b01d6c6d7666611fcb02f0b1ace7879e630c3a40/Week4/tracking-bm-edit-2.gif" width="700">
</p>

#### Tracking with MaskFlownet
<p align="center">
    <img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/c268c06bf013ba3af66b9278586c0607c3274def/Week4/tracking_mask_flownet.gif" width="700">
</p>

#### Tracking with Pyflow

<p align="center">
    <img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/24de1b72ac60445bd94359a4d68473664d899647/Week4/tracking_pyflow.gif" width="700">
</p>

## Report
The report for week 4 is available [here](https://docs.google.com/presentation/d/1JYmlzbrf8hvug4VpYjFmhxA6A4ILqHioQ-ZunkLy9V8/edit?usp=sharing).
