# Week 5

## Introduction
In this project the main goal was to do Multi-target multi-camera tracking, using the previous knowledge on tracking.

* [main.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/main.py): Contains the pipeline to execute the different tasks.
* [ai_city.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/datasets/ai_city.py): Contains the class AICity, where the information of each sequence is prepared. 
* [load_seq.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/datasets/load_seq.py): Contains the class LoadSeq, where the differents Sequences are processed.
* [metrics.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/utils/metrics.py): Contains functions related to get quantitative results.
* [utils.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/utils/utils.py): Contains other functions, such as the one to write a json or read one.
* [visualize.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/utils/visualize.py): Contains the functinos related to plot the different (qualitative) results.
* [cnn_feature_extractor.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/utils/cnn_feature_extractor.py): Contains the class CNNFeatureExtractor which load a pretrained PyTorch model, computes a feature vector of an image and compute the distance between 2 feature vectors in order to realise a matching.
* [multitracking.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/modes/multitracking.py): Contains the functions required to apply the multi-camera multi-tracking.
* [optical_flow.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/modes/optical_flow.py): Contains the functions used to apply the differents algorithms of optical flow estimation.
* [tracking.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/modes/tracking.py): Contains the functions required to apply the methods of single-camera tracking.
* [sort.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/modes/sort.py): Contains the class Sort, which is the one related to compute the Kalman tracking.
* [ultralytics_yolo.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/modes/ultralytics_yolo.py): Contains the class UltralyricsYolo, which contains the funtion to predict and train the Ultralytics Yolo model. 



## Inference
### Available tasks
* **Task 1**: Multi-Target Single-Camera (MTSC) Tracking
* **Task 2**: Multi-Target Multi-Camera (MTMC) Tracking

Run the command below, from Week5/src folder, to obtain the desired task results.

```bash
$ python main.py 
```
Any parameter could be modified in file [Config.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week5/src/config/config.py)

### Directory structure

```bash
├── raw_data
│   ├── AICity
│   │   ├── train
│   │   │   ├── S01
│   │   │   │   ├── c001
│   │   │   │   │   ├── vdo
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   │   ├── c005
│   │   │   │   │   ├── vdo
│   │   │   ├── S03
│   │   │   │   ├── c010
│   │   │   │   │   ├── vdo
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   │   ├── c015
│   │   │   │   │   ├── vdo
│   │   │   ├── S04
│   │   │   │   ├── c016
│   │   │   │   │   ├── vdo
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   │   ├── c040
│   │   │   │   │   ├── vdo
├── Week5
│   ├── src
│   │   ├── main.py
│   │   ├── utils
│   │   │   ├── metrics.py
│   │   │   ├── visualize.py
│   │   │   ├── utils.py
│   │   ├── datasets
│   │   │   ├── ai_city.py
│   │   │   ├── load_seq.py
│   │   ├── modes
│   │   │   ├── optical_flow.py
│   │   │   ├── tracking.py
│   │   │   ├── multitracking.py
│   │   │   ├── tf_models.py
│   │   │   ├── ultralytics_yolo.py
│   │   │   ├── sort.py
│   │   ├── pyflow
│   │   ├── MaskFlownet
│   │   ├── SelFlow
│   │   ├── yolov3
│   │   ├── config
│   │   │   ├── config.py
│   │   │   ├── config_multitracking.py
│   │   │   ├── mask_flownet_config.py
```



## Results
### Task 1: Multi-Target Single-Camera (MTSC) Tracking

Quantitative results obtained after applying MTSC Tracking on those detections obtained after evaluate the model trained.

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

<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/af963ca82b75a6332d701df067bba98a36594f96/Week3/Task11.jpg" width="700">

### Task 2: Multi-Target Multi-Camera (MTMC) Tracking

Quantitative results obtained after applying MTMC Tracking on those detections obtained after evaluate the model trained.

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

<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/af963ca82b75a6332d701df067bba98a36594f96/Week3/Task12.jpg" width="700">

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


| Parameters| Interpolation = Off  Denoise = Off | Interpolation = True  Denoise = Off | Interpolation = Off  Denoise = True | Interpolation = True Denoise = True |
| :---: | :---: | :---: | :---: | :---: |
| IDF1 | 61.22 | 61.67 | 61.81 | 63.61 | 

<p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/4c2e0306c7a1e52557fe0b9abf984e0885bd8eef/Week3/m6w3.gif" width="700">
</p>

## Report
The report for week 5 is available [here](https://docs.google.com/presentation/d/1hDOKs9FtG4Ze7O4RaPLsKDmh25QD2bDDGGLuZz_B0QU/edit?usp=sharing).
