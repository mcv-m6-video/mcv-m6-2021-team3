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
            <th>Configuration</th>
            <th>Scene</th>
            <th>IDF1</th>
            <th>IDP</th>
            <th>IDR</th>
            <th>Precision</th>
            <th>Recall</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Detections: Mask-RCNN <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Car features: ResNet50</td>
            <td>&nbsp;&nbsp;S01</td>
            <td>0.2112</td>
            <td>0.2360</td>
            <td>0.2030</td>
            <td>&nbsp;&nbsp;0.4647</td>
            <td>0.4358</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S03</td>
            <td>0.4318</td>
            <td>0.3300</td>
            <td>0.6417</td>
            <td>&nbsp;&nbsp;0.3870</td>
            <td>0.7703</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S04</td>
            <td>0.2951</td>
            <td>0.2246</td>
            <td>0.4870</td>
            <td>&nbsp;&nbsp;0.3478</td>
            <td>0.7463</td>
        </tr>
        <tr>
            <td rowspan=3>Detections: YoloV3 (our training) <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Car features: ResNet50</td>
            <td>&nbsp;&nbsp;S01</td>
            <td>0.1248</td>
            <td>0.2828</td>
            <td>0.0876</td>
            <td>&nbsp;&nbsp;0.5647</td>
            <td>0.1979</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S03</td>
            <td>0.3314</td>
            <td>0.3073</td>
            <td>0.3597</td>
            <td>&nbsp;&nbsp;0.3653</td>
            <td>0.4275</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S04</td>
            <td>0.1968</td>
            <td>0.1973</td>
            <td>0.1962</td>
            <td>&nbsp;&nbsp;0.3954</td>
            <td>0.3932</td>
        </tr>
        <tr>
            <td rowspan=3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Detections: Mask-RCNN <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Car features: VGG16</td>
            <td>&nbsp;&nbsp;S01</td>
            <td>0.3377</td>
            <td>0.2654</td>
            <td>0.4639</td>
            <td>&nbsp;&nbsp;0.4385</td>
            <td>0.7664</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S03</td>
            <td>0.4599</td>
            <td>0.3157</td>
            <td>0.8467</td>
            <td>&nbsp;&nbsp;0.3446</td>
            <td>0.9243</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S04</td>
            <td>0.2778</td>
            <td>0.1946</td>
            <td>0.4851</td>
            <td>&nbsp;&nbsp;0.3260</td>
            <td>0.8126</td>
        </tr>
        <tr>
            <td rowspan=3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Detections: Mask-RCNN <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Car features: ResNe101</td>
            <td>&nbsp;&nbsp;S01</td>
            <td>0.1992</td>
            <td>0.2083</td>
            <td>0.1908</td>
            <td>&nbsp;&nbsp;0.4452</td>
            <td>0.4079</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S03</td>
            <td>0.4334</td>
            <td>0.3180</td>
            <td>0.6799</td>
            <td>&nbsp;&nbsp;0.3848</td>
            <td>0.8226</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S04</td>
            <td>0.2430</td>
            <td>0.1771</td>
            <td>0.3870</td>
            <td>&nbsp;&nbsp;0.3198</td>
            <td>0.6987</td>
        </tr>
        <tr>
            <td rowspan=3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Detections: Mask-RCNN <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Car features: MobileNet</td>
            <td>&nbsp;&nbsp;S01</td>
            <td>0.2002</td>
            <td>0.2086</td>
            <td>0.1924</td>
            <td>&nbsp;&nbsp;0.4453</td>
            <td>0.4107</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S03</td>
            <td>0.4349</td>
            <td>0.3235</td>
            <td>0.6631</td>
            <td>&nbsp;&nbsp;0.3921</td>
            <td>0.8036</td>
        </tr>
        <tr>
            <td>&nbsp;&nbsp;S04</td>
            <td>0.2420</td>
            <td>0.1782</td>
            <td>0.3767</td>
            <td>&nbsp;&nbsp;0.3245</td>
            <td>0.6858</td>
        </tr>
    </tbody>
</table>

<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/af963ca82b75a6332d701df067bba98a36594f96/Week3/Task12.jpg" width="700">


| Parameters| Interpolation = Off  Denoise = Off | Interpolation = True  Denoise = Off | Interpolation = Off  Denoise = True | Interpolation = True Denoise = True |
| :---: | :---: | :---: | :---: | :---: |
| IDF1 | 61.22 | 61.67 | 61.81 | 63.61 | 

<p align="center">
<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/4c2e0306c7a1e52557fe0b9abf984e0885bd8eef/Week3/m6w3.gif" width="700">
</p>

## Report
The report for week 5 is available [here](https://docs.google.com/presentation/d/1hDOKs9FtG4Ze7O4RaPLsKDmh25QD2bDDGGLuZz_B0QU/edit?usp=sharing).
