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

#### YoloV3:

<table>
    <thead>
        <tr>
            <th colspan=7>IDF1 S01</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Technique</td>
            <td>c001</td>
            <td>c002</td>
            <td>c003</td>
            <td>c004</td>
            <td>c005</td>
            <td>Avg</td>
        </tr>
        <tr>
            <td>Overlap</td>
            <td>0.4595</td>
            <td>0.5708</td>
            <td>0.3839</td>
            <td>0.4746</td>
            <td>0.2596</td>
            <td>0.4297</td>
        </tr>
        <tr>
            <td>Kalman</td>
            <td>0.5952</td>
            <td>0.6008</td>
            <td>0.4763</td>
            <td>0.5167</td>
            <td>0.3546</td>
            <td>0.5087</td>
        </tr>
        <tr>
            <td>Overlap + OF</td>
            <td>0.4017</td>
            <td>0.5790</td>
            <td>0.3842</td>
            <td>0.4745</td>
            <td>0.2587</td>
            <td>0.4196</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th colspan=8>IDF1 S03</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Technique</td>
            <td>c010</td>
            <td>c011</td>
            <td>c012</td>
            <td>c013</td>
            <td>c014</td>
            <td>c015</td>
            <td>Avg</td>
        </tr>
        <tr>
            <td>Overlap</td>
            <td>0.9391</td>
            <td>0.7990</td>
            <td>0.3185</td>
            <td>0.7742</td>
            <td>0.5632</td>
            <td>1.0000</td>
            <td>0.7323</td>
        </tr>
        <tr>
            <td>Kalman</td>
            <td>0.8236</td>
            <td>0.7347</td>
            <td>0.3536</td>
            <td>0.7708</td>
            <td>0.6988</td>
            <td>1.0000</td>
            <td>0.7303</td>
        </tr>
        <tr>
            <td>Overlap + OF</td>
            <td>0.9400</td>
            <td>0.7985</td>
            <td>0.3320</td>
            <td>0.7743</td>
            <td>0.5632</td>
            <td>1.0000</td>
            <td>0.7347</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th colspan=14>IDF1 S04</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Technique</td>
            <td>c016</td>
            <td>c017</td>
            <td>c018</td>
            <td>c019</td>
            <td>c020</td>
            <td>c021</td>
            <td>c022</td>
            <td>c023</td>
            <td>c024</td>
            <td>c025</td>
            <td>c026</td>
            <td>c027</td>
            <td>c028</td>
        </tr>
        <tr>
            <td>Overlap</td>
            <td>0.568</td>
            <td>0.681</td>
            <td>0.603</td>
            <td>0.940</td>
            <td>0.583</td>
            <td>0.364</td>
            <td>0.707</td>
            <td>0.932</td>
            <td>0.774</td>
            <td>0.685</td>
            <td>0.574</td>
            <td>0.407</td>
            <td>0.959</td>
        </tr>
        <tr>
            <td>Kalman</td>
            <td>0.446</td>
            <td>0.686</td>
            <td>0.602</td>
            <td>0.926</td>
            <td>0.429</td>
            <td>0.365</td>
            <td>0.551</td>
            <td>0.815</td>
            <td>0.840</td>
            <td>0.611</td>
            <td>0.744</td>
            <td>0.455</td>
            <td>0.962</td>
        </tr>
        <tr>
            <td>Overlap + OF</td>
            <td>0.568</td>
            <td>0.681</td>
            <td>0.603</td>
            <td>0.940</td>
            <td>0.583</td>
            <td>0.364</td>
            <td>0.707</td>
            <td>0.933</td>
            <td>0.733</td>
            <td>0.685</td>
            <td>0.573</td>
            <td>0.407</td>
            <td>0.958</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th colspan=14>IDF1 S04</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Technique</td>
            <td>c029</td>
            <td>c030</td>
            <td>c031</td>
            <td>c032</td>
            <td>c033</td>
            <td>c034</td>
            <td>c035</td>
            <td>c036</td>
            <td>c037</td>
            <td>c038</td>
            <td>c039</td>
            <td>c040</td>
            <td>Avg</td>
        </tr>
        <tr>
            <td>Overlap</td>
            <td>0.262</td>
            <td>0.447</td>
            <td>0.818</td>
            <td>0.612</td>
            <td>0.155</td>
            <td>0.202</td>
            <td>0.279</td>
            <td>0.211</td>
            <td>0.346</td>
            <td>0.414</td>
            <td>0.784</td>
            <td>0.679</td>
            <td>0.559</td>
        </tr>
        <tr>
            <td>Kalman</td>
            <td>0.316</td>
            <td>0.509</td>
            <td>0.554</td>
            <td>0.534</td>
            <td>0.192</td>
            <td>0.220</td>
            <td>0.361</td>
            <td>0.203</td>
            <td>0.330</td>
            <td>0.443</td>
            <td>0.791</td>
            <td>0.638</td>
            <td>0.541</td>
        </tr>
        <tr>
            <td>Overlap + OF</td>
            <td>0.262</td>
            <td>0.447</td>
            <td>0.818</td>
            <td>0.612</td>
            <td>0.155</td>
            <td>0.202</td>
            <td>0.280</td>
            <td>0.211</td>
            <td>0.357</td>
            <td>0.414</td>
            <td>0.782</td>
            <td>0.679</td>
            <td>0.558</td>
        </tr>
    </tbody>
</table>

<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/aad7c559e713213e909fef2b60c4fa4a4a4645b4/Week5/MTSC_agvYolo.png" width="700" align="center">

Some qualitative results are presented below:

<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/4ab87fccba6e855b719a87fdb2f8ecbd43aba3a9/Week5/MTSC_results.gif" width="700" align="center">


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

<img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/aad7c559e713213e909fef2b60c4fa4a4a4645b4/Week5/MTMC.png" width="700" align="center">

Some qualitative results for MTMC are presented next:

<p align="center">
    <img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/b0295c0e7d8bf2d594924f3cd9866f176f51aff1/Week5/mt_c013.gif" width="700">
    <img src="https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/b0295c0e7d8bf2d594924f3cd9866f176f51aff1/Week5/mt_c010.gif" width="700">
</p>

## Report
The report for week 5 is available [here](https://docs.google.com/presentation/d/1hDOKs9FtG4Ze7O4RaPLsKDmh25QD2bDDGGLuZz_B0QU/edit?usp=sharing).
