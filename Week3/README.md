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
|   |   |   ├── utils.py
|   |   |   ├── detect2.py
|   |   |   ├── tf_models.py
|   |   |   ├── yolov3.py
|   |   |   ├── sort.py
│   │   ├── data
│   │   ├── models
│   │   ├── yolov3
│   │   ├── runs
│   │   ├── config
│   │   │   ├── config.py
```

## Results
### Task 1.1: Off-the-shelf


### Task 1.2: Fine-tuning

The results obtained after applying the fine-tuning

| alpha | 1 | 1.5 | 2 | 3 | 4 | 5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AP50 250 frames | 0.119 | 0.168 | 0.240 | 0.409 | 0.171 | 0.100 |
| AP50 all frames | 0.0146 | 0.1424 | 0.1426 | 0.1201 | 0.0166 | 0.0132 |

### Task 1.3: K-fold Cross validation

Results obtained with the best method when applying K-fold Cross Validation.

| AP50 | alpha | rho |
| :---: | :---: | :---: |
| 0.51 | 1.6 | 0.0025 | 

### Task 2.1: Tracking by Overlap

Tracking results when it is used the Overlap method.


### Task 2.2: Tracking with a Kalman Filter

Tracking results when it is used the Kalman method.


### Task 2.3: IDF1 for Multiple Object Tracking

The IDF1 results obtained when using both tracking methods.



<table>
    <thead>
        <tr>
            <th colspan=2></th>
            <th>KNN</th>
            <th>MOG2</th>
            <th>GMG</th>
            <th>LSBP</th>
            <th>Ours</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>AP50</td>
            <td>250 frames</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.3577</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.3937</td>
            <td>&nbsp;0.1554</td>
            <td>&nbsp;0.0013</td>
            <td>&nbsp;&nbsp;0.51</td>
        </tr>
        <tr>
            <td>&nbsp;All frames</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.3307</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.2544</td>
            <td>&nbsp;0.0018</td>
            <td>&nbsp;0.0210</td>
            <td>&nbsp;&nbsp;0.23</td>
        </tr>
        <tr>
            <td colspan=2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Configuration</td>
            <td>&nbsp;Shadows detector <br> &nbsp;&nbsp;&nbsp;threshold = 60</td>
            <td>&nbsp;Shadows detector <br> &nbsp;&nbsp;&nbsp;threshold = 12</td>
            <td>Default</td>
            <td>Default</td>
            <td>Task 2</td>
        </tr>
    </tbody>
</table>


### Task 4: Color sequences

The results obtained using different color spaces (HSV,LAB,YCbCr) for the non-adaptive and the adaptive methods.

<table>
    <thead>
        <tr>
            <th colspan=3></th>
            <th>LAB</th>
            <th>HSV</th>
            <th>YCbCr</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=6>AP50</td>
            <td rowspan=2>&nbsp;&nbsp;&nbsp;&nbsp;Adaptive</td>
            <td>150 frames</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.291</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.015</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.334</td>
        </tr>
        <tr>
            <td>&nbsp;All frames</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.112</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.004</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.114</td>
        </tr>
        <tr>
            <td colspan=2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters</td>
            <td>alpha = 1.43 <br> &nbsp;rho = 0.001</td>
            <td>alpha = 1.71 <br> &nbsp;rho = 0.001</td>
            <td>alpha = 1.75 <br> &nbsp;rho = 0.03</td>
        </tr>
        <tr>
            <td rowspan=2>Non-adaptive</td>
            <td>150 frames</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.488</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.027</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.400</td>
        </tr>
        <tr>
            <td>&nbsp;All frames</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.193</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.014</td>
            <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.212</td>
        </tr>
        <tr>
            <td colspan=2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameters</td>
            <td>alpha = 1.43</td>
            <td>alpha = 1.71</td>
            <td>alpha = 1.75</td>
        </tr>
    </tbody>
</table>


## Report
The report for week 3 is available [here](https://docs.google.com/presentation/d/1M0Vw8quKhlRDudc1A5YYByKr7KclVOnSPspYGjmRxCo/edit?usp=sharing).
