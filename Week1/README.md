# Week 1

## Introduction
In this project the main goal was to get used to the different datasets and the metrics that will be used in future weeks. The project contains several source code files that implement the required functionality.

* [main.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/main.py): Contains the pipeline to execute the different tasks.
* [datasets.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/datasets.py): Contains functions related to reading and organizing the information from the different datasets.
* [visualize.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/visualize.py): Gets functions to plot the different results. 
* [utils.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/utils.py): Has other functions, such as one related to adding noise to the bounding boxes. 
* [metrics.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/metrics.py): Contains function to evaluate the performance of the method.
* [voc_eval.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/voc_eval.py): Contains the funcions to compute de mAP.


## Inference
### Available tasks
* **Task 1.1**: IoU & mAP for (ground truth + noise).
* **Task 1.2**: mAP for provided object detections (Mask RCNN, SDD-512, YOLOv3).
* **Task 2**: Temporal analysis (IoU vs time).
* **Task 3**: Optical flow evaluation metrics (MSEN & PEPN).
* **Task 4**: Visual representation optical flow.

Run the command below to obtain the desired task results.

```bash
$ python main.py ${TASK_NUMBER}
```

### Directory structure

```bash
├── AICity_data
│   ├── train
│   │   ├── S03
│   │   │   ├── c010
│   │   │   │   ├── vdo
│   │   │   │   ├── det
├── data_stereo_flow
│   ├── training
│   │   ├── flow_noc
├── results_opticalflow_kitti
│   ├── results
├── ai_challenge_s03_c010-full_annotation.xml
├── Week1
│   ├── main.py
│   ├── datasets.py
│   ├── visualize.py
│   ├── utils.py
│   ├── metrics.py
│   ├── voc_eval.py
```

## Results
### Task 1.1: IoU & mAP for (ground truth + noise)

### Task 1.2: Mean Average Precision (mAP) for Mask RCNN, SDD-512 and YOLOv3


| <center>**Model Detector**</center> | <center>**mAP**</center> | <center>**mAP(%)**</center> |
| :---: | :---: | :---: |
| Mask RCNN | <center>0.419</center> | <center>41.9</center> |
| SDD-512 | <center>0.367</center> | <center>36.7</center> |
| YOLOv3 | <center>0.416</center> | <center>41.6</center> |

### Task 2: Temporal analysis (IoU vs time)

In terms of temporal analysis we have used the IoU over time. Displayed below the results of each detector:

| <center>**Mask RCNN**</center> | <center>**SDD-512**</center> | <center>**YOLOv3**</center> |
| :---: | :---: | :---: |
| ![det_mask_rcnn](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/task2/det_mask_rcnn.gif) | ![det_ssd512](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/task2/det_ssd512.gif) | ![det_yolo3](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/task2/det_yolo3.gif) |

### Task 3: Optical flow evaluation metrics (MSEN & PEPN)

| :-<center>**Model Detector**</center>-: | :-<center>**mAP**</center>-: |
| :---: | :---: | :---: | :---: |
| Seq 45 | Seq 45 | Seq 45 | Seq 45 |

## Report
The report for week 1 is available [here](https://docs.google.com/presentation/d/1fW_KEDz9zGoJzBtoJuXenhzcsG9WRU2GkyU0DSTTnB4/edit?usp=sharing).
