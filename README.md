# mcv-m6-2021-team3

## Week 1

### Introduction
In this project the main goal was to get used to the different datasets and the metrics that will be used in future weeks. The project contains several source code files that implement the required functionality.

* main.py: Contains the pipeline to execute the different tasks.
* datasets.py: Contains functions related to reading and organizing the information from the different datasets.
* visualize.py: Gets functions to plot the different results. 
* utils.py: Has other functions, such as one related to adding noise to the bounding boxes. 
* metrics.py: Contains function to evaluate the performance of the method.

# Team 6 

| Members | Contact |
| :---         |   :---    | 
| Gemma Alaix   | gemma.alaix@e-campus.uab.cat | 
| Josep Brugués    | josep.brugues@e-campus.uab.cat  |
| Clara Garcia    | clara.garciamo@e-campus.uab.cat  |
| Aitor Sánchez | aitor.sancheza@gmail.com |

## Requirements

Python 3.7 or later with all [requirements.txt](https://https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/requirements.txt) dependencies installed. To install run:
```bash
$ pip install -r requirements.txt
```

## Available tasks
* **Task 1.1**: IoU & mAP for (ground truth + noise).
* **Task 1.2**: mAP for provided object detections (Mask RCNN, SDD-512, YOLO-v3). 
* **Task 2**: Temporal analysis (IoU vs time).
* **Task 3**: Optical flow evaluation metrics (MSEN & PEPN).
* **Task 4**: Visual representation optical flow.
