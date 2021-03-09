### Week 1

## Introduction
In this project the main goal was to get used to the different datasets and the metrics that will be used in future weeks. The project contains several source code files that implement the required functionality.

* main.py: Contains the pipeline to execute the different tasks.
* datasets.py: Contains functions related to reading and organizing the information from the different datasets.
* visualize.py: Gets functions to plot the different results. 
* utils.py: Has other functions, such as one related to adding noise to the bounding boxes. 
* metrics.py: Contains function to evaluate the performance of the method.

## Inference
# Available tasks
* **Task 1.1**: IoU & mAP for (ground truth + noise).
* **Task 1.2**: mAP for provided object detections (Mask RCNN, SDD-512, YOLO-v3).
* **Task 2**: Temporal analysis (IoU vs time).
* **Task 3**: Optical flow evaluation metrics (MSEN & PEPN).
* **Task 4**: Visual representation optical flow.

Run the command below to obtain the desired task results.

```bash
$ python main.py ${TASK_NUMBER}
```
