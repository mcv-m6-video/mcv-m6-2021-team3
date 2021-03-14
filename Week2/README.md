# Week 2

## Introduction
In this project the main goal was to get used to different background estimation methods. The project contains several source code files that implement the required functionality.

* [main.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/main.py): Contains the pipeline to execute the different tasks.
* [ai_city.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/ai_city.py): Contains the class AICity, where the data is processed in order to obtain the background estimation.
* [metrics.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/metrics.py):
* [refinement.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/refinement.py): Contains the functions related to the post processing of the background and the bounding box generation.


## Inference
### Available tasks
* **Task 1.1**: Gaussian distribution.
* **Task 1.2**: Evaluate results (mAP).
* **Task 2.1**: Recursive Gaussian modeling.
* **Task 2.2**: Evaluate and compare to non-recursive.
* **Task 3**: Compare with state-of-the-art.
* **Task 4**: Color sequences.


Run the command below, from Week2/src folder, to obtain the desired task results.

```bash
$ python main.py 
```

### Directory structure

```bash
├── data
│   ├── AICity
│   │   ├── train
│   │   │   ├── S03
│   │   │   │   ├── c010
│   │   │   │   │   ├── vdo
├── Week2
│   ├── src
│   │   ├── main.py
│   │   │   ├── utils
│   │   │   │   ├── ai_city.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── refinement.py
```

## Results
### Task 1.1: Gaussian distribution


### Task 1.2: Evaluate results (mAP)


### Task 2.1: Recursive Gaussian modeling


### Task 2.2: Evaluate and compare to non-recursive


### Task 3: Compare with state-of-the-art


### Task 4: Color sequences


## Report
The report for week 2 is available [here](https://docs.google.com/presentation/d/1q8MU8wWAj79WdowlBbMnYNEO6wsHu3VKxZdveiHwr_s/edit?usp=sharing).
