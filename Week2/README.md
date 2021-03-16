# Week 2

## Introduction
In this project the main goal was to get used to different background estimation methods. The project contains several source code files that implement the required functionality.

* [main.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/main.py): Contains the pipeline to execute the different tasks.
* [ai_city.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/ai_city.py): Contains the class AICity, where the data is processed in order to obtain the background estimation.
* [grid_search.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week2/src/utils/gridd_search.py): Contains the function to do a grid search in order to find the best values for the different parameters.
* [metrics.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/metrics.py): Contains functions related to get quantitative results.
* [refinement.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/refinement.py): Contains the functions related to the post processing of the background and the bounding box generation.
* [utils.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/utils.py): Contains functions other functions, such as the one to write a json or read one.
* [visualize.py](https://github.com/mcv-m6-video/mcv-m6-2021-team3/tree/main/Week2/src/utils/visualize.py): Contains the functinos related to plot the different results(qualitative).


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
$ python main.py ${TASK_NUMBER}
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

The results obtained from the non-adaptive grayscale method using different alphas in order to fix the threshold.

| alpha | 1 | 1.5 | 2 | 3 | 4 | 5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AP50 |  |  |  |  |  |  |
| Configuration |  |  |  |  |  |  |


### Task 2.1: Recursive Gaussian modeling

Results obtained by the best parameters (alpha and rho) using the adaptive grayscale method.

| AP50 | alpha | rho |
| :---: | :---: | :---: |
| 0.51 | 1.6 | 0.0025 | 


### Task 2.2: Evaluate and compare to non-recursive

The comparisson between the best result in Task 1 and in Task 2.1.

|  | AP50 | alpha | rho |
| :---: | :---: | :---: | :---: |
| Adaptive | 0.51 | 1.6 | 0.025 |
| Non-adaptive | 0.43 | 2.6 | 0 |


### Task 3: Compare with state-of-the-art

The results obtained from the different OpenCV background estimation methods in comparisson with the best result obtained by us.

<table>
    <thead>
        <tr>
            <th></th>
            <th></th>
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
            <td>0.3577</td>
            <td>0.3937</td>
            <td>0.1554</td>
            <td>0.0013</td>
            <td></td>
        </tr>
        <tr>
            <td>All frames</td>
            <td>0.2025</td>
            <td>0.1544</td>
            <td>0.0008</td>
            <td>0.021</td>
            <td></td>
        </tr>
        <tr>
            <td colspan=2>Configuration</td>
            <td>Shadows detector <br> threshold = 60</td>
            <td>Shadows detector <br> threshold = 12</td>
            <td>Default</td>
            <td>Default</td>
            <td>OURS</td>
        </tr>
    </tbody>
</table>


### Task 4: Color sequences


## Report
The report for week 2 is available [here](https://docs.google.com/presentation/d/1q8MU8wWAj79WdowlBbMnYNEO6wsHu3VKxZdveiHwr_s/edit?usp=sharing).
