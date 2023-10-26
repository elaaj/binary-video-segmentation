# binary-video-segmentation

## Table of Contents

- [Requirements](#Requirements)
- [Description](#Description)
- [Installation](#Installation)
- [Usage](#usage)


## Requirements

This project aims at performing a segmentation of each frame of 4 given videos using two possible methods, one based on multiple OpenCv tools, and one on the OpenCV's Watershed implementation.
For each frame, representing an object on a turn-table, both algorithms will remove the background, retaining only object and table, then saving each segmented frame in a new video.
The code also includes an evaluation of the results through the use of a confusion matrix and other metrics.

## Description:

The **main.ipynb** jupyter notebook hosts the main, which interfaces the two segmentation algorithms, using the chosen one over the chosen video.
The Python files **watershedOpenCV.py** and **customAlgorithm.py** as hinted by their names host the related segmentation algorithms.

## Installation

Before of running the main notebook, there are some requirements:

- Python 3.
- The following python modules:
  - matplotlib
  - opencv-python
  - numpy
  - sklearn
 
Also, the "data" folder must be placed outside of the folder containing all the algorithms, or the path in the main file must be adapted to the new data location.

## Usage

The main notebook can be run from any IDE which supports Jupyter Notebooks.



```bash
git clone https://github.com/elaaj/binary-video-segmentation
```
