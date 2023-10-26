# binary-video-segmentation

## Table of Contents

- [Requirements](#Requirements)
- [Description](#Description)
- [Installation](#Installation)
- [Usage](#usage)
- [License](#license)


## Requirements

This project aims at performing a segmentation of each frame of 4 given videos using two possible methods, one based on multiple OpenCv tools, and one on the OpenCV's Watershed implementation.
For each frame, representing an object on a turn-table, both algorithms will remove the background, retaining only object and table, then saving each segmented frame in a new video.
The code also includes an evaluation of the results through the use of a confusion matrix and other metrics.

## Description:

The **main.ipynb** jupyter notebook hosts the main, which interfaces the two segmentation algorithms, using the chosen one over the chosen video.
The files **watershedOpenCV** and **customAlgorithm** as hinted by their names host the related segmentation algorithms.

## Installation

Before of running the main notebook, there are some requirements:

- Python 3.
- The following python modules:
  - matplotlib
  - opencv-python
  - numpy
  - sklearn

## Usage

The main notebook can be run from any IDE which supports Jupyter Notebooks.



```bash
git clone https://github.com/elaaj/binary-video-segmentation
```


## License

MIT License

Copyright (c) 2022 Elsa Sejdi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
