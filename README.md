# Deformable Dilated Faster-RCNN for Universal Lesion Detection in CT Images

## Requirements
The following software needs to be installed on the system to setup the project:
* GCC (version 5.0.0 or greater)
* NVCC (Cuda Toolkit 10.2) 
* Python (version 3.7.x)

## Setup
The following steps are required to setup the network:

1. Install the required pip packages
    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/102/torch_stable.html
    ```
2. Compile the program
    ```bash
    python setup.py install develop
    ```
3. Download and extract the relevant data for training (raw data is ~200GB + extracted data is ~21GB)
    ```bash
    python __main__.py data_extraction
    ```
4. Run the training, generate reports and previews
    ```bash
    python __main__.py run
    ```

## Features
### Faster R-CNN
* [x] deformable convolution
* [x] deformable ROI pooling
* [x] deformable convolution and ROI pooling
* [x] dilated kernel using different rate values


## License
MIT License

Copyright (c) 2021 Fabio Hellmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.