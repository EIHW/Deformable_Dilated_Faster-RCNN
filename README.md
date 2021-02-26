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
