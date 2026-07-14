# Detection-based remote control of a wheeled robot with a camera

# Overview
This project consists of a vision-based remote controller for wheeled robots. The setup uses a mounted webcam to observe a Limo robot. Position and orientation and obtained through a FasterRCNN, which the system will then use to provide commands to the robot to follow a given path. The goal for this project is to acheive real-time control with this vision-based remote controller.

# Current Status
See https://mooncyli.blogspot.com/ for detailed daily logs
Current Progress: I
- Implemented FasterRCNN with MobileNet V2 backbone
- Implemented custom detection heads and losses for position and orientation
- Tested with epfl dataset
    - Did not perform well, dataset not suited for this task
- Tested with synthetic dataset with lineart of Limo robot
    - Did not perform well, lineart may not be suitable
- Created functions to extract ground truth annotations from april tags
    - Will be used to create training and validation dataset
- Created function to predict and display position and orientation from live webcam feed
- Implemented simple testing model with only MobileNet V2 backbone
- Created new synthetic dataset with real picture of Limo robot with simpler classes
- Changed loss to be a combination of cross entropy loss and custom loss

# Repository Structure
BU-RISE/
|
|--april_tags                   # functions to interpret apriltag data into ground truth annotations
|--epfl                         # training model with epfl dataset
|--graphs/                      # graphs from training
|--library_model_functions/     # imported functions for model
|--lineart_limo                 # training model with synthetic lineart limo dataset
|--model/                       # model architecture for position and orientation prediction with faster rcnn
|--simple_model/                # simple testing model with only backbone
|--simple_testing/              # testing simple model with 16 position classes and 4 orientation classes
|--unit_tests/                  # unit tests for each head of model
|--util/                        # utility functions
|--main.py                      # runs all code
|--.gitattributes               # tells git to commit any saved models as large files
|--requirements.txt             # for downloading requirements
|--README.md                    # information about project
|--config.py                    # global variables


# Installation
1. Clone the repository
'''git clone https://github.com/mooncyli-burise/BU-RISE.git'''

2. Install dependencies
'''pip install -r requirements.txt'''

# Datasets
EPFL Multi-view Car Dataset
https://www.epfl.ch/labs/cvlab/data/data-pose-index-php/

Apriltag test_files from lib-dt-apriltags
https://github.com/duckietown/lib-dt-apriltags/tree/977b18ae778b3c631ab9979f14f74fe5463c08da/test/test_files

# References
https://arxiv.org/pdf/1702.01499
https://link.springer.com/article/10.1007/s10462-024-10721-6#Sec17
https://github.com/duckietown/lib-dt-apriltags/tree/977b18ae778b3c631ab9979f14f74fe5463c08da
https://github.com/pytorch/pytorch/tree/v2.12.0
https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html