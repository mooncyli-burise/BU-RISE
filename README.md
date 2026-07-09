# Detection-based remote control of a wheeled robot with a camera

# Overview
The goal of this project is to design vision-based remote controllers for wheeled robots. The overall setup consists of a monocular camera observing a robot. The system should first detect the robot in the image, as well as its orientation (assuming a planar surface for simplicity), and then provide the commands to the robot to follow a given path. The pose detector and the controller should be fast enough for real-time control. 

# Current Status
Current Progress:

# Repository Structure
BU-RISE/
|
|--april_tag_test_data/         # apriltag dataset to test apriltag detectors
|--april_tags/                  # functions for detecting data from apriltags and creating ground truth list
|--epfl_dataset/                # 
|--epfl_dataset_functions/      # 
|--graphs/                      # 
|--library_model_functions/     # 
|--model/                       # 
|--util/                        # 
|--.gitattributes               # 
|--requirements.txt             # 
|--README.md                    # 
|--best_robot_detector.pth      # 
|--checkpoint.pth               # 
|--config.py                    # 
|--detect.py                    # 
|--train.py                     # 


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

# Training

# Inference

# References
https://arxiv.org/pdf/1702.01499
https://link.springer.com/article/10.1007/s10462-024-10721-6#Sec17
https://github.com/duckietown/lib-dt-apriltags/tree/977b18ae778b3c631ab9979f14f74fe5463c08da
https://github.com/pytorch/pytorch/tree/v2.12.0
https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html