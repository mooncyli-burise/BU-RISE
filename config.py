DATA_DIR = "/home/roboticslab/BU-RISE/"

CENTER_LOSS_WEIGHT = 1.0
ORIENTATION_LOSS_WEIGHT = 0.01

HEIGHT = 120 #px
WIDTH = 160 #px
CAMERA_PARAMS = (
    615.0, # fx
    615.0, # fy
    WIDTH/2, # cx
    HEIGHT/2 # cy
)
TAG_SIZE = 0.15 #tag size in meters
TEST_SIZE = 50

POSE_WEIGHT = 2.0
X_CLASSES = 4
Y_CLASSES = 4
ANGLE_CLASSES = 72
#TODO: oh shoot i need to retest everything because i havent been regeneratign the data each time oops...
TOTAL_CLASSES = X_CLASSES*Y_CLASSES*ANGLE_CLASSES
DIMENSIONS = '4x4x72'