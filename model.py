#modifying model to add diff backbone
import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from robot_pred import RobotPredictor

#backbone that extracts features
backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features

#number of feature maps created by this backbone (1280 for mobilenet v2)
backbone.out_channels = 1280

#specifies the base side lengths and aspect ratios to use for anchor generator (5x3=15 total different sizes)
anchor_generator = AnchorGenerator(
      sizes=((32,64,128,256,512),),
             aspect_ratios=((0.5,1.0,2.0),)
)

#extracts features within the bounding box
#['0'] because features only captured at one scale
#output_size determines size of feature map (7x7)
#sampling_ratio determines size of area used to aggregate the feature map
#for each bin/feature map in the 7x7 feature map into one feature
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

#initialize model with values
model = FasterRCNN(
    backbone,
    num_classes=2,  
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler\
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.pose_head = RobotPredictor(in_features)

# print(model.roi_heads)
# print(model.roi_heads.box_predictor)