import torch
import torch.nn as nn

class RobotPredictor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.center = nn.Linear(in_features, 2)
        self.orientation = nn.Linear(in_features, 72)

    def forward(self, x):
        center = self.center(x)
        orientation = self.orientation(x)
        return center, orientation
    

#create roi head class that overrides torchvision.models.detection.roi_heads.RoIHeads
#include most of same torchvision logic (go find) and then add loss