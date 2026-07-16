import torchvision
import torch
import torch.nn as nn
from config import TOTAL_CLASSES

class GridNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.mobilenet_v2(
            weights=None #TODO: change back to "DEFAULT" if it makes it worse
        ).features

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, TOTAL_CLASSES),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.fc(x)
        return x