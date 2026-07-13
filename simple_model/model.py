import torchvision
import torch
import torch.nn as nn

class GridNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.mobilenet_v2(
            weights="DEFAULT"
        ).features

        self.requires_grad_(True)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.fc(x)
        return x