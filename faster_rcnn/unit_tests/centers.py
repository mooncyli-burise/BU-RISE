import torch
import torch.nn.functional as F
from faster_rcnn.model.robot_pred import RobotPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def test_center_head_can_learn_two_centers():
    torch.manual_seed(0)

    in_features = 1024
    head = RobotPredictor(in_features)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

    features = torch.randn(2, in_features)
    targets = torch.tensor([
        [0.20, -0.10],
        [-0.30, 0.25],
    ])

    for _ in range(500):
        predicted_centers, _ = head(features)
        loss = F.smooth_l1_loss(predicted_centers, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predicted_centers, _ = head(features)

    print("predicted centers:", predicted_centers.tolist())
    print("target centers:", targets.tolist())

    assert torch.allclose(predicted_centers, targets, atol=0.02)