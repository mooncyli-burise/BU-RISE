import torch
import torch.nn.functional as F
from faster_rcnn.model.robot_pred import RobotPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def test_roi_classifier_can_learn_objectness():
    torch.manual_seed(0)

    in_features = 1024
    classifier = FastRCNNPredictor(in_features, num_classes=2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

    features = torch.randn(2, in_features)
    targets = torch.tensor([1, 0])  # 1 = robot, 0 = background

    for _ in range(500):
        class_logits, _ = classifier(features)
        loss = F.cross_entropy(class_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    class_logits, _ = classifier(features)
    predictions = class_logits.argmax(dim=1)

    print(f"predictions: {predictions.tolist()}, targets: {targets.tolist()}")

    assert torch.equal(predictions, targets)

def test_box_regressor_can_learn_two_boxes():
    torch.manual_seed(0)

    in_features = 1024
    predictor = FastRCNNPredictor(in_features, num_classes=2)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.01)

    features = torch.randn(2, in_features)

    # These are encoded box-regression targets: dx, dy, dw, dh.
    targets = torch.tensor([
        [0.10, -0.20, 0.05, 0.15],
        [-0.10, 0.30, -0.05, 0.10],
    ])

    for _ in range(500):
        _, box_regression = predictor(features)

        # Predictor outputs 4 values per class:
        # columns 0:4 = background, columns 4:8 = robot (class 1).
        robot_box_regression = box_regression[:, 4:8]
        loss = F.smooth_l1_loss(robot_box_regression, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, box_regression = predictor(features)
    predictions = box_regression[:, 4:8]

    print("predicted box deltas:", predictions.tolist())
    print("target box deltas:", targets.tolist())

    assert torch.allclose(predictions, targets, atol=0.02)