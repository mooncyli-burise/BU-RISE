import torch
from simple_model_modified.loss_function.py import OrientationLossFunction
from faster_rcnn.model.robot_pred import RobotPredictor

def test_orientation_loss():
    orientation_loss = OrientationLossFunction()

    # test 1: correct bin?
    logits = torch.zeros(1, 72)       # one prediction, 72 orientation bins
    targets = torch.tensor([10])      # target bin 10
    logits[0][10] = 100

    loss = orientation_loss(logits, targets)
    print("loss:", loss)
    print("correct loss should be around 0")

    # test 2: similarity?
    
    targets = torch.tensor([10])      # target bin 10
    logits_near = torch.zeros(1, 72)       # one prediction, 72 orientation bins
    logits_near[0][12] = 100

    logits_far = torch.zeros(1, 72)       # one prediction, 72 orientation bins
    logits_far[0][46] = 100

    near_loss = orientation_loss(logits_near, targets)
    far_loss = orientation_loss(logits_far, targets)
    print("near loss:", near_loss)
    print("far loss:", far_loss)
    print("near loss should be less than far loss")

    # test 3: wrap around
    targets = torch.tensor([0])
    wraparound_logits = torch.zeros(1,72)
    wraparound_logits[0][71] = 100

    wraparound_loss = orientation_loss(wraparound_logits, targets)
    print("wraparound loss:", wraparound_loss)
    print("wraparound loss should be around 1")

def test_orientation_head_output_shape():
    batch_size = 4
    in_features = 1024  # use the same input-feature count as your RoI head

    orientation_head = RobotPredictor(in_features)
    features = torch.randn(batch_size, in_features)

    centers, orientation_logits = orientation_head(features)

    assert centers.shape == (batch_size, 2), "center prediction is correct shape"
    assert orientation_logits.shape == (batch_size, 72), "orientation prediction is correct shape"
    assert torch.isfinite(orientation_logits).all(), "finite number of orientation logits"
    print("center shape:", centers.shape)
    print("orientation shape:", orientation_logits.shape)
    print("is finite? ", torch.isfinite(orientation_logits).all())


def test_orientation_head_can_learn_one_bin():
    torch.manual_seed(0)

    in_features = 1024
    #target_bin = torch.tensor([10])

    orientation_head = RobotPredictor(in_features)
    orientation_loss = OrientationLossFunction()
    optimizer = torch.optim.Adam(orientation_head.parameters(), lr=0.01)

    features = torch.randn(2, in_features)
    targets = torch.tensor([10, 40])

    for _ in range(200):
        _, logits = orientation_head(features)
        loss = orientation_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, orientation_logits = orientation_head(features)
    predicted_bin = orientation_logits.argmax(dim=1)

    for prediction, target in zip(predicted_bin, targets):
        print(f"prediction: {prediction.item()}, target: {target.item()}")
    
    assert torch.equal(logits.argmax(dim=1), targets)