import torch

from simple_model.pose_loss import PoseLossFunction


def test_pose_loss_is_zero_for_perfect_prediction():
    loss_fn = PoseLossFunction()

    logits = torch.full((1, 64), -10.0)
    logits[0, 0] = 10.0
    targets = torch.tensor([0])

    loss = loss_fn(logits, targets)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_pose_loss_matches_average_cost_for_uniform_logits():
    loss_fn = PoseLossFunction()

    logits = torch.zeros(1, 64)
    targets = torch.tensor([7])

    loss = loss_fn(logits, targets)
    expected = loss_fn.A[:, 7].mean()

    assert torch.isclose(loss, expected, atol=1e-6)


def test_pose_loss_nearby_pose_same_orientation():
    """Target class 7, prediction concentrated on class 11 (same orientation, adjacent pose)."""
    loss_fn = PoseLossFunction()

    # Target: class 7 (pose=1, orientation=3)
    # Prediction: class 11 (pose=2, orientation=3) with high confidence
    logits = torch.full((1, 64), -10.0)
    logits[0, 11] = 10.0
    targets = torch.tensor([7])

    loss = loss_fn(logits, targets)
    
    # Expected: cost A[11, 7] (since softmax puts ~1.0 on class 11)
    expected_cost = loss_fn.A[11, 7].item()
    
    print(f"Nearby pose (same orientation): loss={loss:.6f}, expected_cost={expected_cost:.6f}")
    assert torch.isclose(loss, torch.tensor(expected_cost), atol=1e-5)


def test_pose_loss_nearby_orientation_same_pose():
    """Target class 7, prediction concentrated on class 6 (same pose, adjacent orientation)."""
    loss_fn = PoseLossFunction()

    # Target: class 7 (pose=1, orientation=3)
    # Prediction: class 6 (pose=1, orientation=2) with high confidence
    logits = torch.full((1, 64), -10.0)
    logits[0, 6] = 10.0
    targets = torch.tensor([7])

    loss = loss_fn(logits, targets)
    
    # Expected: cost A[6, 7] (since softmax puts ~1.0 on class 6)
    expected_cost = loss_fn.A[6, 7].item()
    
    print(f"Nearby orientation (same pose): loss={loss:.6f}, expected_cost={expected_cost:.6f}")
    assert torch.isclose(loss, torch.tensor(expected_cost), atol=1e-5)


def test_pose_loss_nearby_both_pose_and_orientation():
    """Target class 20, prediction concentrated on class 17 (nearby in both)."""
    loss_fn = PoseLossFunction()

    # Target: class 20 (pose=5, orientation=0)
    # Prediction: class 17 (pose=4, orientation=1) with high confidence
    logits = torch.full((1, 64), -10.0)
    logits[0, 17] = 10.0
    targets = torch.tensor([20])

    loss = loss_fn(logits, targets)
    
    # Expected: cost A[17, 20] (since softmax puts ~1.0 on class 17)
    expected_cost = loss_fn.A[17, 20].item()
    
    print(f"Nearby both pose and orientation: loss={loss:.6f}, expected_cost={expected_cost:.6f}")
    assert torch.isclose(loss, torch.tensor(expected_cost), atol=1e-5)
