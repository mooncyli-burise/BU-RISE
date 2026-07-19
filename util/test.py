import torch

from simple_testing.og.simple_model_objects import data_loader, device
from simple_model.model import GridNet
from simple_model.pose_loss import PoseLossFunction

def test():
    torch.manual_seed(0)

    model = GridNet().to(device)
    criterion = PoseLossFunction().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Take exactly one fixed batch and reuse it every step.
    images, targets = next(iter(data_loader))
    images = images.to(device)
    targets = targets.to(device)

    model.train()

    for step in range(1000):
        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == targets).float().mean().item()
            print(f"step {step}: loss={loss.item():.4f}, accuracy={accuracy:.3f}")

    model.eval()

    with torch.no_grad():
        predictions = model(images).argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()

    print(f"Final one-batch accuracy: {accuracy:.3f}")