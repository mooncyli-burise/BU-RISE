import torch

@torch.no_grad()
def compute_validation_loss(model, data_loader, device):
    model.train()   # Faster R-CNN only computes losses in training mode

    total_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        total_loss += loss.item()

    model.eval()

    return total_loss / len(data_loader)