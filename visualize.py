import matplotlib.pyplot as plt
import torch
from train import train_model, device, dataset

idx = 0  # choose any sample

# Get sample
image, target = dataset[idx]

# Run inference
train_model.eval()
with torch.no_grad():
    output = train_model([image.to(device)])[0]

# Convert image for plotting
img = image.permute(1, 2, 0).cpu().numpy()

# Print ground truth
print("Ground Truth")
print("------------")
print("Boxes:", target["boxes"])
print("Labels:", target["labels"])

if "center" in target:
    print("Center:", target["center"])

if "orientation" in target:
    print("Orientation:", target["orientation"])

print()

# Print predictions
print("Prediction")
print("----------")
print("Scores:", output["scores"])
print("Boxes:", output["boxes"])

if "center" in output:
    print("Center:", output["center"])

if "orientation" in output:
    print("Orientation:", output["orientation"])

# Display image
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis("off")
plt.show()