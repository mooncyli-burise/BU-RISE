import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from backbone_model.simple_model_modified.model import GridNet
from backbone_model.real_world_objects import device, dataset_test, dataset_real_world
from config import WIDTH, HEIGHT

def visualize(model_path):
    idx = 0  # choose any sample
    model = GridNet().to(device)

    model.load_state_dict(torch.load(model_path,
                                    map_location=device))
    model.to(device)
    model.eval()

    # Get sample
    image, target = dataset_real_world[idx]

    with torch.no_grad():
        images = image.unsqueeze(0).to(device)  # (1, 3, H, W)
        logits = model(images)

        scale = torch.tensor([WIDTH, HEIGHT], device=device)

        pred_center = logits["center"][0]
        pred_center *= scale
        pred_orientation = logits["orientation"][0].argmax()

        if target:
            gt_center = target["center"]
            gt_center *= scale
            gt_orientation = target["orientation"]


    # Convert image for plotting
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)

    img = image.permute(1, 2, 0).cpu()
    img = img * std + mean          # undo normalization
    img = img.clamp(0, 1)
    img = (img.numpy() * 255).astype(np.uint8)

    # Print ground truth
    print("Ground Truth")
    print("------------")

    if "center" in target:
        print("Centers:", gt_center)

    if "orientation" in target:
        print("Orientations (bins):", gt_orientation)
        print("Orientations (angle):", gt_orientation*5)

    print()

    # Print predictions
    print("Prediction")
    print("----------")

    if "center" in logits:
        print("Centers:", pred_center)

    if "orientation" in logits:
        print("Orientations (bins):", pred_orientation)
        print("Orientations (angle):", pred_orientation*5)

    
    if target:
        orientation_error = torch.abs(pred_orientation - gt_orientation)
        orientation_error = torch.minimum(orientation_error, 360 - orientation_error)

        print()
        print("Center Error:", torch.norm(pred_center-gt_center))
        print("Orientation Error:", orientation_error*5)

    cx, cy = pred_center.cpu().tolist()

    # cv2.putText(img,
    #             f"{pred_orientation*5} deg",
    #             (int(cx), int(cy-10)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.6,
    #             (0,255,0),
    #             2)

    # show predicted center point (red)
    cv2.circle(img, (int(cx), int(cy)), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.putText(img,
                f"({cx}, {cy}",
                (int(cx), int(cy-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2)

    if target:
        # show actual center point (green)
        cx, cy = gt_center.cpu().tolist()

        cv2.circle(img, (int(cx), int(cy)), radius=2, color=(0, 255, 0), thickness=-1)
        cv2.putText(img,
                    f"({cx}, {cy}",
                    (int(cx), int(cy-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)


    # Display image
    #cv2.imshow("Robot Detection", img_bgr)

    plt.figure()
    plt.imshow(img, aspect="equal")
    plt.axis("off")
    plt.show()
