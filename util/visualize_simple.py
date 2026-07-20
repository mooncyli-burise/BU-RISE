import matplotlib.pyplot as plt
import torch
import cv2
#TODO: imports not updated
from simple_model_modified.model import GridNet
from simple_testing.simple_model_objects_modified import device, dataset

def visualize(model_path):
    idx = 0  # choose any sample
    model = GridNet().to(device)

    model.load_state_dict(torch.load(model_path,
                                    map_location=device))
    model.to(device)
    model.eval()

    # Get sample
    image, target = dataset[idx]

    with torch.no_grad():
        logits = model([image.to(device)])[0]

        pred_center = logits["center"]
        gt_center = target["center"]

        pred_orientation = logits["orientation"].argmax(dim=1)
        gt_orientation = target["orientation"]


    # Convert image for plotting
    img = image.permute(1, 2, 0).cpu().numpy()
    #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

    if "orientations" in logits:
        print("Orientations (bins):", pred_orientation)
        print("Orientations (angle):", pred_orientation*5)

    cx, cy = pred_center.to_listt()

    cv2.putText(img,
                f"{pred_orientation*5} deg",
                (cx, cy-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    cv2.circle(img, (cx, cy), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.putText(img,
                f"({cx}, {cy}",
                (cx, cy-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)


    # Display image
    #cv2.imshow("Robot Detection", img_bgr)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

        # key = input("c = next image, q = quit: ")

        # if key == "c":
        #     idx += 1
        #     if idx >= len(dataset):
        #         idx = 0
        # elif key == "q":
        #     break