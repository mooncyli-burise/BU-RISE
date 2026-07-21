import matplotlib.pyplot as plt
import torch
import cv2
#TODO: imports not updated
from backbone_model.simple_model_modified.model import create_model
from faster_rcnn.synthetic_limo_testing.limo_objects import device, dataset

def visualize(model_path):
    idx = 0  # choose any sample
    model = create_model()
    model.load_state_dict(torch.load(model_path,
                                    map_location=device))
    model.to(device)
    model.eval()

    # Get sample
    image, target = dataset[idx]

    with torch.no_grad():
        output = model([image.to(device)])[0]

    best = output["scores"].argmax()

    # Convert image for plotting
    img = image.permute(1, 2, 0).cpu().numpy()
    #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Print ground truth
    print("Ground Truth")
    print("------------")
    print("Boxes:", target["boxes"])
    print("Labels:", target["labels"])

    if "centers" in target:
        print("Centers:", target["centers"])

    if "orientations" in target:
        print("Orientations (bins):", target["orientations"])
        print("Orientations (angle):", target["orientations"][best].item()*5)

    print()

    # Print predictions
    print("Prediction")
    print("----------")
    print("Scores:", output["scores"])
    print("Boxes:", output["boxes"][best])

    if "centers" in output:
        print("Centers:", output["centers"][best])

    if "orientations" in output:
        print("Orientations (bins):", output["orientations"][best])
        print("Orientations (angle):", output["orientations"][best].item()*5)


    x1, y1, x2, y2 = output["boxes"][best].int().tolist()
    cx, cy = output["centers"][best].int().tolist()

    cv2.rectangle(img,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2)
    angle = output["orientations"][best].item()
    cv2.putText(img,
                f"{angle*5} deg",
                (x1, y1-10),
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



    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
