import torch
import cv2
import math

from backbone_model.simple_model_objects_modified import device, data_loader, data_loader_test
from backbone_model.simple_model_modified.model import GridNet

from april_tags.get_data import get_apriltag_video
from april_tags.create_ground_truth import create_ground_truth_vid

from config import WIDTH, HEIGHT

def detect_predict(model_path):
    #load trained model
    eval_model = GridNet().to(device)
    eval_model.load_state_dict(torch.load(model_path, map_location=device))
    eval_model.eval()

    images = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    #init cam
    if not cap.isOpened():
        print("Failed to open camera")
        exit()

    while True:
        ret, frame = cap.read()

        #convert frame to RGB and normalize pixel values to [0, 1] to match pytorch format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(frame_rgb)
        image = image.permute(2,0,1)
        image = image.float() / 255.0
        image = image.to(device)
        image = image.unsqueeze(0)

        gt = create_ground_truth_vid(get_apriltag_video(frame_rgb))

        # print("center:", gt[0]["center"])
        # print("angle:", gt[0]["orientation"])

        #run inference
        with torch.no_grad():
            logits = eval_model(image)

        scale = torch.tensor([WIDTH, HEIGHT], device=device)
        pred_center = logits["center"] * scale

        pred_orientation = logits["orientation"].argmax(dim=1) * 5
       

        if(len(gt)>0):
            gt_center = gt[0]["center"]
            gt_center = torch.tensor(gt_center, device=device)

            gt_orientation = gt[0]["orientation"]
            gt_orientation = torch.tensor(gt_orientation, device=device)

            # Print ground truth
            print("\nGround Truth")
            print("------------")

            print("Centers:", gt_center)

            print("Orientations (angle):", gt_orientation)

            print()

        # Print predictions
        print("Prediction")
        print("----------")

        if "center" in logits:
            print("Centers:", pred_center)

        if "orientation" in logits:
            print("Orientations (angle):", pred_orientation)

        error = torch.abs(pred_orientation - gt_orientation)
        error = torch.minimum(error, 360 - error)

        print()
        print("Center Error:", torch.norm(pred_center-gt_center))
        print("Orientation Error:", error)

        print(logits["class"])

        #predicted center coords
        cx, cy = pred_center.squeeze(0).cpu().tolist()

        # draw line in direction of angle
        length = 20  # length of the arrow in pixels

        angle = pred_orientation.item()  # degrees
        theta = math.radians(angle)

        end_x = int(cx + length * math.sin(theta))
        end_y = int(cy - length * math.cos(theta))  # subtract because image y-axis points down

        cv2.line(
            frame,
            (int(cx), int(cy)),
            (end_x, end_y),
            (0, 0, 255),   # red
            2
        )

        # show predicted center point (red)
        cv2.circle(frame, (int(cx), int(cy)), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(frame,
                    f"({cx}, {cy}",
                    (int(cx), int(cy-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

        # show actual center point (green)
        cx, cy = gt_center.cpu().tolist()

        # draw line in direction of angle
        angle = gt_orientation.item()  # degrees
        theta = math.radians(angle)

        end_x = int(cx + length * math.sin(theta))
        end_y = int(cy - length * math.cos(theta))  # subtract because image y-axis points down

        cv2.line(
            frame,
            (int(cx), int(cy)),
            (end_x, end_y),
            (0, 255, 0),   # green
            2
        )

        # draw green gt center
        cv2.circle(frame, (int(cx), int(cy)), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.putText(frame,
                    f"({cx}, {cy}",
                    (int(cx), int(cy-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)
        
        cv2.imshow("Robot Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()