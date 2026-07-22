import torch
import cv2
import math

from april_tags.get_data import get_apriltag_video
from april_tags.create_ground_truth import create_ground_truth_vid

from pure_pursuit import pure_pursuit

from config import WIDTH, HEIGHT

def update(path, lookahead):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    #init cam
    if not cap.isOpened():
        print("Failed to open camera")
        exit()

    lastFoundIndex = 0

    while True:
        ret, frame = cap.read()

        #convert frame to RGB and normalize pixel values to [0, 1] to match pytorch format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(frame_rgb)
        image = image.permute(2,0,1)
        image = image.float() / 255.0
        image = image.unsqueeze(0)

        gt = create_ground_truth_vid(get_apriltag_video(frame_rgb))

        # print("center:", gt[0]["center"])
        # print("angle:", gt[0]["orientation"])5
       

        if(len(gt)>0):
            gt_center = gt[0]["center"]
            gt_center = torch.tensor(gt_center)

            gt_orientation = gt[0]["orientation"]
            gt_orientation = torch.tensor(gt_orientation)

            # Print ground truth
            print("\nGround Truth")
            print("------------")

            print("Centers:", gt_center)

            print("Orientations (angle):", gt_orientation)

            print()

        length = 20 # length in px of direction line

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

        goalPt, lastFoundIndex, turnVel = pure_pursuit.pure_pursuit_step(path, (cx, cy), angle, lookahead, lastFoundIndex)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()