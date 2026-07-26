import cv2
import os
import math
from util.files import crop_to_ratio
from config import APRILTAG_HEIGHT, APRILTAG_WIDTH

def print_apriltags(image_folder, ground_truth):
    image_files = sorted(
        (
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
    )

    if len(image_files) != len(ground_truth):
        print(
            f"Warning: {len(image_files)} images but "
            f"{len(ground_truth)} ground truth entries."
        )

    for filename, gt in zip(image_files, ground_truth):
        image = cv2.imread(os.path.join(image_folder, filename))
        cropped = crop_to_ratio(image)
        downscaled = cv2.resize(cropped, (APRILTAG_WIDTH, APRILTAG_HEIGHT), interpolation=cv2.INTER_AREA)

        if image is None:
            continue

        h, w = downscaled.shape[:2]

        # Skip images without a detected robot
        if gt.get("class", 1) == 0:
            cv2.putText(
                downscaled,
                "No Robot",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        else:
            # Convert normalized coordinates back to pixels
            cx = int(gt["center"][0] * w)
            cy = int(gt["center"][1] * h)

            angle = gt["orientation"]
            theta = math.radians(angle)

            length = 25
            end_x = int(cx + length * math.sin(theta))
            end_y = int(cy - length * math.cos(theta))

            # Draw center
            cv2.circle(downscaled, (cx, cy), 4, (0, 255, 0), -1)

            # Draw orientation arrow
            cv2.line(downscaled, (cx, cy), (end_x, end_y), (0, 255, 0), 2)

            # Draw text
            cv2.putText(
                downscaled,
                f"({cx}, {cy})",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                downscaled,
                f"{angle:.1f} deg",
                (cx + 5, cy + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Ground Truth", downscaled)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()