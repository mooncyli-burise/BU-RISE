import math
import cv2
import json
import os
from config import APRILTAG_HEIGHT, APRILTAG_WIDTH, TAG_SIZE, TAG_SIZE_LIMO
from util.normalize_pixel_coords import normalize_coords
from util.files import get_sorted_files, crop_to_ratio
from april_tags.get_data import get_apriltag_by_image

def create_ground_truth(sequence_folder, target_folder = "", tag_size = TAG_SIZE):
    image_files = get_sorted_files(sequence_folder)

    ground_truth = []

    for file in image_files:
        image = cv2.imread(file)

        height, width = image.shape[:2]   # 720, 1280

        cropped = crop_to_ratio(image)

        downscaled = cv2.resize(cropped, (APRILTAG_WIDTH, APRILTAG_HEIGHT), interpolation=cv2.INTER_AREA)
        tags = get_apriltag_by_image(downscaled, TAG_SIZE_LIMO)

        if len(tags) > 0:
            print(f"Tag detected in {file}")

            cx, cy = tags[0].center
            cx, cy = normalize_coords(cx, cy, False, APRILTAG_WIDTH, APRILTAG_HEIGHT)

            rotation_matrix = tags[0].pose_R
            orientation = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / math.pi

            ground_truth.append({
                "center": (cx, cy),
                "orientation": orientation,
                "class": 1
            })
        else:
            print(f"No tag detected in {file}")

            ground_truth.append({
                "center": (0, 0),
                "orientation": 0,
                "class": 0
            })

    with open(os.path.join(target_folder, "ground_truth.json"), "w") as file:
            json.dump(ground_truth, file, indent = 4)
        
    return ground_truth

def create_ground_truth_vid(tags):
    ground_truth = []

    if(len(tags)>0):
        cx, cy = tags[0].center.astype(int)
        rotation_matrix = tags[0].pose_R
        orientation = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / math.pi
        ground_truth.append({
            "center": (cx, cy),
            "orientation": orientation,
            "class": 1
        })
    return ground_truth