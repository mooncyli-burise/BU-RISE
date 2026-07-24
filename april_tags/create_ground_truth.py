import math
from config import APRILTAG_HEIGHT, APRILTAG_WIDTH
from util.normalize_pixel_coords import normalize_coords

def create_ground_truth(all_tags):
    ground_truth = []

    for tags in all_tags:
        if(len(tags)>0):
            cx, cy = tags[0].center.astype(int)
            cx, cy = normalize_coords(cx, cy, False, APRILTAG_WIDTH, APRILTAG_HEIGHT)
            print("cx:", cx, "cy:", cy)
            rotation_matrix = tags[0].pose_R
            orientation = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / math.pi
            print("orientation:", orientation)
            print()
            ground_truth.append({
                "center": (cx, cy),
                "orientation": orientation
            })
        else:
            print("No tag")
            print()
    return ground_truth

def create_ground_truth_vid(tags):
    ground_truth = []

    if(len(tags)>0):
        cx, cy = tags[0].center.astype(int)
        rotation_matrix = tags[0].pose_R
        orientation = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / math.pi
        ground_truth.append({
            "center": (cx, cy),
            "orientation": orientation
        })
    return ground_truth