import math

def create_ground_truth(all_tags):
    ground_truth = []

    for tags in all_tags:
        if(len(tags)>0):
            cx, cy = tags[0].center.astype(int)
            rotation_matrix = tags[0].pose_R
            orientation = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / math.pi
            ground_truth.append({
                "center": (cx, cy),
                "orientation": orientation
            })
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