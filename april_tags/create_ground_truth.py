from math import atan2, pi

def create_ground_truth(all_tags):
    ground_truth = []

    for tags in all_tags:
        center = tags[0].pose_t[:2]
        rotation_matrix = tags[0].pose_R
        orientation = atan2(rotation_matrix[2][1], rotation_matrix[1][1]) * 180 / pi
        ground_truth.append({
            "center": center,
            "orientation": orientation
        })
    return ground_truth