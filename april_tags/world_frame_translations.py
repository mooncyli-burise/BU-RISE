import numpy as np
from april_tags.get_data import get_apriltag_images
from config import K_inverse

def get_rotation_and_translation_matrix():
    all_tags = get_apriltag_images("") #TODO: input initialization april tag image here
    return all_tags[0].pose_R, all_tags[0].pose_t

def get_world_coords(x, y, z=0):
    pose = np.array([[x],
                     [y],
                     [z]])
    R, T = get_rotation_and_translation_matrix()
    R_inverse = np.linalg.inv(R)
    T = np.array([[0],
                  [0],
                  [T[2]]])
    new_pose = np.dot(R_inverse, pose-T)
    return new_pose
    # return new_pose[0][0], new_pose[0][1], new_pose[0][2] #return x, y, and z separately


def convert_to_cam(x, y, z=0):
    pose = np.array([[x],
                     [y],
                     [z]])
    return np.dot(K_inverse, pose)