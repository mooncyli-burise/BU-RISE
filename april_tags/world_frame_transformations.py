import numpy as np
from april_tags.get_data import get_apriltag_images
from config import K_inverse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_rotation_and_translation_matrix():
    all_tags = get_apriltag_images('april_tags/init') # input initialization april tag image here
    return all_tags[0][0].pose_R, all_tags[0][0].pose_t

def print_transformation(R, T, pose, new_pose):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    axis_length = 5.0

    # -----------------------
    # World frame
    # -----------------------
    origin = np.zeros(3)

    ax.quiver(*origin, axis_length, 0, 0, color="r", linewidth=2)
    ax.quiver(*origin, 0, axis_length, 0, color="g", linewidth=2)
    ax.quiver(*origin, 0, 0, axis_length, color="b", linewidth=2)

    ax.text(axis_length, 0, 0, "Xw")
    ax.text(0, axis_length, 0, "Yw")
    ax.text(0, 0, axis_length, "Zw")

    # -----------------------
    # Camera frame
    # -----------------------
    camera_origin = T.flatten()

    cam_x = R[:, 0] * axis_length
    cam_y = R[:, 1] * axis_length
    cam_z = R[:, 2] * axis_length

    ax.quiver(*camera_origin, *cam_x, color="r", linestyle="--")
    ax.quiver(*camera_origin, *cam_y, color="g", linestyle="--")
    ax.quiver(*camera_origin, *cam_z, color="b", linestyle="--")

    ax.text(*(camera_origin + cam_x), "Xc")
    ax.text(*(camera_origin + cam_y), "Yc")
    ax.text(*(camera_origin + cam_z), "Zc")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    pose = pose.flatten()
    new_pose = new_pose.flatten()
    ax.scatter(*pose, color="m", s=50)
    ax.text(*pose, "camera point", color="m")

    ax.scatter(*new_pose, color="k", s=50)
    ax.text(*new_pose, "world point", color="k")

    scale = 10
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    ax.set_box_aspect((1, 1, 1))
    plt.show()

def get_world_coords(x, y, z=0):
    pose = np.array([[x],
                     [y],
                     [z]])
    R, T = get_rotation_and_translation_matrix()
    R_inverse = np.linalg.inv(R)
    print("R inverse:", R_inverse)
    T_z_only = np.array([[0],
                         [0],
                         [T[2][0]]])
    new_pose = np.dot(R_inverse, pose-T_z_only)
    print_transformation(R_inverse, T_z_only, pose, new_pose)
    return new_pose
    # return new_pose[0][0], new_pose[0][1], new_pose[0][2] # return x, y, and z separately


def convert_to_cam(x, y, z=0):
    pose = np.array([[x],
                     [y],
                     [z]])
    return np.dot(K_inverse, pose)