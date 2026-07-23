import numpy as np
from april_tags.get_data import get_apriltag_images
from config import TAG_SIZE, WIDTH, HEIGHT
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def get_homography():
    all_tags = get_apriltag_images('april_tags/init') # input initialization april tag image here
    return all_tags[0][0].homography

def get_world_coords(x,y):
    H = get_homography()

    p = np.array([x, y, 1.0])

    world = np.linalg.inv(H) @ p
    world /= world[2]

    return world[:2] * TAG_SIZE/2

def plot_world_frame(ax, axis_length=5.0):
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

def plot_camera_frame(ax, axis_length = 5.0):
    R_ct, t_ct = get_rotation_and_translation_matrix()

    R_wc = R_ct.T
    camera_origin = -R_wc @ t_ct
    # -----------------------
    # Camera frame
    # -----------------------

    cam_x = R_wc[:, 0] * axis_length
    cam_y = R_wc[:, 1] * axis_length
    cam_z = R_wc[:, 2] * axis_length

    ax.quiver(*camera_origin, *cam_x, color="r", linestyle="--")
    ax.quiver(*camera_origin, *cam_y, color="g", linestyle="--")
    ax.quiver(*camera_origin, *cam_z, color="b", linestyle="--")

    ax.text(*(camera_origin + cam_x), "Xc")
    ax.text(*(camera_origin + cam_y), "Yc")
    ax.text(*(camera_origin + cam_z), "Zc")

    # # representation of camera's fov
    # corners = [
    #     [0, 0],
    #     [WIDTH, 0],
    #     [WIDTH, HEIGHT],
    #     [0, HEIGHT]
    # ]

    # world_corners = []

    # for pose in corners:
    #     world_corners.append(get_world_coords(pose[0], pose[1]))

    # for i in range(4):
    #     p1 = world_corners[i].flatten()
    #     p2 = world_corners[(i+1)%4].flatten()

    #     ax.plot(
    #         [p1[0],p2[0]],
    #         [p1[1],p2[1]],
    #         [p1[2],p2[2]]
    #     )

        

def plot_coords(ax, pose):
    pose = pose.flatten()
    ax.scatter(
        pose[0],
        pose[1],
        pose[2],
        # color="red",
        # s=50,          # marker size
        # label="Point"
    )


def print_transformations(pixel_coords, world_coords):
    R, T = get_rotation_and_translation_matrix()
    
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

    new_pose = new_pose.flatten()

    ax.plot(
        [camera_origin[0], new_pose[0]],
        [camera_origin[1], new_pose[1]],
        [camera_origin[2], new_pose[2]],
        label="ray"
    )

    ax.scatter(*new_pose, color="k", s=50)
    ax.text(*new_pose, "world point", color="k")

    scale = 10
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    ax.set_box_aspect((1, 1, 1))
    plt.show()

def get_rotation_and_translation_matrix():
    all_tags = get_apriltag_images('april_tags/init') # input initialization april tag image here
    return all_tags[0][0].pose_R, all_tags[0][0].pose_t

def show_homography_view(frame, H):
    """
    Display a bird's-eye view using a homography.

    Parameters
    ----------
    frame : np.ndarray
        Original camera image.
    H : np.ndarray
        3x3 homography matrix.
    """

    # Depending on your homography convention you may need H or inv(H)
    H_world = np.linalg.inv(H)

    warped = cv2.warpPerspective(
        frame,
        H_world,
        (800, 800)
    )

    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye", warped)

    cv2.waitKey(1)

# def print_transformation(R, T, pose, new_pose):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection="3d")

#     axis_length = 5.0

#     # -----------------------
#     # World frame
#     # -----------------------
#     origin = np.zeros(3)

#     ax.quiver(*origin, axis_length, 0, 0, color="r", linewidth=2)
#     ax.quiver(*origin, 0, axis_length, 0, color="g", linewidth=2)
#     ax.quiver(*origin, 0, 0, axis_length, color="b", linewidth=2)

#     ax.text(axis_length, 0, 0, "Xw")
#     ax.text(0, axis_length, 0, "Yw")
#     ax.text(0, 0, axis_length, "Zw")

#     # -----------------------
#     # Camera frame
#     # -----------------------
#     camera_origin = T.flatten()

#     cam_x = R[:, 0] * axis_length
#     cam_y = R[:, 1] * axis_length
#     cam_z = R[:, 2] * axis_length

#     ax.quiver(*camera_origin, *cam_x, color="r", linestyle="--")
#     ax.quiver(*camera_origin, *cam_y, color="g", linestyle="--")
#     ax.quiver(*camera_origin, *cam_z, color="b", linestyle="--")

#     ax.text(*(camera_origin + cam_x), "Xc")
#     ax.text(*(camera_origin + cam_y), "Yc")
#     ax.text(*(camera_origin + cam_z), "Zc")

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

#     new_pose = new_pose.flatten()

#     ax.plot(
#         [camera_origin[0], new_pose[0]],
#         [camera_origin[1], new_pose[1]],
#         [camera_origin[2], new_pose[2]],
#         label="ray"
#     )

#     ax.scatter(*new_pose, color="k", s=50)
#     ax.text(*new_pose, "world point", color="k")

#     scale = 10
#     ax.set_xlim(-scale, scale)
#     ax.set_ylim(-scale, scale)
#     ax.set_zlim(-scale, scale)
#     ax.set_box_aspect((1, 1, 1))
#     plt.show()

# def get_world_coords(pose):
#     # pose = np.array([[x],
#     #                  [y],
#     #                  [z]])
#     R, T = get_rotation_and_translation_matrix()
#     R_inverse = np.linalg.inv(R)
#     # print("R:", R)
#     # print("R inverse:", R_inverse)
#     T_z_only = np.array([[0],
#                          [0],
#                          [T[2][0]]])
#     # print("cam height:",T[2][0])
#     rotated_ray = np.dot(R_inverse, pose)
#     multiplied_ray = rotated_ray * (-1*T[2][0]/rotated_ray[2][0])
#     new_pose = T_z_only + multiplied_ray
#     # print("Camera Z axis:")
#     # print(R_inverse[:,2])
#     print_transformation(R_inverse, T_z_only, pose, new_pose)
#     return new_pose
#     # return new_pose[0][0], new_pose[0][1], new_pose[0][2] # return x, y, and z separately

# #TODO: make function that plots the 2d world position of robot/apriltag

# def get_ray_world(camera_ray):
#     R, T = get_rotation_and_translation_matrix()

#     R_wc = np.linalg.inv(R)

#     ray_world = R_wc @ camera_ray

#     camera_origin = np.array([
#         0,
#         0,
#         T[2][0]
#     ])

#     return camera_origin, ray_world

# def intersect_ground(camera_origin, ray):

#     lam = -camera_origin[2] / ray[2]

#     return camera_origin + lam * ray

# def convert_to_cam(x, y, z=1):
#     pose = np.array([[x],
#                      [y],
#                      [z]])
#     return np.dot(K_inverse, pose)