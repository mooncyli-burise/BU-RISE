from april_tags.get_data import get_apriltag_images
from april_tags.world_frame_transformations import get_world_coords, plot_camera_frame, plot_world_frame, plot_coords
import numpy as np
import matplotlib.pyplot as plt

def test_world_transformations():
    all_tags = get_apriltag_images('april_tags/test')
        
    for image_tags in all_tags:
        for tag in image_tags:
            x, y = tag.center.astype(int)

            world = get_world_coords(x, y)

            plt.scatter(world[0], world[1])

    plt.grid()

    plt.xlim(-2,2)
    plt.ylim(-2,2)
    
    plt.show()

def test_multiple_tags():
    tags = get_apriltag_images('april_tags/test')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_world_frame(ax)
    plot_camera_frame(ax)

    for tag in tags:
        x, y = tag[0].center.astype(int)
        pose = get_world_coords(x,y)
        plot_coords(ax, pose)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    print("untransformed coords: ", f"({x}, {y})")
    print("pose: ", pose)

    plt.show()