import numpy as np

import config
from april_tags.get_data import get_apriltag_by_image

class WorldFrame:
    def __init__(self, init_image):
        # get homography from first tag of the image (only one in image)
        self.H = get_apriltag_by_image(init_image)[0].homography
        self.H_inv = np.linalg.inv(self.H)

    def pixel_to_world(self, pose):
        x = pose[0]
        y = pose[1]

        if 0<=x<=1 and 0<=y<=1:
            x *= config.APRILTAG_WIDTH
            y *= config.APRILTAG_HEIGHT
        
        p = np.array([x, y, 1.0])
    
        world = self.H_inv @ p
        world /= world[2]
    
        return world[:2] * config.TAG_SIZE/2

    def world_to_pixel(self, pose):
        x = pose[0]
        y = pose[1]
        
        p = np.array([x, y, 1.0])
    
        pixel = self.H @ p
        pixel /= pixel[2]
    
        return pixel[:2] / (config.TAG_SIZE/2)        

    def plot_homography_grid(self):
        return

    def plot_world_coords(self):
        return