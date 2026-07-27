from localization.model import Model
from localization.apriltag import AprilTag
from localization.world_frame import WorldFrame
from localization.camera import Camera

"""
model's predicted poses + homography to return real world coords
"""

class Detector:
    def __init__(self, model_path):
        self.model = Model(model_path)
        self.apriltag = AprilTag()
        self.camera = Camera()

        # use first camera frame as init image - START PROGRAM WITH INIT APRILTAG IN FRAME ALWAYS
        self.worldframe = WorldFrame(self.camera.get_frame())

    def predict_robot_pose(self, frame):
        pred_center, pred_orientation = self.model.predict(frame)

        pred_center_world = self.worldframe.pixel_to_world(pred_center)

        return pred_center_world, pred_orientation
    
    def actual_robot_pose(self, frame):
        gt_center, gt_orientation = self.apriltag.get_ground_truth(frame)

        gt_center_world = self.worldframe.pixel_to_world(gt_center)

        return gt_center_world, gt_orientation