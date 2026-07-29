import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math
import cv2

from localization.detector import Detector
from localization.camera import Camera

class LocalizationNode(Node):
    def __init__(self, model_path, init_image_path):
        super().__init__("localization")
        self.detector = Detector(model_path, init_image_path)
        self.camera = Camera()

        # create publisher for model
        self.model_pose_publisher = self.create_publisher(
            PoseStamped,
            "/pred_pose",
            10
        )

         # create publisher
        self.apriltag_pose_publisher = self.create_publisher(
            PoseStamped,
            "/ground_truth_pose",
            10
        )

        # timer 
        self.model_timer = self.create_timer(
            0.05,
            self.localization_callback
        )

    def localization_callback(self):
        frame = self.camera.get_frame()

        if frame is None:
            return
        
        pred_center, pred_orientation, pred_class = self.detector.predict_pose(frame)

        gt_center, gt_orientation, gt_class = self.detector.ground_truth_pose(frame)

        print("Pred class:", pred_class)
        print("GT class:", gt_class)

        # resize  to be same res as what is sent into the model
        frame = Camera.model_resize(frame)

        if pred_class == 1:
            self.publish_pose(
                pred_center,
                pred_orientation,
                self.model_pose_publisher
            )

            # plot predictions (red)
            frame = self.detector.plot(frame, pred_center, pred_orientation, (0,0,255))
            
        
        if gt_class == 1:
            self.publish_pose(
                gt_center,
                gt_orientation,
                self.apriltag_pose_publisher
            )

            # plot gt (green)
            frame = self.detector.plot(frame, gt_center, gt_orientation, (0,255,0))
            
        Detector.print_all(pred_center, pred_orientation, pred_class, gt_center, gt_orientation, gt_class)            

        Camera.send_stream(frame)

        
    def publish_pose(self, center, orientation, publisher):
        # convert orientation to format for PoseStamped (only change yaw)
        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # add center to message
        msg.pose.position.x = center[0]
        msg.pose.position.y = center[1]
        msg.pose.position.z = 0.0

        # add orientation to message (convert to radians first for trig)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin((orientation / 180 * math.pi) / 2.0)
        msg.pose.orientation.w = math.cos((orientation / 180 * math.pi) / 2.0)

        publisher.publish(msg)


def main():
    import rclpy

    saved_model = "/workspace/ros2_ws/src/localization/localization/saved_models/best_finetuning_model_lr1e-3.pth"
    init_image_path = "/workspace/ros2_ws/src/localization/localization/init_image/initialization_apriltag.jpg"

    rclpy.init()
    node = LocalizationNode(saved_model, init_image_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

# colcon build
# source install/setup.bash
# ros2 run localization localization_node