import rclpy
from rclpy import Node
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler

from localization.detector import Detector
from localization.camera import Camera

class LocalizationNode(Node):
    def __init__(self, model_path, init_image):
        super().__init__("localization")
        self.detector = Detector(model_path, init_image)
        self.camera = Camera()

        # create publisher for model
        self.model_pose_publisher = self.create_publisher(
            PoseStamped,
            "/robot_pose",
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
        
        pred_center, pred_orientation = self.detector.predict_pose(frame)

        gt_center, gt_orientation = self.detector.ground_truth_pose(frame)

        self.publish_pose(
            pred_center,
            pred_orientation,
            self.model_pose_publisher
        )

        self.publish_pose(
            gt_center,
            gt_orientation,
            self.apriltag_pose_publisher
        )

        
    def publish_pose(self, center, orientation, publisher):
        # convert orientation to format for PoseStamped (only change yaw)
        q = quaternion_from_euler(0.0, 0.0, orientation)

        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # add center to message
        msg.pose.position.x = center[0]
        msg.pose.position.y = center[1]
        msg.pose.position.z = 0.0

        # add orientation to message
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        publisher.publish(msg)