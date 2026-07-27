import rclpy
from rclpy import Node
from geometry_msgs.msg import Twist, PoseStamped

from pure_pursuit.controller import PurePursuit

class PurePursuitNode(Node):
    def __init__(self, point, lookAheadDis):
        super().__init__("pure_pursuit")
        self.controller = PurePursuit(lookAheadDis)

        # for publishing outputs to limo
        self.cmd_publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            "/robot_pose",
            self.pose_callback,
            10
        )

        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            "/robot_pose",
            self.pose_callback,
            10
        )

        # single point for now
        self.path = point

        # pure pursuit inputs
        self.current_pose = None
        self.current_heading = None

        # timer 
        self.timer = self.create_timer(
            0.05,
            self.control_loop
        )

    def control_loop(self):
        # TODO: update current pose
        if self.current_pose is None:
            return

        speed, steering = self.controller.compute_control(
            self.path, 
            self.current_pose,
            self.current_heading
        )

        cmd = Twist()

        cmd.linear.x = speed
        cmd.angular.z = steering

        self.cmd_publisher.publish(cmd)