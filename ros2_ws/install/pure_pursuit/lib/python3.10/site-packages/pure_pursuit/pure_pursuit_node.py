import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import math
import numpy as np

from pure_pursuit.controller import PurePursuit

def ask_limo_number():
    limo_topic = "/limo"+input("Enter the last 3 numbers on the front of your LIMO: ").strip()+"/cmd_vel"
    return limo_topic

class PurePursuitNode(Node):
    def __init__(self, point, lookAheadDis, limo_topic):
        super().__init__("pure_pursuit")
        self.controller = PurePursuit(lookAheadDis)

        # for publishing outputs to limo
        self.cmd_publisher = self.create_publisher(
            Twist,
            limo_topic,
            10
        )

        # choose either to use apriltag or model poses
        # ros2 run pure_pursuit pure_pursuit_node --ros-args -p pose_source:=model
        # ros2 run pure_pursuit pure_pursuit_node --ros-args -p pose_source:=apriltag
        self.declare_parameter("pose_source", "model")
        pose_source = self.get_parameter(
            "pose_source"
        ).get_parameter_value().string_value

        if pose_source == "apriltag":
            topic = "/ground_truth_pose"
        elif pose_source == "model":
            topic = "/pred_pose"
        else:
            topic = "/robot_pose"

        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            topic,
            self.pose_callback,
            10
        )

        self.get_logger().info(f"Using pose source: {topic}")

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
        if self.current_pose is None or self.current_heading is None:
            print("No pose or heading")
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

        print("\nPure Pursuit")
        print("----------------")
        print("pose:", self.current_pose)
        print("heading:", self.current_heading)
        print("linear vel:", speed)
        print("angular vel:", steering)
        print()

    def pose_callback(self, msg):
        self.current_pose = (
            msg.pose.position.x,
            msg.pose.position.y
        )

        q = msg.pose.orientation

        self.current_heading = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        ) * 180 / math.pi


def main():
    import rclpy

    topic = ask_limo_number()

    try:
        point = [[1, 0.5]] # in meters
        lookAheadDis = 1

        rclpy.init()
        node = PurePursuitNode(point, lookAheadDis, topic)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")  
    finally:
        # Always force a zero-velocity state on exit
        if node is not None:
            stop_twist = Twist()
            stop_twist.linear.x = 0.0
            stop_twist.angular.z = 0.0
            print("\nEmergency stop sent. Exiting cleanly.")
            node.cmd_publisher.publish(stop_twist)
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        

if __name__ == "__main__":
    main()

# colcon build
# source install/setup.bash
# ros2 run pure_pursuit pure_pursuit_node