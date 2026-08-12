#!/usr/bin/env python3

"""
Script that sets the initial pose for AMCL.
"""

import time
import rclpy
from rclpy.node import Node
import transforms3d
from geometry_msgs.msg import PoseWithCovarianceStamped


class InitPosePublisher(Node):
    def __init__(self):
        super().__init__("init_pose_publisher")

        self.declare_parameter("x", value=0.0)
        self.declare_parameter("y", value=0.0)
        self.declare_parameter("theta", value=0.0)
        self.declare_parameter("cov", value=0.5**2)
        self.declare_parameter("repeat_count", value=5)
        self.declare_parameter("repeat_interval_sec", value=0.5)

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        deadline = time.monotonic() + 30.0
        while self.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the AMCL /initialpose subscriber")
            self.get_logger().info("Waiting for AMCL Initial pose subscriber")
            rclpy.spin_once(self, timeout_sec=0.2)

    def send_init_pose(self):
        x = self.get_parameter("x").value
        y = self.get_parameter("y").value
        theta = self.get_parameter("theta").value
        cov = self.get_parameter("cov").value
        repeat_count = self.get_parameter("repeat_count").value
        repeat_interval_sec = self.get_parameter("repeat_interval_sec").value
        self.get_logger().info(
            f"Setting initial AMCL pose to [x: {x}, y: {y}, theta: {theta}] ..."
        )
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        quat = transforms3d.euler.euler2quat(0, 0, theta)
        msg.pose.pose.orientation.w = quat[0]
        msg.pose.pose.orientation.x = quat[1]
        msg.pose.pose.orientation.y = quat[2]
        msg.pose.pose.orientation.z = quat[3]
        msg.pose.covariance = [
            cov,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # Pos X
            0.0,
            cov,
            0.0,
            0.0,
            0.0,
            0.0,  # Pos Y
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # Pos Z
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # Rot X
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # Rot Y
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            cov,  # Rot Z
        ]
        for attempt in range(1, repeat_count + 1):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(msg)
            self.get_logger().info(
                f"Published initial AMCL pose ({attempt}/{repeat_count})"
            )
            rclpy.spin_once(self, timeout_sec=0.1)
            if attempt < repeat_count:
                time.sleep(repeat_interval_sec)


if __name__ == "__main__":
    # Start ROS node and action client
    rclpy.init()
    pub = InitPosePublisher()

    # Send initial pose to AMCL node
    pub.send_init_pose()
    pub.destroy_node()
    rclpy.shutdown()
