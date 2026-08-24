#!/usr/bin/env python3

"""Spawn the demo fire extinguisher at an explicit scene coordinate."""

import os

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity
import rclpy
from rclpy.node import Node
import transforms3d


MODEL_NAME = "fire_extinguisher"


class FireExtinguisherSpawner(Node):
    def __init__(self):
        super().__init__("fire_extinguisher_spawner")
        self.client = self.create_client(SpawnEntity, "/spawn_entity")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for Gazebo spawn service...")

        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", 0.30)
        self.declare_parameter("yaw", 0.0)

    def spawn(self):
        object_x = float(self.get_parameter("x").value)
        object_y = float(self.get_parameter("y").value)
        object_z = float(self.get_parameter("z").value)
        object_yaw = float(self.get_parameter("yaw").value)

        model_file = os.path.join(
            get_package_share_directory("tb3_worlds"),
            "models",
            MODEL_NAME,
            "model.sdf",
        )
        with open(model_file, encoding="utf-8") as stream:
            model_xml = stream.read()

        request = SpawnEntity.Request()
        request.name = MODEL_NAME
        request.xml = model_xml
        request.initial_pose.position.x = object_x
        request.initial_pose.position.y = object_y
        request.initial_pose.position.z = object_z
        quaternion = transforms3d.euler.euler2quat(0, 0, object_yaw)
        request.initial_pose.orientation.w = quaternion[0]
        request.initial_pose.orientation.x = quaternion[1]
        request.initial_pose.orientation.y = quaternion[2]
        request.initial_pose.orientation.z = quaternion[3]

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            message = response.status_message if response is not None else "no response"
            raise RuntimeError(f"Failed to spawn {MODEL_NAME}: {message}")
        self.get_logger().info(
            f"Spawned {MODEL_NAME} at x={object_x:.2f}, y={object_y:.2f}, z={object_z:.2f}"
        )


def main():
    rclpy.init()
    node = FireExtinguisherSpawner()
    try:
        node.spawn()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
