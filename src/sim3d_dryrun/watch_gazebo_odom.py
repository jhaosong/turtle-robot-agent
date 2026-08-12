#!/usr/bin/env python3

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class OdomWatcher(Node):
    def __init__(
        self,
        topic_name: str,
        print_period_sec: float,
        max_messages: int,
        idle_timeout_sec: float,
    ):
        super().__init__("sim3d_odom_watcher")
        self._topic_name = topic_name
        self._print_period_sec = print_period_sec
        self._max_messages = max_messages
        self._idle_timeout_sec = idle_timeout_sec
        self._message_count = 0
        self._last_print_time = 0.0
        self._last_message_time = time.time()
        self._subscription = self.create_subscription(Odometry, topic_name, self._on_odom, 10)
        self._idle_timer = self.create_timer(0.5, self._check_idle)

        print(
            json.dumps(
                {
                    "watcher": "sim3d_odom_watcher",
                    "topic": self._topic_name,
                    "print_period_sec": self._print_period_sec,
                    "status": "listening",
                },
                ensure_ascii=True,
            )
        )

    def _on_odom(self, msg: Odometry):
        now = time.time()
        self._last_message_time = now
        self._message_count += 1
        if self._print_period_sec > 0 and now - self._last_print_time < self._print_period_sec:
            return
        self._last_print_time = now

        orientation = msg.pose.pose.orientation
        yaw = self._quaternion_to_yaw(orientation.z, orientation.w)
        print(
            json.dumps(
                {
                    "watcher": "sim3d_odom_watcher",
                    "message_index": self._message_count,
                    "frame_id": msg.header.frame_id,
                    "child_frame_id": msg.child_frame_id,
                    "pose": {
                        "x": round(msg.pose.pose.position.x, 3),
                        "y": round(msg.pose.pose.position.y, 3),
                        "yaw": round(yaw, 3),
                    },
                    "velocity": {
                        "linear_x": round(msg.twist.twist.linear.x, 3),
                        "angular_z": round(msg.twist.twist.angular.z, 3),
                    },
                },
                ensure_ascii=True,
            )
        )

        if self._max_messages > 0 and self._message_count >= self._max_messages:
            self.get_logger().info("Reached max odom message count, shutting down.")
            raise SystemExit(0)

    def _check_idle(self):
        if self._idle_timeout_sec <= 0:
            return
        if time.time() - self._last_message_time >= self._idle_timeout_sec:
            print(
                json.dumps(
                    {
                        "watcher": "sim3d_odom_watcher",
                        "topic": self._topic_name,
                        "message_count": self._message_count,
                        "status": "idle_timeout",
                    },
                    ensure_ascii=True,
                )
            )
            raise SystemExit(0)

    @staticmethod
    def _quaternion_to_yaw(z: float, w: float) -> float:
        return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def main():
    parser = argparse.ArgumentParser(description="Print compact /odom pose updates from the Gazebo demo.")
    parser.add_argument("--topic", default="/odom", help="Odometry topic. Defaults to /odom.")
    parser.add_argument(
        "--print-period-sec",
        type=float,
        default=0.5,
        help="Minimum seconds between printed odom summaries. Defaults to 0.5.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="Exit after this many odom messages. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--idle-timeout-sec",
        type=float,
        default=5.0,
        help="Exit after this many seconds without odom messages. 0 disables timeout.",
    )
    args = parser.parse_args()

    rclpy.init(args=None)
    node = OdomWatcher(
        topic_name=args.topic,
        print_period_sec=args.print_period_sec,
        max_messages=args.max_messages,
        idle_timeout_sec=args.idle_timeout_sec,
    )

    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
