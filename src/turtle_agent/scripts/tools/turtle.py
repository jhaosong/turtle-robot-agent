#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import threading
import time
import os
from math import cos, sin, sqrt
from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import Twist
from langchain.agents import tool
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Empty
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, TeleportAbsolute, TeleportRelative, Kill, SetPen

cmd_vel_pubs = {}
_pose_cache: Dict[str, Pose] = {}
_pose_events: Dict[str, threading.Event] = {}
_pose_lock = threading.Lock()
_ros_tools_node: Optional["TurtleToolsNode"] = None
_ros_tools_executor: Optional[MultiThreadedExecutor] = None
_ros_tools_thread: Optional[threading.Thread] = None
COMMAND_STEP_SECONDS = 0.15


def _trace_turtle(label: str, payload):
    if os.getenv("ROSA_TRACE", "false").strip().lower() not in ("1", "true", "yes", "on"):
        return
    print(f"\n[ROSA TRACE] {label}")
    print(payload)


class TurtleToolsNode(Node):
    """ROS2 helper node for TurtleSim tool execution."""

    def __init__(self):
        super().__init__("rosa_turtle_tools")
        self._ros_publishers: Dict[str, object] = {}
        self._ros_subscriptions: Dict[str, object] = {}
        self._ros_clients: Dict[tuple, object] = {}
        self.ensure_publisher("turtle1")
        self.ensure_pose_subscription("turtle1")

    def ensure_publisher(self, name: str):
        name = normalize_name(name)
        if name not in self._ros_publishers:
            self._ros_publishers[name] = self.create_publisher(
                Twist, f"/{name}/cmd_vel", 10
            )
        cmd_vel_pubs[name] = self._ros_publishers[name]
        return self._ros_publishers[name]

    def remove_publisher(self, name: str):
        name = normalize_name(name)
        publisher = self._ros_publishers.pop(name, None)
        if publisher is not None:
            self.destroy_publisher(publisher)
        remove_cmd_vel_pub(name)

    def ensure_pose_subscription(self, name: str):
        name = normalize_name(name)
        if name not in self._ros_subscriptions:
            event = _pose_events.setdefault(name, threading.Event())

            def callback(msg: Pose, turtle_name=name, turtle_event=event):
                with _pose_lock:
                    _pose_cache[turtle_name] = msg
                turtle_event.set()

            self._ros_subscriptions[name] = self.create_subscription(
                Pose, f"/{name}/pose", callback, 10
            )
        _pose_events.setdefault(name, threading.Event())
        return self._ros_subscriptions[name]

    def get_pose(self, name: str, timeout: float = 5.0) -> Pose:
        name = normalize_name(name)
        self.ensure_pose_subscription(name)
        event = _pose_events[name]
        if not event.wait(timeout):
            raise TimeoutError(f"Timed out waiting for /{name}/pose.")
        with _pose_lock:
            return _pose_cache[name]

    def get_client(self, srv_type, service_name: str, timeout: float = 5.0):
        key = (service_name, srv_type)
        client = self._ros_clients.get(key)
        if client is None:
            client = self.create_client(srv_type, service_name)
            self._ros_clients[key] = client
        if not client.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Service {service_name} not available.")
        return client

    def call_service(
        self, srv_type, service_name: str, request, timeout: float = 5.0
    ):
        _trace_turtle(
            "ROS2 SERVICE CALL",
            {
                "service": service_name,
                "service_type": srv_type.__name__,
                "request": request,
            },
        )
        client = self.get_client(srv_type, service_name, timeout=timeout)
        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done():
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out calling service {service_name}.")
            time.sleep(0.05)
        result = future.result()
        if result is None:
            exc = future.exception()
            if exc is not None:
                raise RuntimeError(str(exc))
            raise RuntimeError(f"Service {service_name} returned no result.")
        _trace_turtle(
            "ROS2 SERVICE RESPONSE",
            {
                "service": service_name,
                "response": result,
            },
        )
        return result


def normalize_name(name: str) -> str:
    return name.replace("/", "")


def init_ros_turtle_tools():
    """Initialize the shared ROS2 node used by TurtleSim tools."""
    global _ros_tools_node, _ros_tools_executor, _ros_tools_thread
    if _ros_tools_node is not None:
        return _ros_tools_node

    if not rclpy.ok():
        rclpy.init(args=None)

    _ros_tools_node = TurtleToolsNode()
    _ros_tools_executor = MultiThreadedExecutor()
    _ros_tools_executor.add_node(_ros_tools_node)
    _ros_tools_thread = threading.Thread(
        target=_ros_tools_executor.spin, daemon=True
    )
    _ros_tools_thread.start()
    return _ros_tools_node


def shutdown_ros_turtle_tools():
    """Shutdown the shared ROS2 node used by TurtleSim tools."""
    global _ros_tools_node, _ros_tools_executor, _ros_tools_thread
    if _ros_tools_executor is not None:
        _ros_tools_executor.shutdown()
    if _ros_tools_thread is not None:
        _ros_tools_thread.join(timeout=1.0)
    if _ros_tools_node is not None:
        _ros_tools_node.destroy_node()
    _ros_tools_node = None
    _ros_tools_executor = None
    _ros_tools_thread = None
    cmd_vel_pubs.clear()
    _pose_cache.clear()
    _pose_events.clear()


def get_ros_node() -> TurtleToolsNode:
    node = init_ros_turtle_tools()
    if node is None:
        raise RuntimeError("Failed to initialize ROS2 TurtleSim tools.")
    return node


def mark_pose_stale(name: str):
    """Force the next pose read to wait for a fresh message."""
    name = normalize_name(name)
    event = _pose_events.get(name)
    if event is not None:
        event.clear()


def add_cmd_vel_pub(name: str, publisher):
    global cmd_vel_pubs
    cmd_vel_pubs[normalize_name(name)] = publisher


def remove_cmd_vel_pub(name: str):
    global cmd_vel_pubs
    cmd_vel_pubs.pop(normalize_name(name), None)


def within_bounds(x: float, y: float) -> tuple:
    """
    Check if the given x, y coordinates are within the bounds of the turtlesim environment.

    :param x: The x-coordinate.
    :param y: The y-coordinate.
    """
    if 0 <= x <= 11 and 0 <= y <= 11:
        return True, "Coordinates are within bounds."
    else:
        return False, f"({x}, {y}) will be out of bounds. Range is [0, 11] for each."


def will_be_within_bounds(
    name: str, velocity: float, lateral: float, angle: float, duration: float = 1.0
) -> tuple:
    """Check if the turtle will be within bounds after publishing a twist command."""
    # Get the current pose of the turtle
    pose = get_turtle_pose.invoke({"names": [name]})
    if "Error" in pose:
        return False, pose["Error"]
    current_x = pose[name].x
    current_y = pose[name].y
    current_theta = pose[name].theta

    # Calculate the new position and orientation
    if abs(angle) < 1e-6:  # Straight line motion
        new_x = (
            current_x
            + (velocity * cos(current_theta) - lateral * sin(current_theta)) * duration
        )
        new_y = (
            current_y
            + (velocity * sin(current_theta) + lateral * cos(current_theta)) * duration
        )
    else:  # Circular motion
        radius = sqrt(velocity**2 + lateral**2) / abs(angle)
        center_x = current_x - radius * sin(current_theta)
        center_y = current_y + radius * cos(current_theta)
        angle_traveled = angle * duration
        new_x = center_x + radius * sin(current_theta + angle_traveled)
        new_y = center_y - radius * cos(current_theta + angle_traveled)

        # Check if any point on the circle is out of bounds
        for t in range(int(duration) + 1):
            angle_t = current_theta + angle * t
            x_t = center_x + radius * sin(angle_t)
            y_t = center_y - radius * cos(angle_t)
            in_bounds, _ = within_bounds(x_t, y_t)
            if not in_bounds:
                return (
                    False,
                    f"The circular path will go out of bounds at ({x_t:.2f}, {y_t:.2f}).",
                )

    # Check if the final x, y coordinates are within bounds
    in_bounds, message = within_bounds(new_x, new_y)
    if not in_bounds:
        return (
            False,
            f"This command will move the turtle out of bounds to ({new_x:.2f}, {new_y:.2f}).",
        )

    return True, f"The turtle will remain within bounds at ({new_x:.2f}, {new_y:.2f})."


@tool
def spawn_turtle(name: str, x: float, y: float, theta: float) -> str:
    """
    Spawn a turtle at the given x, y, and theta coordinates.

    :param name: name of the turtle.
    :param x: x-coordinate.
    :param y: y-coordinate.
    :param theta: angle.
    """
    in_bounds, message = within_bounds(x, y)
    if not in_bounds:
        return message

    # Remove any forward slashes from the name
    name = normalize_name(name)

    try:
        node = get_ros_node()
        request = Spawn.Request()
        request.x = float(x)
        request.y = float(y)
        request.theta = float(theta)
        request.name = name
        node.call_service(Spawn, "/spawn", request)
        node.ensure_publisher(name)
        node.ensure_pose_subscription(name)

        return f"{name} spawned at x: {x}, y: {y}, theta: {theta}."
    except Exception as e:
        return f"Failed to spawn {name}: {e}"


@tool
def kill_turtle(names: List[str]):
    """
    Removes a turtle from the turtlesim environment.

    :param names: List of names of the turtles to remove (do not include the forward slash).
    """

    # Remove any forward slashes from the names
    names = [normalize_name(name) for name in names]
    response = ""
    node = get_ros_node()

    for name in names:
        try:
            request = Kill.Request()
            request.name = name
            node.call_service(Kill, "/kill", request)
            node.remove_publisher(name)
            response += f"Successfully killed {name}.\n"
        except Exception as e:
            response += f"Failed to kill {name}: {e}\n"

    return response


@tool
def clear_turtlesim():
    """Clears the turtlesim background and sets the color to the value of the background parameters."""
    try:
        node = get_ros_node()
        node.call_service(Empty, "/clear", Empty.Request())
        return "Successfully cleared the turtlesim background."
    except Exception as e:
        return f"Failed to clear the turtlesim background: {e}"


@tool
def get_turtle_pose(names: List[str]) -> dict:
    """
    Get the pose of one or more turtles.

    :param names: List of names of the turtles to get the pose of.
    """

    # Remove any forward slashes from the names
    names = [normalize_name(name) for name in names]
    poses = {}

    # Get the pose of each turtle
    node = get_ros_node()
    for name in names:
        try:
            msg = node.get_pose(name, timeout=5.0)
            poses[name] = msg
        except Exception:
            return {
                "Error": f"Failed to get pose for {name}: /{name}/pose not available."
            }
    return poses


@tool
def teleport_absolute(
    name: str, x: float, y: float, theta: float, hide_pen: bool = True
):
    """
    Teleport a turtle to exact coordinates with a specific heading angle.
    Use this to position the turtle precisely before drawing.

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param x: The x-coordinate, range: [0, 11]. 0 is left edge, 11 is right edge.
    :param y: The y-coordinate, range: [0, 11]. 0 is bottom edge, 11 is top edge.
    :param theta: Heading angle in radians. 0=right, π/2≈1.57=up, π≈3.14=left, 3π/2≈4.71=down
    :param hide_pen: If True (default), pen is turned off during teleport so no line is drawn
    """
    in_bounds, message = within_bounds(x, y)
    if not in_bounds:
        return message

    try:
        node = get_ros_node()
        if hide_pen:
            set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 1, "off": 1})
        mark_pose_stale(name)
        request = TeleportAbsolute.Request()
        request.x = float(x)
        request.y = float(y)
        request.theta = float(theta)
        node.call_service(
            TeleportAbsolute, f"/{normalize_name(name)}/teleport_absolute", request
        )
        if hide_pen:
            set_pen.invoke(
                {"name": name, "r": 30, "g": 30, "b": 255, "width": 1, "off": 0}
            )
        current_pose = get_turtle_pose.invoke({"names": [name]})

        return f"{name} new pose: ({current_pose[name].x}, {current_pose[name].y}) at {current_pose[name].theta} radians."
    except Exception as e:
        return f"Failed to teleport the turtle: {e}"


@tool
def teleport_relative(name: str, linear: float, angular: float):
    """
    Teleport a turtle relative to its current position and orientation.
    Use this to adjust heading without drawing, or to move without precise positioning.

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param linear: distance to move forward (positive) or backward (negative)
    :param angular: angle to rotate in RADIANS. Positive = counterclockwise, negative = clockwise
    """
    in_bounds, message = will_be_within_bounds(name, linear, 0.0, angular)
    if not in_bounds:
        return message

    try:
        node = get_ros_node()
        mark_pose_stale(name)
        request = TeleportRelative.Request()
        request.linear = float(linear)
        request.angular = float(angular)
        node.call_service(
            TeleportRelative, f"/{normalize_name(name)}/teleport_relative", request
        )
        current_pose = get_turtle_pose.invoke({"names": [name]})
        return f"{name} new pose: ({current_pose[name].x}, {current_pose[name].y}) at {current_pose[name].theta} radians."
    except Exception as e:
        return f"Failed to teleport the turtle: {e}"


@tool
def publish_twist_to_cmd_vel(
    name: str,
    velocity: float,
    lateral: float,
    angle: float,
    steps: int = 1,
):
    """
    Publish a Twist message to move the turtle. This DRAWS a line as the turtle moves.
    Each step represents a short fixed motion slice. The `velocity` and `angle`
    arguments are interpreted as the desired linear distance and heading change
    per step, rather than raw per-second rates.
    
    For STRAIGHT lines: set angle=0 and use velocity for distance per step.
    For CURVED lines: combine velocity and angle (creates an arc).
    For ROTATION only: set velocity=0 and use angle for turn amount per step.

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param velocity: desired forward distance per step. Positive=forward, negative=backward.
    :param lateral: desired lateral distance per step. Positive=left, negative=right. Usually 0.
    :param angle: desired heading change in radians per step. Positive=counterclockwise, negative=clockwise.
    :param steps: Number of motion slices to execute.
    """
    # Remove any forward slashes from the name
    name = normalize_name(name)

    # Check if the movement will keep the turtle within bounds
    in_bounds, message = will_be_within_bounds(
        name, velocity / COMMAND_STEP_SECONDS, lateral / COMMAND_STEP_SECONDS, angle / COMMAND_STEP_SECONDS,
        duration=steps * COMMAND_STEP_SECONDS
    )
    if not in_bounds:
        return message

    vel = Twist()
    vel.linear.x, vel.linear.y, vel.linear.z = (
        velocity / COMMAND_STEP_SECONDS,
        lateral / COMMAND_STEP_SECONDS,
        0.0,
    )
    vel.angular.x, vel.angular.y, vel.angular.z = 0.0, 0.0, angle / COMMAND_STEP_SECONDS

    try:
        node = get_ros_node()
        pub = cmd_vel_pubs.get(name) or node.ensure_publisher(name)
        mark_pose_stale(name)
        _trace_turtle(
            "ROS2 TOPIC PUBLISH PLAN",
            {
                "topic": f"/{name}/cmd_vel",
                "steps": steps,
                "step_seconds": COMMAND_STEP_SECONDS,
                "twist": {
                    "linear": {
                        "x": vel.linear.x,
                        "y": vel.linear.y,
                        "z": vel.linear.z,
                    },
                    "angular": {
                        "x": vel.angular.x,
                        "y": vel.angular.y,
                        "z": vel.angular.z,
                    },
                },
            },
        )

        for _ in range(steps):
            pub.publish(vel)
            time.sleep(COMMAND_STEP_SECONDS)
    except Exception as e:
        return f"Failed to publish {vel} to /{name}/cmd_vel: {e}"
    finally:
        current_pose = get_turtle_pose.invoke({"names": [name]})
        return (
            f"New Pose ({name}): x={current_pose[name].x}, y={current_pose[name].y}, "
            f"theta={current_pose[name].theta} rads, "
            f"linear_velocity={current_pose[name].linear_velocity}, "
            f"angular_velocity={current_pose[name].angular_velocity}."
        )


@tool
def stop_turtle(name: str):
    """
    Stop a turtle by publishing a Twist message with zero linear and angular velocities.

    :param name: name of the turtle
    """
    return publish_twist_to_cmd_vel.invoke(
        {
            "name": name,
            "velocity": 0.0,
            "lateral": 0.0,
            "angle": 0.0,
        }
    )


@tool
def reset_turtlesim():
    """
    Resets the turtlesim, removes all turtles, clears any markings, and creates a new default turtle at the center.
    """
    try:
        node = get_ros_node()
        node.call_service(Empty, "/reset", Empty.Request())

        # Clear the cmd_vel publishers
        global cmd_vel_pubs
        cmd_vel_pubs.clear()
        node.ensure_publisher("turtle1")
        node.ensure_pose_subscription("turtle1")

        return "Successfully reset the turtlesim environment. Ignore all previous commands, failures, and goals."
    except Exception as e:
        return f"Failed to reset the turtlesim environment: {e}"


@tool
def set_pen(name: str, r: int, g: int, b: int, width: int, off: int):
    """
    Control the turtle's pen for drawing lines.
    Turn pen OFF before teleporting to reposition without drawing.
    Turn pen ON before using publish_twist_to_cmd_vel to draw lines.

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param r: red value (0-255)
    :param g: green value (0-255)
    :param b: blue value (0-255)
    :param width: width of the pen line (1-5 recommended)
    :param off: 0 = pen ON (will draw), 1 = pen OFF (will not draw)
    """
    # Remove any forward slashes from the name
    name = normalize_name(name)
    try:
        node = get_ros_node()
        request = SetPen.Request()
        request.r = int(r)
        request.g = int(g)
        request.b = int(b)
        request.width = int(width)
        request.off = int(off)
        node.call_service(SetPen, f"/{name}/set_pen", request)
        return f"Successfully set the pen color for the turtle: {name}."
    except Exception as e:
        return f"Failed to set the pen color for the turtle: {e}"


@tool
def has_moved_to_expected_coordinates(
    name: str, expected_x: float, expected_y: float, tolerance: float = 0.1
) -> str:
    """
    Check if the turtle has moved to the expected position.

    :param name: name of the turtle
    :param expected_x: expected x-coordinate
    :param expected_y: expected y-coordinate
    :param tolerance: tolerance level for the comparison
    """
    current_pose = get_turtle_pose.invoke({"names": [name]})
    current_x = current_pose[name].x
    current_y = current_pose[name].y

    distance = ((current_x - expected_x) ** 2 + (current_y - expected_y) ** 2) ** 0.5
    if distance <= tolerance:
        return (
            f"{name} has moved to the expected position ({expected_x}, {expected_y})."
        )
    else:
        return f"{name} has NOT moved to the expected position ({expected_x}, {expected_y})."


@tool
def draw_line_segment(name: str, x1: float, y1: float, x2: float, y2: float) -> str:
    """
    Draw a single straight line from point (x1,y1) to point (x2,y2).
    
    This is a high-level convenience tool that automatically:
    1. Calculates the angle and distance needed
    2. Turns off the pen
    3. Teleports to the starting point with correct heading
    4. Turns on the pen
    5. Draws the line
    
    Use this instead of manually doing the calculate/teleport/draw sequence.

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param x1: starting x coordinate
    :param y1: starting y coordinate
    :param x2: ending x coordinate
    :param y2: ending y coordinate
    :return: status message with final position
    """
    from math import atan2, sqrt
    
    # Calculate angle and distance
    dx = x2 - x1
    dy = y2 - y1
    angle = atan2(dy, dx)
    distance = sqrt(dx**2 + dy**2)
    
    # Check bounds
    in_bounds_start, msg = within_bounds(x1, y1)
    if not in_bounds_start:
        return f"Start point {msg}"
    
    in_bounds_end, msg = within_bounds(x2, y2)
    if not in_bounds_end:
        return f"End point {msg}"
    
    # Turn off pen
    set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 2, "off": 1})
    
    # Teleport to start with correct angle
    teleport_absolute.invoke({"name": name, "x": x1, "y": y1, "theta": angle, "hide_pen": True})
    
    # Turn on pen
    set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 2, "off": 0})
    
    # Teleport with the pen on so turtlesim draws the exact segment without velocity drift.
    result = teleport_absolute.invoke(
        {"name": name, "x": x2, "y": y2, "theta": angle, "hide_pen": False}
    )
    
    return f"Line drawn from ({x1},{y1}) to ({x2},{y2}). {result}"


@tool
def draw_rectangle(
    name: str, x: float, y: float, width: float, height: float, filled: bool = False
) -> str:
    """
    Draw a perfect rectangle with exact corners and no angle drift.
    
    This tool automatically handles the complex workflow of teleporting to each edge
    with the exact angle to ensure perfectly straight lines and right angles.
    
    The rectangle is drawn counterclockwise starting from the bottom-left corner.

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param x: x coordinate of bottom-left corner
    :param y: y coordinate of bottom-left corner
    :param width: width of the rectangle (extends to the right)
    :param height: height of the rectangle (extends upward)
    :param filled: if True, fills the rectangle with horizontal lines (not just outline)
    :return: status message with rectangle bounds
    """
    # Check all corners are in bounds
    corners = [
        (x, y, "bottom-left"),
        (x + width, y, "bottom-right"),
        (x + width, y + height, "top-right"),
        (x, y + height, "top-left"),
    ]
    
    for cx, cy, corner_name in corners:
        in_bounds, msg = within_bounds(cx, cy)
        if not in_bounds:
            return f"Rectangle {corner_name} corner {msg}"
    
    # Draw exact edges with teleport services instead of velocity commands.
    draw_line_segment.invoke({"name": name, "x1": x, "y1": y, "x2": x + width, "y2": y})
    draw_line_segment.invoke({"name": name, "x1": x + width, "y1": y, "x2": x + width, "y2": y + height})
    draw_line_segment.invoke({"name": name, "x1": x + width, "y1": y + height, "x2": x, "y2": y + height})
    draw_line_segment.invoke({"name": name, "x1": x, "y1": y + height, "x2": x, "y2": y})
    
    # Fill if requested
    if filled:
        fill_step = 0.1  # Distance between fill lines
        y_current = y + fill_step
        while y_current < y + height:
            set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 1, "off": 1})
            teleport_absolute.invoke({"name": name, "x": x, "y": y_current, "theta": 0, "hide_pen": True})
            set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 1, "off": 0})
            teleport_absolute.invoke(
                {"name": name, "x": x + width, "y": y_current, "theta": 0, "hide_pen": False}
            )
            y_current += fill_step
    
    return f"Rectangle drawn: bottom-left=({x},{y}), width={width}, height={height}, filled={filled}"


@tool
def draw_polyline(name: str, points: List[tuple], closed: bool = False) -> str:
    """
    Draw a series of connected straight line segments through multiple points.
    
    This tool automatically calculates angles and distances for each segment
    and uses the proper teleport technique to ensure clean, precise lines.
    
    Example: draw_polyline('turtle1', [(2,2), (5,2), (5,5), (2,5)], closed=True)
    draws a square from (2,2) to (5,2) to (5,5) to (2,5) and back to (2,2).

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param points: list of (x, y) coordinate tuples to connect, e.g., [(1,1), (3,4), (5,2)]
    :param closed: if True, draws a final line segment back to the first point
    :return: status message with number of segments drawn
    """
    if len(points) < 2:
        return "Error: Need at least 2 points to draw a polyline"
    
    # Check all points are in bounds
    for i, (x, y) in enumerate(points):
        in_bounds, msg = within_bounds(x, y)
        if not in_bounds:
            return f"Point {i} at ({x},{y}) {msg}"
    
    # Draw each segment
    segments_drawn = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        draw_line_segment.invoke({"name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        segments_drawn += 1
    
    # Close the shape if requested
    if closed and len(points) > 2:
        x1, y1 = points[-1]
        x2, y2 = points[0]
        draw_line_segment.invoke({"name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        segments_drawn += 1
    
    return f"Polyline drawn with {segments_drawn} segments through {len(points)} points. Closed: {closed}"


@tool
def calculate_rectangle_bounds(x: float, y: float, width: float, height: float) -> dict:
    """
    Calculate all four corner coordinates of a rectangle.
    
    This is useful for planning layouts and checking for overlaps before drawing.
    Returns a dictionary with clearly labeled corner positions.

    :param x: x coordinate of bottom-left corner
    :param y: y coordinate of bottom-left corner
    :param width: width of the rectangle
    :param height: height of the rectangle
    :return: dict with 'bottom_left', 'bottom_right', 'top_left', 'top_right', 'center', and ranges
    """
    return {
        "bottom_left": (x, y),
        "bottom_right": (x + width, y),
        "top_right": (x + width, y + height),
        "top_left": (x, y + height),
        "center": (x + width/2, y + height/2),
        "x_range": (x, x + width),
        "y_range": (y, y + height),
        "width": width,
        "height": height,
    }


@tool
def check_rectangles_overlap(rect1: tuple, rect2: tuple) -> dict:
    """
    Check if two rectangles overlap or intersect.
    
    Essential for validating that doors, windows, and other components don't
    conflict with each other before drawing.
    
    Each rectangle is specified as (x, y, width, height) where (x,y) is the
    bottom-left corner.

    :param rect1: tuple of (x, y, width, height) for first rectangle
    :param rect2: tuple of (x, y, width, height) for second rectangle
    :return: dict with 'overlap' (bool), 'message', and 'details' about the overlap
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Calculate bounds
    r1_left = x1
    r1_right = x1 + w1
    r1_bottom = y1
    r1_top = y1 + h1
    
    r2_left = x2
    r2_right = x2 + w2
    r2_bottom = y2
    r2_top = y2 + h2
    
    # Check for overlap
    # Rectangles overlap if they overlap in both x and y dimensions
    x_overlap = not (r1_right <= r2_left or r2_right <= r1_left)
    y_overlap = not (r1_top <= r2_bottom or r2_top <= r1_bottom)
    
    overlap = x_overlap and y_overlap
    
    if overlap:
        # Calculate overlap region
        overlap_left = max(r1_left, r2_left)
        overlap_right = min(r1_right, r2_right)
        overlap_bottom = max(r1_bottom, r2_bottom)
        overlap_top = min(r1_top, r2_top)
        
        overlap_width = overlap_right - overlap_left
        overlap_height = overlap_top - overlap_bottom
        
        return {
            "overlap": True,
            "message": f"Rectangles overlap! Overlap region: ({overlap_left:.2f},{overlap_bottom:.2f}) to ({overlap_right:.2f},{overlap_top:.2f})",
            "details": {
                "overlap_region": {
                    "x": overlap_left,
                    "y": overlap_bottom,
                    "width": overlap_width,
                    "height": overlap_height,
                },
                "rect1_bounds": f"x:[{r1_left:.2f},{r1_right:.2f}] y:[{r1_bottom:.2f},{r1_top:.2f}]",
                "rect2_bounds": f"x:[{r2_left:.2f},{r2_right:.2f}] y:[{r2_bottom:.2f},{r2_top:.2f}]",
            }
        }
    else:
        return {
            "overlap": False,
            "message": "Rectangles do not overlap. Safe to draw both.",
            "details": {
                "rect1_bounds": f"x:[{r1_left:.2f},{r1_right:.2f}] y:[{r1_bottom:.2f},{r1_top:.2f}]",
                "rect2_bounds": f"x:[{r2_left:.2f},{r2_right:.2f}] y:[{r2_bottom:.2f},{r2_top:.2f}]",
            }
        }


@tool
def draw_circle(name: str, center_x: float, center_y: float, radius: float, segments: int = 36) -> str:
    """
    Draw a circle by approximating it with multiple small arc segments.
    
    The circle is drawn by moving the turtle in a circular path while the pen is down.
    More segments = smoother circle, but slower to draw. 36 segments usually looks good.
    
    Technical details: This uses the turtle's curved motion capability (velocity + angular velocity)
    to draw smooth circular arcs. The circle is drawn counterclockwise starting from the rightmost point.

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param center_x: x coordinate of circle center
    :param center_y: y coordinate of circle center
    :param radius: radius of the circle
    :param segments: number of segments to approximate the circle (default 36, more = smoother)
    :return: status message
    """
    from math import pi, cos, sin
    
    # Validate parameters
    if radius <= 0:
        return f"Radius must be positive, got {radius}"
    
    if segments <= 0:
        return f"Segments must be positive, got {segments}"
    
    # Check corners of bounding box
    for point_name, x, y in [
        ("center", center_x, center_y),
        ("rightmost", center_x + radius, center_y),
        ("leftmost", center_x - radius, center_y),
        ("topmost", center_x, center_y + radius),
        ("bottommost", center_x, center_y - radius),
    ]:
        in_bounds, msg = within_bounds(x, y)
        if not in_bounds:
            return f"Circle {point_name} point {msg}"
    
    # Calculate arc parameters
    # We'll draw the circle as small arcs
    # Arc length per segment = 2*pi*radius / segments
    # Angular velocity = 2*pi / total_time
    # Linear velocity = arc_length / time_per_segment
    
    angle_per_segment = 2 * pi / segments  # radians per segment
    arc_length_per_segment = 2 * pi * radius / segments
    
    # Start at rightmost point of circle (center_x + radius, center_y)
    # Heading should be tangent to circle = pi/2 (pointing up)
    start_x = center_x + radius
    start_y = center_y
    start_theta = pi / 2  # pointing up (tangent to circle at right side)
    
    # Move to start position
    set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 2, "off": 1})
    teleport_absolute.invoke({"name": name, "x": start_x, "y": start_y, "theta": start_theta, "hide_pen": True})
    set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 2, "off": 0})
    
    # Draw each segment
    for i in range(segments):
        publish_twist_to_cmd_vel.invoke({
            "name": name,
            "velocity": arc_length_per_segment,
            "lateral": 0,
            "angle": angle_per_segment,
            "steps": 1
        })
    
    return f"Circle drawn: center=({center_x},{center_y}), radius={radius}, segments={segments}"


@tool
def draw_arc(
    name: str,
    center_x: float,
    center_y: float,
    radius: float,
    start_angle: float,
    arc_angle: float,
    segments: int = 18
) -> str:
    """
    Draw an arc (part of a circle) from start_angle for arc_angle radians.
    
    This is perfect for drawing curved shapes like clouds, rainbows, or semicircles.
    The arc is drawn counterclockwise if arc_angle is positive, clockwise if negative.
    
    Examples:
    - Semicircle (top half): start_angle=0, arc_angle=π (3.14159)
    - Quarter circle: start_angle=0, arc_angle=π/2 (1.5708)
    - Cloud bump: start_angle=0, arc_angle=π (half circle)

    :param name: name of the turtle (without forward slash, e.g., 'turtle1')
    :param center_x: x coordinate of arc center
    :param center_y: y coordinate of arc center
    :param radius: radius of the arc
    :param start_angle: starting angle in radians (0=right, π/2=up, π=left, 3π/2=down)
    :param arc_angle: how many radians to sweep (positive=counterclockwise, negative=clockwise)
    :param segments: number of segments to approximate the arc (default 18)
    :return: status message
    """
    from math import pi, cos, sin, fabs
    
    # Validate parameters
    if radius <= 0:
        return f"Radius must be positive, got {radius}"
    
    if segments <= 0:
        return f"Segments must be positive, got {segments}"
    
    if fabs(arc_angle) < 0.01:
        return f"Arc angle too small: {arc_angle} radians"
    
    # Calculate start position on the arc
    start_x = center_x + radius * cos(start_angle)
    start_y = center_y + radius * sin(start_angle)
    
    # Check if start point is in bounds
    in_bounds, msg = within_bounds(start_x, start_y)
    if not in_bounds:
        return f"Arc start point {msg}"
    
    # Calculate end position to check bounds
    end_angle = start_angle + arc_angle
    end_x = center_x + radius * cos(end_angle)
    end_y = center_y + radius * sin(end_angle)
    
    in_bounds, msg = within_bounds(end_x, end_y)
    if not in_bounds:
        return f"Arc end point {msg}"
    
    # Calculate motion parameters
    angle_per_segment = arc_angle / segments
    arc_length_per_segment = fabs(arc_angle) * radius / segments
    
    # Start heading should be tangent to the circle
    # Tangent is perpendicular to radius, so add π/2 to start_angle
    start_theta = start_angle + (pi / 2 if arc_angle > 0 else -pi / 2)
    
    # Move to start position
    set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 2, "off": 1})
    teleport_absolute.invoke({"name": name, "x": start_x, "y": start_y, "theta": start_theta, "hide_pen": True})
    set_pen.invoke({"name": name, "r": 0, "g": 0, "b": 0, "width": 2, "off": 0})
    
    # Draw each segment
    for i in range(segments):
        publish_twist_to_cmd_vel.invoke({
            "name": name,
            "velocity": arc_length_per_segment,
            "lateral": 0,
            "angle": angle_per_segment,
            "steps": 1
        })
    
    return f"Arc drawn: center=({center_x},{center_y}), radius={radius}, start={start_angle:.2f}rad, sweep={arc_angle:.2f}rad"
