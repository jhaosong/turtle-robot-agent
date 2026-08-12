import json
import math
import os
import subprocess
from typing import Dict, Optional

from langchain_core.tools import tool


def _trace(label: str, payload):
    if os.getenv("SIM3D_TRACE", "true").strip().lower() not in ("1", "true", "yes", "on"):
        return
    print(f"\n[SIM3D TRACE] {label}")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def _compact_trace() -> bool:
    return os.getenv("SIM3D_TRACE_COMPACT", "true").strip().lower() in ("1", "true", "yes", "on")


def _emit_tool_trace(tool_name: str, result: dict):
    if _compact_trace():
        compact = {
            "tool": tool_name,
            "type": result.get("type"),
            "command": result.get("command"),
            "executed": result.get("executed"),
            "returncode": result.get("returncode"),
        }
        _trace("ROS2 COMMAND", compact)
    else:
        _trace(f"TOOL RESULT {tool_name}", result)


def _execute_ros2_command(command: str, execute_by_default: bool = True) -> dict:
    enabled = os.getenv("SIM3D_EXECUTE_ROS2", "false").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        return {
            "executed": False,
            "execution_mode": "dry_run",
        }
    inspection_enabled = os.getenv("SIM3D_EXECUTE_INSPECTION", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not execute_by_default and not inspection_enabled:
        return {
            "executed": False,
            "execution_mode": "inspection_skipped",
        }

    shell_prefix = os.getenv("SIM3D_ROS2_SHELL_PREFIX", "").strip()
    timeout_sec = float(os.getenv("SIM3D_ROS2_TIMEOUT_SEC", "120"))
    final_command = command if not shell_prefix else f"{shell_prefix} {json.dumps(command)}"

    try:
        completed = subprocess.run(
            final_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "executed": True,
            "execution_mode": "shell",
            "shell_prefix": shell_prefix or None,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "executed": True,
            "execution_mode": "shell",
            "shell_prefix": shell_prefix or None,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
            "timeout_sec": timeout_sec,
            "error": "timeout",
        }


def _yaw_to_quaternion(yaw: float) -> Dict[str, float]:
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(yaw / 2.0),
        "w": math.cos(yaw / 2.0),
    }


def _twist_yaml(linear_x: float, angular_z: float) -> str:
    return (
        f"{{linear: {{x: {linear_x:.3f}, y: 0.0, z: 0.0}}, "
        f"angular: {{x: 0.0, y: 0.0, z: {angular_z:.3f}}}}}"
    )


def _cmd_vel_command(
    linear_x: float,
    angular_z: float,
    duration_sec: float,
    topic_name: str = "/cmd_vel",
    rate_hz: int = 2,
) -> dict:
    duration_sec = max(float(duration_sec), 0.1)
    rate_hz = max(int(rate_hz), 1)
    requested_times = max(1, math.ceil(duration_sec * rate_hz))
    max_times = int(os.getenv("SIM3D_CMD_VEL_MAX_TIMES", "0"))
    publish_times = requested_times if max_times <= 0 else min(requested_times, max_times)
    motion_command = (
        f"ros2 topic pub --times {publish_times} {topic_name} geometry_msgs/msg/Twist "
        f'"{_twist_yaml(linear_x, angular_z)}" '
        f"--rate {rate_hz}"
    )
    stop_times = int(os.getenv("SIM3D_CMD_VEL_STOP_TIMES", "2"))
    needs_stop = abs(linear_x) > 1e-9 or abs(angular_z) > 1e-9
    stop_command = (
        f"ros2 topic pub --times {max(stop_times, 1)} {topic_name} geometry_msgs/msg/Twist "
        f'"{_twist_yaml(0.0, 0.0)}" '
        f"--rate {rate_hz}"
    )
    should_append_stop = needs_stop and stop_times > 0
    command = f"{motion_command} && {stop_command}" if should_append_stop else motion_command
    return {
        "command": command,
        "motion_command": motion_command,
        "stop_command": stop_command if should_append_stop else None,
        "duration_sec": duration_sec,
        "requested_times": requested_times,
        "times": publish_times,
        "max_times": max_times if max_times > 0 else None,
        "stop_times": stop_times if should_append_stop else 0,
        "rate_hz": rate_hz,
        "topic_name": topic_name,
        "twist": {"linear_x": linear_x, "angular_z": angular_z},
    }


@tool
def generate_nav2_goal_command(
    x: float,
    y: float,
    yaw: float = 0.0,
    frame_id: str = "map",
    action_name: str = "/navigate_to_pose",
) -> dict:
    """Generate a plausible ros2 action send_goal command for a standard Nav2 NavigateToPose server."""
    quat = _yaw_to_quaternion(yaw)
    yaml_goal = (
        "{pose: {header: {frame_id: '"
        + frame_id
        + "'}, pose: {position: {x: "
        + f"{x:.3f}, y: {y:.3f}, z: 0.0"
        + "}, orientation: {x: "
        + f"{quat['x']:.6f}, y: {quat['y']:.6f}, z: {quat['z']:.6f}, w: {quat['w']:.6f}"
        + "}}}}"
    )
    command = (
        f"ros2 action send_goal {action_name} nav2_msgs/action/NavigateToPose "
        f'"{yaml_goal}"'
    )
    result = {
        "type": "nav2_goal",
        "command": command,
        "goal": {"x": x, "y": y, "yaw": yaw, "frame_id": frame_id},
    }
    result.update(_execute_ros2_command(command))
    _emit_tool_trace("generate_nav2_goal_command", result)
    return result


@tool
def generate_cmd_vel_command(
    linear_x: float,
    angular_z: float,
    duration_sec: float = 1.0,
    topic_name: str = "/cmd_vel",
    rate_hz: int = 2,
) -> dict:
    """Generate a plausible ros2 topic pub command for geometry_msgs/Twist velocity control on a simple mobile base."""
    result = {
        "type": "cmd_vel",
        **_cmd_vel_command(linear_x, angular_z, duration_sec, topic_name, rate_hz),
    }
    result.update(_execute_ros2_command(result["command"]))
    _emit_tool_trace("generate_cmd_vel_command", result)
    return result


@tool
def generate_drive_distance_command(
    distance_m: float,
    linear_x: float = 0.2,
    topic_name: str = "/cmd_vel",
    rate_hz: int = 2,
) -> dict:
    """Generate and optionally execute /cmd_vel commands to drive a requested straight-line distance, then stop."""
    distance_m = float(distance_m)
    speed = abs(float(linear_x)) or 0.2
    signed_speed = math.copysign(speed, distance_m)
    duration_sec = abs(distance_m) / speed
    result = {
        "type": "drive_distance",
        "distance_m": distance_m,
        **_cmd_vel_command(signed_speed, 0.0, duration_sec, topic_name, rate_hz),
    }
    result.update(_execute_ros2_command(result["command"]))
    _emit_tool_trace("generate_drive_distance_command", result)
    return result


@tool
def generate_circle_motion_command(
    radius_m: float,
    revolutions: float = 1.0,
    linear_x: float = 0.2,
    clockwise: bool = False,
    topic_name: str = "/cmd_vel",
    rate_hz: int = 2,
) -> dict:
    """Generate and optionally execute /cmd_vel commands to drive a circular arc, then stop."""
    radius_m = max(abs(float(radius_m)), 0.01)
    revolutions = max(abs(float(revolutions)), 0.01)
    speed = abs(float(linear_x)) or 0.2
    angular_z = speed / radius_m
    if clockwise:
        angular_z *= -1
    duration_sec = (2.0 * math.pi * radius_m * revolutions) / speed
    result = {
        "type": "circle_motion",
        "radius_m": radius_m,
        "revolutions": revolutions,
        "clockwise": clockwise,
        **_cmd_vel_command(speed, angular_z, duration_sec, topic_name, rate_hz),
    }
    result.update(_execute_ros2_command(result["command"]))
    _emit_tool_trace("generate_circle_motion_command", result)
    return result


@tool
def generate_tf_echo_command(
    target_frame: str = "map",
    source_frame: str = "base_link",
) -> dict:
    """Generate a plausible ros2 tf2_echo inspection command."""
    command = f"ros2 run tf2_ros tf2_echo {target_frame} {source_frame}"
    result = {
        "type": "tf_echo",
        "command": command,
        "target_frame": target_frame,
        "source_frame": source_frame,
    }
    result.update(_execute_ros2_command(command, execute_by_default=False))
    _emit_tool_trace("generate_tf_echo_command", result)
    return result


@tool
def generate_topic_echo_command(
    topic_name: str,
    message_type: Optional[str] = None,
    once: bool = True,
) -> dict:
    """Generate a plausible ros2 topic echo command for inspecting a topic."""
    command = f"ros2 topic echo {topic_name}"
    if once:
        command += " --once"
    result = {
        "type": "topic_echo",
        "command": command,
        "topic_name": topic_name,
        "message_type_hint": message_type,
    }
    result.update(_execute_ros2_command(command, execute_by_default=False))
    _emit_tool_trace("generate_topic_echo_command", result)
    return result


@tool
def generate_service_call_command(
    service_name: str,
    service_type: str,
    request_yaml: str = "{}",
) -> dict:
    """Generate a plausible ros2 service call command."""
    command = f"ros2 service call {service_name} {service_type} '{request_yaml}'"
    result = {
        "type": "service_call",
        "command": command,
        "service_name": service_name,
        "service_type": service_type,
        "request_yaml": request_yaml,
    }
    result.update(_execute_ros2_command(command, execute_by_default=False))
    _emit_tool_trace("generate_service_call_command", result)
    return result


def get_tools():
    return [
        generate_nav2_goal_command,
        generate_drive_distance_command,
        generate_circle_motion_command,
        generate_cmd_vel_command,
        generate_tf_echo_command,
        generate_topic_echo_command,
        generate_service_call_command,
    ]


def get_tool_summary() -> str:
    lines = []
    for tool_fn in get_tools():
        lines.append(f"- {tool_fn.name}: {tool_fn.description}")
    return "\n".join(lines)
