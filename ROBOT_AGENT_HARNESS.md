# ROS2 TurtleBot Agent Harness

For the current dependency boundary, complete environment variables, and
regression-test command, see `src/robot_agent/README.md`.

This is the first usable implementation of `deerflow_robotics_agent_design.html`.
It follows the design's MVP path: a DeerFlow-style lead-agent factory, explicit
runtime state/events/checkpoints, high-level TurtleBot tools, a ROS2 adapter,
and an LLM-generated Behavior Tree skill.

## File-to-reference mapping

| New file group | Reference used | Adapted responsibility |
| --- | --- | --- |
| `agents/lead_agent/*` | DeerFlow `agents/lead_agent/agent.py`, `prompt.py`; BTPG structured planning | `make_lead_agent`, advisory task planner, final tool assembly, sectioned system prompt |
| `middlewares/loop_detection.py` | DeerFlow loop detection middleware | Block repeated identical calls at the tool boundary |
| `runtime/*` | DeerFlow runtime lifecycle design | run ID, event journal, JSON checkpoint |
| `ros/adapter.py` | TurtleBot `tb3_behaviors/navigation.py`, `vision.py` | Nav2 `NavigateToPose` ActionClient, safe stop publisher, and camera bridge |
| `perception/color_detection.py` | TurtleBot `tb3_behaviors/vision.py` | Headless HSV/blob detection for red, green, and blue image-plane targets |
| `skills/behavior_tree.py` | BTPG `bt_generating_agent.py` | structured LLM plan, node validation, BT XML export |
| `tools/registry.py` | design tool contract + TurtleBot locations | high-level tools returning `ToolResult` |

## Run without a simulator

```bash
cd /Users/chenzhaosong/Downloads/rosa-main
chmod +x demo_robot_agent.sh
./demo_robot_agent.sh
```

Example goals:

```text
Go to location1.
Create a reusable behavior tree that visits location1 then location3 and stops.
Inspect the camera for a red object.
```

Each run first creates a short structured task plan. This is advisory context:
the lead agent still chooses tools, and the deterministic tool boundary validates
them. The default CLI backend is a dry run. `navigate_to` returns the concrete Nav2
command but does not execute it. Every run is written to `robot_agent_runs/<run_id>/`
as `events.jsonl` and `checkpoint.json`; a BT skill also writes JSON and XML.

## Execute against ROS2 later

When Nav2 and the TurtleBot simulator are ready:

```bash
ROBOT_AGENT_EXECUTE_ROS2=true ./demo_robot_agent.sh --execute-ros2
```

Use `ROBOT_AGENT_ROS_BACKEND=rclpy` to use the in-process Nav2 ActionClient
adapter instead of the transparent CLI adapter. The required topics/action can
be configured through the `ROBOT_AGENT_*` environment variables in
`config/settings.py`.

For TurtleBot camera inspection, use the `rclpy` backend and ensure the ROS2
environment contains `cv_bridge`, OpenCV, and a publisher on
`ROBOT_AGENT_CAMERA_TOPIC` (default `/camera/image_raw`). The detector only
reports colored blobs (`red`, `green`, `blue`); it does not claim object class
or world coordinates. The CLI backend deliberately returns a planned/blocked
result for camera inspection instead of fabricating detections.

## Current deliberate boundary

`inspect_for_color` is the only tool that reads camera data, and it stores a
small semantic detection list in the world model. `find_object` only queries
that stored evidence. Raw image and scan data are intentionally never exposed
to the LLM. LiDAR obstacle handling and arbitrary object classification are not
implemented in this MVP.

The generated XML follows the bundled TurtleBot demo's `SetLocations` and
`GoToPose` interface. `Wait` and `Stop` are declared custom nodes: the harness
can execute them now, but the older C++ BT demo needs matching plugins before
it can run those generated XML files directly.
