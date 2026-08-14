# Robot Agent Harness

`robot_agent` is a DeerFlow-style LangChain v1 harness for high-level TurtleBot
tasks over ROS2. It keeps LLM planning/tool selection separate from deterministic
ROS transport, semantic state, goal verification, and safety controls.

## Environment boundary

`robot_agent` uses LangChain v1 `create_agent` and owns its model-provider
loader. Install the root package or `src/robot_agent/requirements.txt` in a
dedicated virtual environment or use the provided Docker launchers.

The TurtleBot world supplies named navigation poses from
`turtlebot3_behavior_demos/tb3_worlds/maps/sim_house_locations.yaml` unless
`ROBOT_AGENT_LOCATION_FILE` overrides it.

The launcher assumes this repository layout because `cli.py` resolves the
project root with `Path(__file__).resolve().parents[2]`:

```text
repository/
├── demo_robot_agent.sh
├── src/robot_agent/
└── turtlebot3_behavior_demos/
```

## Run

Create the isolated Python environment once:

```bash
python3 -m venv .venv-robot-agent
source .venv-robot-agent/bin/activate
python -m pip install -r src/robot_agent/requirements.txt
```

Dry run, with ROS2 commands planned but not sent:

```bash
./demo_robot_agent.sh
```

Or run the same dry-run agent in its isolated Docker environment:

```bash
chmod +x demo_robot_agent_docker.sh
./demo_robot_agent_docker.sh
```

The script mounts the repository at `/app`, passes the root `.env` at runtime,
and keeps `robot_agent_runs/` on the host. After the first successful build,
reuse the image without reinstalling dependencies:

```bash
SKIP_BUILD=true ./demo_robot_agent_docker.sh
```

Dry-run navigation persists `robot_state.last_planned_pose` and reports it as
`dry_run_planned_pose`, but deliberately leaves confirmed `pose` and
`visited_locations` unchanged. A later run in the same session can inspect the
planned target without confusing it with Nav2-confirmed arrival. Only a real
successful navigation updates confirmed pose and visited locations; the goal
monitor therefore never marks a planned-only action as completed.

Real ROS2 execution with the in-process Nav2/camera backend:

```bash
ROBOT_AGENT_EXECUTE_ROS2=true \
ROBOT_AGENT_ROS_BACKEND=rclpy \
./demo_robot_agent.sh --execute-ros2 --ros-backend rclpy
```

### Integrated TurtleBot3 simulation

This is the primary end-to-end run path. It starts TurtleBot3 Gazebo, Nav2,
AMCL, RViz, the camera pipeline, and the interactive agent in one container.

Prerequisites:

- Docker Desktop is running.
- On Windows, run these commands inside the `Ubuntu-22.04` WSL terminal.
- WSLg is available (`echo "$DISPLAY"` and `ls /mnt/wslg` should succeed).
- The repository root `.env` contains the configured LLM credentials.

First build and launch:

```bash
cd ~/rosa-main
chmod +x demo_robot_agent_ros_docker.sh run_robot_agent_ros.sh
./demo_robot_agent_ros_docker.sh
```

GUI mode is the default: Gazebo and RViz open through WSLg on the Windows
desktop. Startup waits for active Nav2 and AMCL lifecycle nodes, `/odom`,
`/scan`, `/navigate_to_pose`, and the `map -> base_link` transform. The agent
prompt appears after this message:

```text
ROS2 simulation is ready. Agent commands will now execute through rclpy/Nav2.
ROS2 TurtleBot agent. Type `exit` to quit.
```

The first build can take several minutes. Reuse the completed images on later
runs without rebuilding:

```bash
SKIP_BUILD=true ./demo_robot_agent_ros_docker.sh
```

Use headless mode only when a display is intentionally unavailable:

```bash
ROBOT_AGENT_GUI=false SKIP_BUILD=true ./demo_robot_agent_ros_docker.sh
```

Suggested agent checks:

```text
Navigate to location1.
Navigate to location2, then go to location3.
Navigate to x=1.0, y=2.0, yaw=0.0.
Report the robot state.
Inspect for blue objects, then find a blue object.
Search for a blue object while moving through location1, location2, and location3.
Stop the robot.
```

Named targets are defined in
`turtlebot3_behavior_demos/tb3_worlds/maps/sim_house_locations.yaml`. Coordinate
navigation uses the `map` frame and is bounded by the configured workspace
limits.

While the container is running, inspect ROS2 from another WSL terminal:

```bash
docker exec -it rosa-robot-agent-ros bash -lc '
source /opt/ros/humble/setup.bash
source /turtlebot3_ws/install/setup.bash
source /overlay_ws/install/setup.bash
ros2 lifecycle get /amcl
ros2 lifecycle get /bt_navigator
ros2 action list
ros2 topic list
timeout 5 ros2 run tf2_ros tf2_echo map base_link
'
```

View simulator startup logs:

```bash
docker exec -it rosa-robot-agent-ros tail -f /tmp/turtlebot-demo-world.log
```

Type `exit` at the agent prompt to stop the agent and remove its temporary
container. If the prompt is unavailable, run this from WSL:

```bash
docker rm -f rosa-robot-agent-ros
```

Camera perception uses `/camera/image_raw`. Active perception backoff/retry is
disabled by default. `search_for_object` is a separate continuous-search tool:
each route leg remains one uninterrupted Nav2 action while the configured
detector checks the latest frame at a fixed time interval. A match cancels the
active Nav2 goal and prevents later route locations from running. This requires
the `rclpy` backend; CLI/dry-run mode only plans the first navigation leg and
does not invoke camera detection.

## Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROBOT_AGENT_SESSION_ID` | `default` | Select persistent semantic session state |
| `ROBOT_AGENT_RUN_DIRECTORY` | `robot_agent_runs` | Run journals, checkpoints, and session state |
| `ROBOT_AGENT_LOCATION_FILE` | TurtleBot demo YAML | Named Nav2 target poses |
| `ROBOT_AGENT_EXECUTE_ROS2` | `false` | Enable real transport execution |
| `ROBOT_AGENT_ROS_BACKEND` | `cli` | `cli` or `rclpy` transport |
| `ROBOT_AGENT_NAV_ACTION` | `/navigate_to_pose` | Nav2 action name |
| `ROBOT_AGENT_CMD_VEL_TOPIC` | `/cmd_vel` | Stop/recovery velocity topic |
| `ROBOT_AGENT_ODOM_TOPIC` | `/odom` | Odometry source |
| `ROBOT_AGENT_CAMERA_TOPIC` | `/camera/image_raw` | Camera source |
| `ROBOT_AGENT_MAP_FRAME` | `map` | Navigation frame |
| `ROBOT_AGENT_BASE_FRAME` | `base_link` | Robot frame used for confirmed map pose from TF |
| `ROBOT_AGENT_TOOL_TIMEOUT_SEC` | `30` | ROS operation timeout |
| `ROBOT_AGENT_LOOP_WARN_THRESHOLD` | `3` | Identical-call warning threshold |
| `ROBOT_AGENT_REPEATED_TOOL_LIMIT` | `5` | Identical-call hard stop |
| `ROBOT_AGENT_MAX_TOOL_CALLS` | `12` | LangGraph tool-call budget |
| `ROBOT_AGENT_MAX_CONTINUATIONS` | `12` | Plan continuation budget |
| `ROBOT_AGENT_MAX_NO_PROGRESS_CONTINUATIONS` | `3` | Semantic no-progress stop |
| `ROBOT_AGENT_BT_NAVIGATION_RETRIES` | `1` | Retryable Nav2 attempts per BT node |
| `ROBOT_AGENT_ACTIVE_PERCEPTION_RETRY` | `false` | Enable one backoff + camera retry |
| `ROBOT_AGENT_ACTIVE_PERCEPTION_BACKOFF_SPEED` | `-0.05` | Recovery linear velocity in m/s |
| `ROBOT_AGENT_ACTIVE_PERCEPTION_BACKOFF_DURATION_SEC` | `0.5` | Recovery duration |
| `ROBOT_AGENT_DETECTOR_BACKEND` | `color_blob` | Moving-search detector: `color_blob`, `yolo`, or fail-fast `vlm` placeholder |
| `ROBOT_AGENT_DETECTION_INTERVAL_SEC` | `1.0` | Seconds between detector calls while Nav2 is moving |
| `ROBOT_AGENT_DETECTION_CONFIDENCE_THRESHOLD` | `0.8` | Minimum confidence required to cancel navigation and report a match |
| `ROBOT_AGENT_CENTER_ON_DETECTION` | `true` | Rotate in place after detection until the target is horizontally centered |
| `ROBOT_AGENT_IMAGE_CENTER_TOLERANCE` | `0.08` | Allowed normalized horizontal error around image center |
| `ROBOT_AGENT_CENTERING_MAX_ANGULAR_SPEED` | `0.2` | Maximum visual-centering rotation speed in rad/s |
| `ROBOT_AGENT_CENTERING_GAIN` | `0.5` | Proportional gain from image error to angular velocity |
| `ROBOT_AGENT_CENTERING_TIMEOUT_SEC` | `10.0` | Maximum time allowed for post-detection centering |
| `ROBOT_AGENT_YOLO_MODEL` | `yolov8n.pt` | Ultralytics model used when detector backend is `yolo` |
| `ROBOT_AGENT_YOLO_INPUT_SIZE` | `640` | Bounded YOLO inference image size |
| `ROBOT_AGENT_WORKSPACE_MIN_X/MAX_X` | `-10/10` | Safe map X bounds |
| `ROBOT_AGENT_WORKSPACE_MIN_Y/MAX_Y` | `-10/10` | Safe map Y bounds |
| `ROBOT_AGENT_GUI` | `true` | Show Gazebo and RViz through WSLg; set `false` to use Xvfb headless mode |

## Tests

Focused parity regression tests use only the Python standard library test
runner:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/test_robot_agent -v
```

Real Nav2, camera, and Gazebo integration still require a sourced ROS2 runtime;
unit tests use fake adapters and never send robot commands.

The `1.0` second moving-search cadence is intentionally conservative because
YOLO, Nav2, AMCL, and Gazebo share compute during a full simulation. Tune it
only after observing inference latency and GPU utilization on the target host.
