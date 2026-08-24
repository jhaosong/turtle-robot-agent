# Robot Agent Harness

`robot_agent` is a DeerFlow-style LangChain v1 harness for high-level TurtleBot
tasks over ROS2. It keeps LLM planning/tool selection separate from deterministic
ROS transport, semantic state, goal verification, and safety controls.

## Environment boundary

`robot_agent` uses LangChain v1 `create_agent` and owns its model-provider
loader. The supported runtime is the integrated ROS 2 Docker launcher.

YOLOE-26 is the perception backend and loads lazily on the first semantic
search. Its first run downloads the model and MobileCLIP text encoder.

The default 10 x 10 meter extinguisher room supplies search poses from
`turtlebot3_behavior_demos/tb3_worlds/maps/extinguisher_room_locations.yaml` unless
`ROBOT_AGENT_LOCATION_FILE` overrides it.

The launcher assumes this repository layout because `cli.py` resolves the
project root with `Path(__file__).resolve().parents[2]`:

```text
repository/
├── demo_robot_agent_ros_docker.sh
├── run_robot_agent_ros.sh
├── src/robot_agent/
└── turtlebot3_behavior_demos/
```

## Run

The launcher starts TurtleBot3 Gazebo, Nav2,
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

Agent Python, prompts, configuration, and the runtime resources from
`tb3_worlds` are refreshed from the bind-mounted repository at startup. Those
edits do not require an image rebuild. Rebuild after changing
`requirements.txt`, either Dockerfile, `package.xml`, or `CMakeLists.txt`. The
ROS overlay keeps dependency installation in a stable cached layer.

Use headless mode only when a display is intentionally unavailable:

```bash
ROBOT_AGENT_GUI=false SKIP_BUILD=true ./demo_robot_agent_ros_docker.sh
```

Suggested agent checks for the default room:

```text
Navigate to inspection_start.
Navigate to x=1.0, y=2.0, yaw=0.0.
Report the robot state.
Search for a fire extinguisher in the room.
Find the fire extinguisher, localize it, and photograph it from four sides.
Stop the robot.
```

Named targets are defined in
`turtlebot3_behavior_demos/tb3_worlds/maps/extinguisher_room_locations.yaml`.
The catalog contains only room coverage poses (`inspection_start`, `east_view`,
`north_view`, and `west_view`); it deliberately does not expose the
extinguisher coordinate. Coordinate
navigation uses the `map` frame and is bounded by the configured workspace
limits.

The multi-view task uses two high-level tools. `search_for_object` runs YOLOE
during Nav2 motion and cancels the active route after a confirmed detection.
`circle_object_for_inspection` converts the first bbox center to a bearing,
asks Nav2 to evaluate several left/right baseline candidates with
`ComputePathToPose` and the global costmap, moves to the best safe candidate,
and triangulates the object from a second bearing and the two measured robot
poses. It plans the full deterministic evenly spaced orbit once, then internally
visits, aligns once, and photographs each pose. If one pose is blocked, it checks
only a small fixed angular-nudge fan before failing safely.
Images are saved under the run's `object_views/` directory. A missing
detection, degenerate triangulation, unsafe candidate, or failed Nav2 action
stops the tool instead of falling back to blind motion.

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

Camera perception uses `/camera/image_raw`. The rclpy adapter republishes a
smooth RViz stream on `/camera/yoloe_annotated`; during `search_for_object`,
the latest short-lived YOLOE boxes, labels, and confidence values are drawn on
that stream. A one-frame candidate is retained for the overlay TTL instead of
being erased by the next empty inference result. GUI startup derives a focused
RViz configuration from the installed
TurtleBot navigation config: the right side contains separate `Raw Image` and
`YOLOE Annotated` displays, while the old Selection, Tool Properties, Views,
Selector, Docking, and Realsense panes are removed.
`search_for_object` is a separate continuous-search tool:
each route leg remains one uninterrupted Nav2 action while the configured
detector checks the latest frame at a fixed time interval. A match cancels the
active Nav2 goal and prevents later route locations from running. After a detection-triggered Nav2
cancellation, the adapter publishes zero velocity throughout a short handoff
window. The integrated extinguisher room defaults
`ROBOT_AGENT_CENTER_ON_DETECTION=false`, so safe distance selection belongs to
the active-view planner rather than bbox-size approach. If visual alignment is
explicitly enabled, it runs in two explicit
closed-loop phases: rotation-only control first centers the bbox for several
consecutive frames; translation-only control then approaches or retreats until
the bbox reaches its target height. Horizontal drift during approach stops the
robot and returns control to the rotation phase.

## Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROBOT_AGENT_SESSION_ID` | `default` | Select persistent semantic session state |
| `ROBOT_AGENT_RUN_DIRECTORY` | `robot_agent_runs` | Run journals, checkpoints, and session state |
| `ROBOT_AGENT_LOCATION_FILE` | TurtleBot demo YAML | Named Nav2 target poses |
| `ROBOT_AGENT_NAV_ACTION` | `/navigate_to_pose` | Nav2 action name |
| `ROBOT_AGENT_COMPUTE_PATH_ACTION` | `/compute_path_to_pose` | Read-only Nav2 path-planning action used to score candidates |
| `ROBOT_AGENT_GLOBAL_COSTMAP_SERVICE` | `/global_costmap/get_costmap` | Costmap query used to reject occupied or risky paths |
| `ROBOT_AGENT_CMD_VEL_TOPIC` | `/cmd_vel` | Stop/recovery velocity topic |
| `ROBOT_AGENT_CAMERA_TOPIC` | `/camera/image_raw` | Camera source |
| `ROBOT_AGENT_ANNOTATED_CAMERA_TOPIC` | `/camera/yoloe_annotated` | Smooth RViz stream with current search detections drawn as boxes |
| `ROBOT_AGENT_MAP_FRAME` | `map` | Navigation frame |
| `ROBOT_AGENT_BASE_FRAME` | `base_link` | Robot frame used for confirmed map pose from TF |
| `ROBOT_AGENT_TOOL_TIMEOUT_SEC` | `30` | ROS operation timeout |
| `ROBOT_AGENT_LOOP_WARN_THRESHOLD` | `3` | Identical-call warning threshold |
| `ROBOT_AGENT_REPEATED_TOOL_LIMIT` | `5` | Identical-call hard stop |
| `ROBOT_AGENT_MAX_TOOL_CALLS` | `12` | LangGraph tool-call budget |
| `ROBOT_AGENT_MAX_CONTINUATIONS` | `12` | Plan continuation budget |
| `ROBOT_AGENT_MAX_NO_PROGRESS_CONTINUATIONS` | `3` | Semantic no-progress stop |
| `ROBOT_AGENT_BT_NAVIGATION_RETRIES` | `1` | Retryable Nav2 attempts per BT node |
| `ROBOT_AGENT_DETECTION_INTERVAL_SEC` | `0.2` | Target seconds between detector calls while Nav2 is moving (5 Hz, bounded by inference speed) |
| `ROBOT_AGENT_DETECTION_BOX_THRESHOLD` | `0.05` | Minimum YOLOE confidence shown as a bounding box |
| `ROBOT_AGENT_DETECTION_CONFIDENCE_THRESHOLD` | `0.05` | Minimum confidence required to cancel navigation and report a match |
| `ROBOT_AGENT_DETECTION_TRACKING_CONFIDENCE_THRESHOLD` | `0.01` | Lower confidence accepted only after a target is confirmed, for maintaining visual lock |
| `ROBOT_AGENT_DETECTION_TRACKING_MAX_CENTER_JUMP` | `0.25` | Maximum normalized bbox-center movement accepted between tracking samples |
| `ROBOT_AGENT_DETECTION_CONFIRMATION_FRAMES` | `1` | Matching frames required before canceling Nav2; one frame prevents edge detections from leaving view |
| `ROBOT_AGENT_CENTER_ON_DETECTION` | generic `true`; integrated room `false` | Optional legacy bbox centering/approach after search |
| `ROBOT_AGENT_IMAGE_CENTER_TOLERANCE` | `0.10` | Allowed normalized horizontal error; accepts the middle 40%–60% image region |
| `ROBOT_AGENT_CENTERING_MAX_ANGULAR_SPEED` | `0.25` | Maximum visual-centering rotation speed in rad/s |
| `ROBOT_AGENT_CENTERING_MIN_ANGULAR_SPEED` | `0.10` | Minimum nonzero turn command outside the center deadband |
| `ROBOT_AGENT_CENTERING_GAIN` | `0.8` | Proportional gain from image error to angular velocity |
| `ROBOT_AGENT_TARGET_BOX_SIZE_NORMALIZED` | `0.4` | Desired target bbox height as a fraction of image height after horizontal centering |
| `ROBOT_AGENT_BOX_SIZE_TOLERANCE` | `0.05` | Allowed normalized bbox-height error |
| `ROBOT_AGENT_CENTERING_MAX_LINEAR_SPEED` | `0.25` | Maximum visual approach/retreat speed in m/s |
| `ROBOT_AGENT_CENTERING_MIN_LINEAR_SPEED` | `0.08` | Minimum nonzero approach/retreat command outside the size deadband |
| `ROBOT_AGENT_CENTERING_LINEAR_GAIN` | `1.0` | Proportional gain from bbox-size error to linear velocity |
| `ROBOT_AGENT_CENTERING_STABLE_FRAMES` | `3` | Consecutive in-tolerance frames required for each alignment phase |
| `ROBOT_AGENT_CENTERING_TIMEOUT_SEC` | `30.0` | Maximum time allowed for post-detection centering and approach |
| `ROBOT_AGENT_CENTERING_DETECTION_HOLD_SEC` | `1.0` | Grace period for reacquiring a lost target; zero velocity is sent throughout and stale boxes never drive motion |
| `ROBOT_AGENT_POST_CANCEL_SETTLE_SEC` | `0.5` | Zero-velocity handoff and terminal settling window |
| `ROBOT_AGENT_YOLOE_MODEL` | `yoloe-26s-seg.pt` | Ultralytics open-vocabulary detector model |
| `ROBOT_AGENT_YOLO_INPUT_SIZE` | `640` | Bounded YOLOE inference image size |
| `ROBOT_AGENT_CAMERA_HORIZONTAL_FOV_RAD` | `1.085595` | Waffle Pi Gazebo camera FOV used to convert bbox center to map bearing |
| `ROBOT_AGENT_TRIANGULATION_MIN_BASELINE_M` | `0.25` | Minimum displacement accepted for two-bearing localization |
| `ROBOT_AGENT_TRIANGULATION_MIN_RAY_ANGLE_DEG` | `3.0` | Minimum non-degenerate bearing-ray angle |
| `ROBOT_AGENT_TRIANGULATION_MIN_CONFIDENCE` | `0.25` | Minimum geometry-derived localization confidence |
| `ROBOT_AGENT_BASELINE_CANDIDATE_RADIUS_M` | `0.75` | Radius of the bearing-relative candidate fan |
| `ROBOT_AGENT_BASELINE_ASSUMED_OBJECT_DISTANCE_M` | `1.5` | Assumed range used to orient baseline candidates toward one provisional object point |
| `ROBOT_AGENT_BASELINE_SCORE_ALPHA/BETA/GAMMA` | `3.0/0.35/2.0` | Tangential, path-length, and costmap-risk score weights |
| `ROBOT_AGENT_BASELINE_NAV2_CANDIDATE_COUNT` | `4` | Top cheap-score candidates checked first with Nav2 |
| `ROBOT_AGENT_INSPECTION_RADIUS_M` | `2.0` | Default radius for object-centric viewpoints |
| `ROBOT_AGENT_INSPECTION_MIN_RADIUS_M` | `2.0` | Minimum radius retained when bbox geometry suggests a closer view |
| `ROBOT_AGENT_SCENE` | `extinguisher_room` | Integrated 10 x 10 Gazebo scene |
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

The `0.2` second moving-search target cadence is bounded by actual YOLOE
inference speed because YOLOE, Nav2, AMCL, and Gazebo share compute. Tune it
only after observing inference latency and GPU utilization on the target host.
