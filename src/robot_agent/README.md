# Robot Agent Harness

`robot_agent` is a DeerFlow-style LangChain v1 harness for high-level TurtleBot
tasks over ROS2. It keeps LLM planning/tool selection separate from deterministic
ROS transport, semantic state, goal verification, and safety controls.

## Environment boundary

The repository's root `pyproject.toml` belongs to legacy ROSA and pins LangChain
0.3. `robot_agent` uses LangChain v1 `create_agent`; install
`src/robot_agent/requirements.txt` in a separate virtual environment or ROS2
container and run it with `PYTHONPATH=src` as shown below. Installing
`robot_agent` with `pip install robot_agent` is not supported. Do not upgrade
the root ROSA environment in place.

Two dependencies remain external to this package:

- `turtle_agent.scripts.llm.get_llm` supplies the configured chat model.
- `turtlebot3_behavior_demos/tb3_worlds/maps/sim_house_locations.yaml` supplies
  named navigation poses unless `ROBOT_AGENT_LOCATION_FILE` overrides it.

The launcher assumes this repository layout because `cli.py` resolves the
project root with `Path(__file__).resolve().parents[2]`:

```text
repository/
├── demo_robot_agent.sh
├── src/robot_agent/
├── src/turtle_agent/
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

For a self-contained real-execution test against TurtleBot3 Gazebo + Nav2 +
AMCL, build and launch the integrated ROS2 image:

```bash
chmod +x demo_robot_agent_ros_docker.sh run_robot_agent_ros.sh
./demo_robot_agent_ros_docker.sh
```

By default this starts Gazebo and RViz under Xvfb (no visible GUI). It waits for
the required Nav2 and AMCL lifecycle nodes, `/odom`, `/scan`, the
`/navigate_to_pose` action, and the `map -> base_link` transform before starting
the agent with `execute_ros2=true` and the `rclpy` backend. A successful Nav2
result updates confirmed pose and visited locations. Reuse the completed image
with:

```bash
SKIP_BUILD=true ./demo_robot_agent_ros_docker.sh
```

On WSLg, expose the Gazebo and RViz windows on the Windows desktop with:

```bash
ROBOT_AGENT_GUI=true SKIP_BUILD=true ./demo_robot_agent_ros_docker.sh
```

Camera perception additionally requires OpenCV, `cv_bridge`, and a publisher on
`/camera/image_raw`. Active perception backoff/retry is disabled by default.

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
| `ROBOT_AGENT_WORKSPACE_MIN_X/MAX_X` | `-10/10` | Safe map X bounds |
| `ROBOT_AGENT_WORKSPACE_MIN_Y/MAX_Y` | `-10/10` | Safe map Y bounds |
| `ROBOT_AGENT_GUI` | `false` | Use WSLg instead of Xvfb in the integrated ROS Docker launcher |

## Tests

Focused parity regression tests use only the Python standard library test
runner:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/test_robot_agent -v
```

Real Nav2, camera, and Gazebo integration still require a sourced ROS2 runtime;
unit tests use fake adapters and never send robot commands.
