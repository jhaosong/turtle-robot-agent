# Turtle Robot Agent

A robotics agent built with LangChain/LangGraph, ROS 2 Humble, Nav2, Gazebo,
TurtleBot3, and YOLOE. Users provide natural-language tasks, the agent plans and
selects high-level tools, and the ROS 2 adapter communicates with Nav2, TF,
camera, and velocity-control interfaces.

The current release supports one scene: `extinguisher_room`, a 10 x 10 meter
room with a central platform and a fire extinguisher. The default launcher
starts Gazebo, RViz, Nav2, AMCL, the YOLOE camera overlay, and the interactive
agent.

## Recommended Environment

- Windows 11
- WSL2 with Ubuntu 22.04
- WSLg for Gazebo and RViz
- Docker Desktop with WSL integration enabled for Ubuntu
- At least 8 GB RAM; 16 GB is recommended
- Internet access for the first build and initial YOLOE model setup

The GUI launcher connects to the Windows desktop through `/mnt/wslg`, so GUI
mode should be started from a WSL Ubuntu terminal. Do not run Docker launcher
commands from inside an existing container.

## 1. Prepare WSL And Docker

Install or verify WSL from Windows PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

After installing Docker Desktop, open:

```text
Docker Desktop -> Settings -> Resources -> WSL Integration
```

Enable integration for `Ubuntu-22.04`, then enter WSL:

```powershell
wsl.exe -d Ubuntu-22.04
```

Verify Docker and WSLg from the WSL terminal:

```bash
docker version
echo "$DISPLAY"
test -d /mnt/wslg && echo "WSLg ready"
```

`docker version` should show both Client and Server information. `DISPLAY` is
usually `:0`.

## 2. Clone The Repository

For better Docker build performance, clone into the WSL Linux filesystem rather
than `/mnt/c`:

```bash
cd ~
git clone https://github.com/jhaosong/turtle-robot-agent.git rosa-main
cd ~/rosa-main
chmod +x demo_robot_agent_ros_docker.sh run_robot_agent_ros.sh
```

## 3. Configure The LLM

Create `.env` in the repository root. This file is excluded by `.gitignore` and
must never be committed.

### Azure OpenAI

Minimal configuration for the Azure v1-compatible endpoint:

```bash
cat > .env <<'EOF'
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=replace-with-your-key
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=YOUR-DEPLOYMENT-NAME
AZURE_OPENAI_API_VERSION=
EOF

chmod 600 .env
```

When `AZURE_OPENAI_API_VERSION` is empty, the agent uses Azure's
OpenAI-compatible `/openai/v1/` endpoint. To use the legacy Azure API, set this
variable to an API version supported by your deployment.

### OpenAI

```bash
cat > .env <<'EOF'
LLM_PROVIDER=openai
OPENAI_API_KEY=replace-with-your-key
OPENAI_MODEL=gpt-5.2
EOF

chmod 600 .env
```

## 4. First Build And Run

Run from the repository root in WSL:

```bash
cd ~/rosa-main
docker rm -f rosa-robot-agent-ros 2>/dev/null || true
./demo_robot_agent_ros_docker.sh
```

GUI mode is enabled by default. The first run builds the TurtleBot3 simulation
and agent images and installs ROS, PyTorch, Ultralytics, LangChain, and related
dependencies, so it can take a significant amount of time.

When startup succeeds, the terminal displays:

```text
ROS2 simulation is ready. Agent commands will now execute through rclpy/Nav2.
ROS2 TurtleBot agent. Type `exit` to quit.
>
```

Gazebo displays the simulation world. RViz displays the map, Nav2 paths, raw
camera stream, and YOLOE-annotated camera stream. The first perception task may
download and initialize YOLOE assets; later runs reuse the local files.

## 5. Test Tasks

Start with state and navigation tasks:

```text
Report the robot state.
Navigate to inspection_start.
Navigate to x=2.0, y=-2.0, yaw=1.57.
Move back for 1 meter.
Stop the robot.
```

Test detection during navigation:

```text
Search for the fire extinguisher in the room.
Search for the fire extinguisher on the way to north_view.
```

Test localization and four-viewpoint capture:

```text
Find the fire extinguisher, localize it using bearing triangulation, then photograph it from 4 evenly spaced viewpoints around it.
```

Captured views are written to:

```text
robot_agent_runs/<run_id>/object_views/
```

Enter `exit` to leave the agent, or press `Ctrl+C` to stop the simulation
container.

## 6. Later Runs

After both images have been built successfully, skip the build step:

```bash
cd ~/rosa-main
docker rm -f rosa-robot-agent-ros 2>/dev/null || true
SKIP_BUILD=true ./demo_robot_agent_ros_docker.sh
```

The following changes normally do not require a rebuild:

- `src/robot_agent/**/*.py`
- Prompts and configuration defaults
- Gazebo worlds, maps, models, launch files, and runtime scripts
- Runtime RViz configuration

Remove `SKIP_BUILD=true` after changing any of the following:

- `Dockerfile.robot_agent_ros`
- `turtlebot3_behavior_demos/docker/Dockerfile`
- `src/robot_agent/requirements.txt`
- A ROS `package.xml` or `CMakeLists.txt`

## 7. Headless Mode

When WSLg is unavailable or no GUI is required:

```bash
ROBOT_AGENT_GUI=false ./demo_robot_agent_ros_docker.sh
```

For later runs:

```bash
ROBOT_AGENT_GUI=false SKIP_BUILD=true ./demo_robot_agent_ros_docker.sh
```

## 8. Inspect ROS 2

Keep the agent container running and open another WSL terminal:

```bash
docker exec -it rosa-robot-agent-ros bash -lc '
source /opt/ros/humble/setup.bash
source /turtlebot3_ws/install/setup.bash
source /overlay_ws/install/setup.bash
ros2 topic list
'
```

Inspect localization, TF, and the annotated camera topic:

```bash
docker exec -it rosa-robot-agent-ros bash -lc '
source /opt/ros/humble/setup.bash
source /turtlebot3_ws/install/setup.bash
source /overlay_ws/install/setup.bash
ros2 topic echo /amcl_pose --once
'

docker exec -it rosa-robot-agent-ros bash -lc '
source /opt/ros/humble/setup.bash
source /turtlebot3_ws/install/setup.bash
source /overlay_ws/install/setup.bash
timeout 5 ros2 run tf2_ros tf2_echo map base_link
'

docker exec -it rosa-robot-agent-ros bash -lc '
source /opt/ros/humble/setup.bash
source /turtlebot3_ws/install/setup.bash
source /overlay_ws/install/setup.bash
ros2 topic info /camera/yoloe_annotated --verbose
'
```

## 9. Common Problems

### `docker: command not found`

Docker Desktop is not running, or WSL integration is not enabled for Ubuntu.
Do not run the launcher from inside a container.

### The container name is already in use

```bash
docker rm -f rosa-robot-agent-ros
```

Then run the launcher again.

### `SKIP_BUILD=true but image does not exist`

Do not use `SKIP_BUILD=true` on the first run:

```bash
./demo_robot_agent_ros_docker.sh
```

### The GUI does not appear

```bash
echo "$DISPLAY"
echo "$WAYLAND_DISPLAY"
ls /mnt/wslg
```

If `/mnt/wslg` does not exist, update WSL and start the project from a WSLg
session on the Windows desktop.

### Startup waits for AMCL or map TF

Inspect the simulation log from another WSL terminal:

```bash
docker exec -it rosa-robot-agent-ros \
  tail -n 150 /tmp/turtlebot-demo-world.log
```

Also verify that `/odom`, `/scan`, and `map -> base_link` are available. If the
simulation is unhealthy, stop and recreate the container instead of launching a
second simulation inside the existing container.

### The first build is very slow

The ROS workspace, PyTorch, and YOLOE dependencies are large. Avoid repeatedly
clearing the Docker build cache. Use `SKIP_BUILD=true` for ordinary Python and
world-file changes.

## Architecture

```text
Natural-language goal
        |
LangChain/LangGraph lead agent
        |
High-level tools
        |
ROS 2 rclpy adapter
        |
Nav2 / TF / camera / cmd_vel
        |
TurtleBot3 in Gazebo
```

See [ROBOT_AGENT_FUNCTIONS.md](ROBOT_AGENT_FUNCTIONS.md) for the main files and
tool APIs. See [src/robot_agent/README.md](src/robot_agent/README.md) for agent
internals and environment-variable reference.
