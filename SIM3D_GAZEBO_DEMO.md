# SIM3D Gazebo Demo

This is the minimal Gazebo migration path for the current SIM3D dry-run agent.

The goal is to replace the fake listener node with a real Gazebo receiver while keeping the agent/tool interface the same.

## Communication Shape

```text
agent
-> ROS2 /cmd_vel
-> ros_gz_bridge
-> Gazebo /cmd_vel
-> Gazebo DiffDrive plugin
-> Gazebo /odom and /tf
-> ros_gz_bridge
-> ROS2 /odom and /tf
```

## Files

- `Dockerfile.sim3d_gazebo`: ROS2 Humble + Gazebo Harmonic + ros_gz bridge image.
- `demo_3d_gazebo_docker.sh`: builds/starts the container.
- `demo_3d_gazebo.sh`: starts Gazebo headless server and bridge inside the container.
- `sim3d_gazebo/simple_diff_drive.sdf`: minimal differential-drive robot world.

## Install And Start Container

From repo root on the host:

```bash
cd /Users/chenzhaosong/Downloads/rosa-main
chmod +x demo_3d_gazebo_docker.sh demo_3d_gazebo.sh
./demo_3d_gazebo_docker.sh
```

The first build will be large because it installs Gazebo Harmonic and `ros-humble-ros-gzharmonic`.

If the image already exists:

```bash
SKIP_BUILD=true ./demo_3d_gazebo_docker.sh
```

## Terminal A: Start Gazebo Receiver

Inside the `sim3d-gazebo` container:

```bash
cd /app
./demo_3d_gazebo.sh
```

This starts:

- Gazebo Harmonic server in headless mode
- `ros_gz_bridge` for `/cmd_vel`, `/odom`, and `/tf`

## Terminal B: Run Agent

From the host:

```bash
docker exec -it sim3d-gazebo bash -lc "source /opt/ros/humble/setup.bash && cd /app && SIM3D_EXECUTE_ROS2=true ./demo_3d_dryrun.sh"
```

Example prompt:

```text
go forward 1 meter
```

## Terminal C: Inspect Feedback

From the host:

```bash
docker exec -it sim3d-gazebo bash -lc "source /opt/ros/humble/setup.bash && cd /app && python3 src/sim3d_dryrun/watch_gazebo_odom.py --print-period-sec 0.5"
```

This prints compact pose updates continuously:

```text
{"pose": {"x": 0.42, "y": 0.0, "yaw": 0.0}, "velocity": {"linear_x": 0.2, "angular_z": 0.0}}
```

For a single raw odometry message:

```bash
docker exec -it sim3d-gazebo bash -lc "source /opt/ros/humble/setup.bash && ros2 topic echo /odom nav_msgs/msg/Odometry --once"
```

```bash
docker exec -it sim3d-gazebo bash -lc "source /opt/ros/humble/setup.bash && ros2 run tf2_ros tf2_echo odom base_link"
```

## What Should Stay The Same

The agent still publishes standard ROS2 commands:

```text
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist ...
```

The receiver is Gazebo DiffDrive. It subscribes to `/cmd_vel`.

## Notes

This demo does not use Nav2 yet.

That is intentional. It first validates the low-level mobile base interface:

```text
/cmd_vel in
/odom and /tf out
```

After this works, Nav2 can be added on top for goal-based navigation.

## Odom Echo Note

For bridged topics, the ROS2 CLI may occasionally fail to infer the message type quickly enough:

```text
Could not determine the type for the passed topic
```

Use the explicit type when that happens:

```bash
ros2 topic echo /odom nav_msgs/msg/Odometry --once
```
