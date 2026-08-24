# YOLOE Prompt Assets

This directory contains only the text and visual prompts used by the YOLOE
detector. The canonical Gazebo world and models live in the ROS package at
`turtlebot3_behavior_demos/tb3_worlds`; they are not duplicated here.

The demo runs TurtleBot3 Waffle Pi. Its official Humble Gazebo model places
the ray sensor at `z=0.121 m` relative to `base_link`, whose fixed joint is at
`z=0.010 m`, so the simulated scan plane is approximately `0.131 m` above the
ground. The platform is `0.30 m` high and the extinguisher is spawned with its
base at `z=0.30 m`. Therefore LiDAR sees the platform as a navigation obstacle,
while the extinguisher body is above the scan plane and must be localized from
camera detections.

`yoloe_prompts/catalog.json` maps an open-vocabulary label to text descriptions
and an optional reference image. Relative image paths resolve from that JSON
file's directory.

## Run

From the repository root in WSL:

```bash
docker rm -f rosa-robot-agent-ros 2>/dev/null || true
./demo_robot_agent_ros_docker.sh
```

Then ask the agent:

```text
Navigate to inspection_start, find the fire extinguisher, then inspect it from four viewpoints.
```

## Manual Verification

1. Confirm Gazebo shows only the enclosing walls, central platform, robot, and
   extinguisher.
2. In RViz, verify `/scan` reports the platform boundary while the camera sees
   the extinguisher above it.
3. Run `ros2 topic echo /scan --once` to confirm scan data is active.
4. Run the inspection request and verify four files appear under the run's
   `object_views/` directory.

The prompt catalog, canonical world/model XML, and static-map footprint are
unit tested. Detector confidence and physical visibility still require the
simulator check above.
