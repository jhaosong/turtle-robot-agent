# Fire Extinguisher Simulation Asset

This directory contains a self-contained primitive fire-extinguisher model.
The demo spawn pose is `x=4.0`, `y=2.5`, `z=0.0`. The named robot observation
pose `fire_extinguisher_station` is one meter south of it at `x=4.0`, `y=1.5`,
facing north toward the model.

The repository's `tb3_demo_world.launch.py` automatically spawns this model
instead of the former random red/green/blue boxes. The commands below are for
using the same asset in another world or simulator stack.

## Gazebo Classic

From the repository root:

```bash
export GAZEBO_MODEL_PATH="$PWD/sim_assets/models:${GAZEBO_MODEL_PATH:-}"
ros2 run gazebo_ros spawn_entity.py \
  -entity fire_extinguisher \
  -file "$PWD/sim_assets/models/fire_extinguisher/model.sdf" \
  -x 4.0 -y 2.5 -z 0.0
```

## New Gazebo / Ignition

```bash
export GZ_SIM_RESOURCE_PATH="$PWD/sim_assets/models:${GZ_SIM_RESOURCE_PATH:-}"
ros2 run ros_gz_sim create \
  -name fire_extinguisher \
  -file "$PWD/sim_assets/models/fire_extinguisher/model.sdf" \
  -x 4.0 -y 2.5 -z 0.0
```

Alternatively include it in an SDF world:

```xml
<include>
  <uri>model://fire_extinguisher</uri>
  <pose>4.0 2.5 0.0 0 0 1.5708</pose>
</include>
```

## Manual Verification

1. Start the normal TurtleBot demo, or spawn the model manually with the
   command matching another installed simulator.
2. Confirm the red extinguisher is standing at `(4.0, 2.5)` in the Gazebo GUI.
3. Open the `/camera/image_raw` display in RViz, or run
   `ros2 topic hz /camera/image_raw`, and place the robot at
   `fire_extinguisher_station`.
4. Run `search_for_object` with `label="fire extinguisher"` through a route
   containing `fire_extinguisher_station`.
5. Tune `ROBOT_AGENT_TARGET_BOX_SIZE_NORMALIZED` if the desired stopping
   distance differs from the default `0.35` for the actual camera FOV.

XML well-formedness is unit tested. Visual appearance, detector confidence,
and stopping distance require this manual simulator check.
