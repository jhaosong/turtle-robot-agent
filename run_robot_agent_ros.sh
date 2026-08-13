#!/usr/bin/env bash
set -euo pipefail

export ROBOT_AGENT_EXECUTE_ROS2=true
export ROBOT_AGENT_ROS_BACKEND=rclpy
export ROBOT_AGENT_TOOL_TIMEOUT_SEC="${ROBOT_AGENT_TOOL_TIMEOUT_SEC:-120}"
export ROBOT_AGENT_LOCATION_FILE="${ROBOT_AGENT_LOCATION_FILE:-/app/turtlebot3_behavior_demos/tb3_worlds/maps/sim_house_locations.yaml}"

if [[ "${ROBOT_AGENT_GUI:-false}" == "true" ]]; then
    : "${DISPLAY:?ROBOT_AGENT_GUI=true requires DISPLAY}"
    echo "GUI mode enabled on DISPLAY=${DISPLAY}"
    rviz_config="$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/rviz/tb3_navigation2.rviz"
    if [[ ! -f "${rviz_config}" ]]; then
        echo "Error: TurtleBot3 RViz configuration not found: ${rviz_config}" >&2
        exit 1
    fi
    # TurtleBot3's Humble config ships a disabled Realsense group pointed at
    # an obsolete topic. Adapt that display to this demo's Gazebo camera.
    sed -i -z \
        -e 's|Value: /intel_realsense_r200_depth/image_raw|Value: /camera/image_raw|' \
        -e 's|      Enabled: false\n      Name: Realsense|      Enabled: true\n      Name: Realsense|' \
        -e 's|Name: RealsenseCamera|Name: Image|' \
        -e 's|Name: Realsense\n|Name: Camera\n|' \
        -e 's|\n  X: [0-9-]*\n  Y: [0-9-]*\n*$|\n  X: 0\n  Y: 0\n|' \
        "${rviz_config}"
    if ! grep -Fq 'Value: /camera/image_raw' "${rviz_config}" || \
       ! grep -B1 -F 'Name: Camera' "${rviz_config}" | grep -Fq 'Enabled: true'; then
        echo "Error: Failed to configure the RViz camera display." >&2
        exit 1
    fi
    echo "RViz camera display enabled on /camera/image_raw"
    xvfb_pid=""
else
    export DISPLAY="${DISPLAY:-:99}"
    Xvfb "${DISPLAY}" -screen 0 1280x800x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    xvfb_pid=$!
fi
ros2 launch tb3_worlds tb3_demo_world.launch.py >/tmp/turtlebot-demo-world.log 2>&1 &
simulation_pid=$!

# Keep one TF listener alive during startup. Recreating tf2_echo for every
# readiness poll discards its DDS discovery and TF buffer, which can report a
# false negative forever on a busy simulator even while map -> base_link exists.
: >/tmp/robot-agent-map-tf.log
stdbuf -oL -eL ros2 run tf2_ros tf2_echo map base_link \
    >/tmp/robot-agent-map-tf.log 2>&1 &
tf_probe_pid=$!

cleanup() {
    kill "${tf_probe_pid}" 2>/dev/null || true
    wait "${tf_probe_pid}" 2>/dev/null || true
    kill "${simulation_pid}" 2>/dev/null || true
    wait "${simulation_pid}" 2>/dev/null || true
    if [[ -n "${xvfb_pid}" ]]; then
        kill "${xvfb_pid}" 2>/dev/null || true
        wait "${xvfb_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

required_lifecycle_nodes=(
    /map_server
    /amcl
    /planner_server
    /controller_server
    /bt_navigator
)

lifecycle_node_is_active() {
    local lifecycle_state
    lifecycle_state="$(timeout 3 ros2 lifecycle get "$1" 2>/dev/null || true)"
    [[ "${lifecycle_state}" == active\ * ]]
}

map_transform_is_ready() {
    grep -q -- 'Translation:' /tmp/robot-agent-map-tf.log
}

echo "Waiting for Gazebo and active Nav2/AMCL lifecycle nodes..."
ready=false
last_initial_pose_attempt=-10
: >/tmp/initial-pose-retry.log
for attempt in $(seq 1 180); do
    if ! kill -0 "${simulation_pid}" 2>/dev/null; then
        echo "Error: TurtleBot simulation exited during startup." >&2
        tail -n 100 /tmp/turtlebot-demo-world.log >&2
        exit 1
    fi
    lifecycle_ready=true
    amcl_active=false
    inactive_nodes=()
    for node in "${required_lifecycle_nodes[@]}"; do
        if lifecycle_node_is_active "${node}"; then
            if [[ "${node}" == "/amcl" ]]; then
                amcl_active=true
            fi
        else
            lifecycle_ready=false
            inactive_nodes+=("${node}")
        fi
    done
    action_output="$(timeout 3 ros2 action list 2>/dev/null || true)"
    topic_output="$(timeout 3 ros2 topic list 2>/dev/null || true)"
    action_ready=false
    odom_ready=false
    scan_ready=false
    tf_ready=false
    if grep -Fxq "/navigate_to_pose" <<<"${action_output}"; then
        action_ready=true
    fi
    if grep -Fxq "/odom" <<<"${topic_output}"; then
        odom_ready=true
    fi
    if grep -Fxq "/scan" <<<"${topic_output}"; then
        scan_ready=true
    fi
    if [[ "${odom_ready}" == "true" ]] && map_transform_is_ready; then
        tf_ready=true
    fi
    if [[ "${amcl_active}" == "true" && "${tf_ready}" == "false" ]] && \
       (( attempt - last_initial_pose_attempt >= 10 )); then
        echo "AMCL is active but map TF is missing; publishing the initial pose..."
        python3 /app/turtlebot3_behavior_demos/tb3_worlds/scripts/set_init_amcl_pose.py \
            --ros-args \
            -r __node:=robot_agent_initial_pose_retry \
            -p use_sim_time:=true \
            >>/tmp/initial-pose-retry.log 2>&1 &
        last_initial_pose_attempt=${attempt}
    fi
    if [[ "${lifecycle_ready}" == "true" && \
          "${action_ready}" == "true" && \
          "${odom_ready}" == "true" && \
          "${scan_ready}" == "true" && \
          "${tf_ready}" == "true" ]]; then
        ready=true
        break
    fi
    if (( attempt % 10 == 0 )); then
        inactive_summary="${inactive_nodes[*]:-none}"
        echo "Readiness: lifecycle=${lifecycle_ready} inactive=[${inactive_summary}] action=${action_ready} odom=${odom_ready} scan=${scan_ready} tf=${tf_ready}"
    fi
    sleep 1
done

if [[ "${ready}" != "true" ]]; then
    echo "Error: ROS2 navigation lifecycle was not ready after 180 seconds." >&2
    for node in "${required_lifecycle_nodes[@]}"; do
        echo "--- ${node} ---" >&2
        ros2 lifecycle get "${node}" >&2 || true
    done
    echo "--- map -> base_link TF probe ---" >&2
    tail -n 50 /tmp/robot-agent-map-tf.log >&2
    tail -n 100 /tmp/turtlebot-demo-world.log >&2
    exit 1
fi

echo "ROS2 simulation is ready. Agent commands will now execute through rclpy/Nav2."
exec python /app/src/robot_agent/cli.py --execute-ros2 --ros-backend rclpy
