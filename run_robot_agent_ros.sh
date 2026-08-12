#!/usr/bin/env bash
set -euo pipefail

export ROBOT_AGENT_EXECUTE_ROS2=true
export ROBOT_AGENT_ROS_BACKEND=rclpy
export ROBOT_AGENT_TOOL_TIMEOUT_SEC="${ROBOT_AGENT_TOOL_TIMEOUT_SEC:-120}"
export ROBOT_AGENT_LOCATION_FILE="${ROBOT_AGENT_LOCATION_FILE:-/app/turtlebot3_behavior_demos/tb3_worlds/maps/sim_house_locations.yaml}"

if [[ "${ROBOT_AGENT_GUI:-false}" == "true" ]]; then
    : "${DISPLAY:?ROBOT_AGENT_GUI=true requires DISPLAY}"
    echo "GUI mode enabled on DISPLAY=${DISPLAY}"
    xvfb_pid=""
else
    export DISPLAY="${DISPLAY:-:99}"
    Xvfb "${DISPLAY}" -screen 0 1280x800x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    xvfb_pid=$!
fi
ros2 launch tb3_worlds tb3_demo_world.launch.py >/tmp/turtlebot-demo-world.log 2>&1 &
simulation_pid=$!

cleanup() {
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
    local tf_output
    tf_output="$(timeout 2 ros2 run tf2_ros tf2_echo map base_link 2>&1 || true)"
    grep -q -- 'Translation:' <<<"${tf_output}"
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
    tail -n 100 /tmp/turtlebot-demo-world.log >&2
    exit 1
fi

echo "ROS2 simulation is ready. Agent commands will now execute through rclpy/Nav2."
exec python /app/src/robot_agent/cli.py --execute-ros2 --ros-backend rclpy
