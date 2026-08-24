#!/usr/bin/env bash
set -euo pipefail

export ROBOT_AGENT_TOOL_TIMEOUT_SEC="${ROBOT_AGENT_TOOL_TIMEOUT_SEC:-120}"
export ROBOT_AGENT_SCENE="${ROBOT_AGENT_SCENE:-extinguisher_room}"
if [[ "${ROBOT_AGENT_SCENE}" != "extinguisher_room" ]]; then
    echo "Error: supported scene is extinguisher_room, got ${ROBOT_AGENT_SCENE}." >&2
    exit 1
fi
scene_root=/app/turtlebot3_behavior_demos/tb3_worlds
scene_world="${scene_root}/worlds/extinguisher_room.world"
scene_map="${scene_root}/maps/extinguisher_room_map.yaml"
export ROBOT_AGENT_LOCATION_FILE="${ROBOT_AGENT_LOCATION_FILE:-${scene_root}/maps/extinguisher_room_locations.yaml}"
export ROBOT_AGENT_CENTER_ON_DETECTION="${ROBOT_AGENT_CENTER_ON_DETECTION:-true}"

# The repository is bind-mounted at /app, while ament resolves tb3_worlds from
# the image's overlay. Refresh runtime-only package resources so a synchronized
# world/launch/script edit cannot be paired with stale files when SKIP_BUILD is
# used. package.xml/CMake/dependency changes still require a real image build.
tb3_install_prefix="$(ros2 pkg prefix tb3_worlds)"
tb3_install_share="${tb3_install_prefix}/share/tb3_worlds"
tb3_install_lib="${tb3_install_prefix}/lib/tb3_worlds"
for resource_directory in launch maps models worlds; do
    mkdir -p "${tb3_install_share}/${resource_directory}"
    cp -a \
        "${scene_root}/${resource_directory}/." \
        "${tb3_install_share}/${resource_directory}/"
done
mkdir -p "${tb3_install_lib}"
for executable in "${scene_root}"/scripts/*.py; do
    install -m 0755 "${executable}" "${tb3_install_lib}/$(basename "${executable}")"
done
echo "Refreshed tb3_worlds runtime resources from /app"

initial_x="${ROBOT_AGENT_INITIAL_X:--3.5}"
initial_y="${ROBOT_AGENT_INITIAL_Y:--3.5}"
initial_yaw="${ROBOT_AGENT_INITIAL_YAW:-0.0}"
object_x="${ROBOT_AGENT_OBJECT_X:-0.0}"
object_y="${ROBOT_AGENT_OBJECT_Y:-0.0}"
object_yaw="${ROBOT_AGENT_OBJECT_YAW:-0.0}"

if [[ "${ROBOT_AGENT_GUI:-false}" == "true" ]]; then
    : "${DISPLAY:?ROBOT_AGENT_GUI=true requires DISPLAY}"
    echo "GUI mode enabled on DISPLAY=${DISPLAY}"
    rviz_config="$(ros2 pkg prefix turtlebot3_navigation2)/share/turtlebot3_navigation2/rviz/tb3_navigation2.rviz"
    if [[ ! -f "${rviz_config}" ]]; then
        echo "Error: TurtleBot3 RViz configuration not found: ${rviz_config}" >&2
        exit 1
    fi
    robot_agent_rviz_config=/tmp/robot-agent-navigation.rviz
    python /app/src/robot_agent/runtime/rviz_config.py \
        "${rviz_config}" \
        "${robot_agent_rviz_config}"
    if ! grep -Fq 'Value: /camera/image_raw' "${robot_agent_rviz_config}" || \
       ! grep -Fq 'Value: /camera/yoloe_annotated' "${robot_agent_rviz_config}"; then
        echo "Error: Failed to configure the two RViz camera displays." >&2
        exit 1
    fi
    cp "${robot_agent_rviz_config}" "${rviz_config}"
    echo "RViz right side configured with Raw Image and YOLOE Annotated"
    xvfb_pid=""
else
    export DISPLAY="${DISPLAY:-:99}"
    Xvfb "${DISPLAY}" -screen 0 1280x800x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
    xvfb_pid=$!
fi
ros2 launch tb3_worlds tb3_demo_world.launch.py \
    world:="${scene_world}" \
    map:="${scene_map}" \
    x_pose:="${initial_x}" \
    y_pose:="${initial_y}" \
    yaw_pose:="${initial_yaw}" \
    object_x:="${object_x}" \
    object_y:="${object_y}" \
    object_yaw:="${object_yaw}" \
    >/tmp/turtlebot-demo-world.log 2>&1 &
simulation_pid=$!
tf_probe_pid=""

cleanup() {
    for process_id in \
        "${tf_probe_pid}" \
        "${simulation_pid}" \
        "${xvfb_pid}"; do
        if [[ -n "${process_id}" ]]; then
            kill "${process_id}" 2>/dev/null || true
            wait "${process_id}" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

# Keep one TF listener alive during startup. Recreating tf2_echo for every
# readiness poll discards its DDS discovery and TF buffer, which can report a
# false negative forever on a busy simulator even while map -> base_link exists.
: >/tmp/robot-agent-map-tf.log
stdbuf -oL -eL ros2 run tf2_ros tf2_echo map base_link \
    >/tmp/robot-agent-map-tf.log 2>&1 &
tf_probe_pid=$!

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
    if [[ "${amcl_active}" == "true" && \
          "${odom_ready}" == "true" && \
          "${scan_ready}" == "true" && \
          "${tf_ready}" == "false" ]] && \
       (( attempt - last_initial_pose_attempt >= 10 )); then
        echo "AMCL is active but map TF is missing; publishing the initial pose..."
        python3 /app/turtlebot3_behavior_demos/tb3_worlds/scripts/set_init_amcl_pose.py \
            --ros-args \
            -r __node:=robot_agent_initial_pose_retry \
            -p use_sim_time:=true \
            -p x:="${initial_x}" \
            -p y:="${initial_y}" \
            -p theta:="${initial_yaw}" \
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
exec python /app/src/robot_agent/cli.py
