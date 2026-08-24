ARG ROS_DISTRO=humble

FROM osrf/ros:${ROS_DISTRO}-desktop AS base
ARG ROS_DISTRO
ARG COLCON_PARALLEL_WORKERS=2
ENV ROS_DISTRO=${ROS_DISTRO} \
    COLCON_PARALLEL_WORKERS=${COLCON_PARALLEL_WORKERS} \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    TURTLEBOT3_MODEL=waffle_pi \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    NO_AT_BRIDGE=1
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libcanberra-gtk-module \
        libcanberra-gtk3-module \
        libqt5svg5-dev \
        python3-opencv \
        python3-pip \
        python3-pyqt5.qtwebengine \
        python3-tk \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir matplotlib transforms3d

RUN mkdir -p /turtlebot3_ws/src /tmp/runtime-root \
    && chmod 0700 /tmp/runtime-root
WORKDIR /turtlebot3_ws/src
COPY dependencies.repos .
RUN vcs import < dependencies.repos

WORKDIR /turtlebot3_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && apt-get update \
    && rosdep install --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} -y \
    && colcon build --symlink-install --parallel-workers ${COLCON_PARALLEL_WORKERS} \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

FROM base AS overlay
RUN mkdir -p /overlay_ws/src
WORKDIR /overlay_ws

# Resolve ROS dependencies from package metadata before copying frequently
# edited worlds, launch files, and scripts. This expensive network layer stays
# cached unless package.xml itself changes.
COPY tb3_worlds/package.xml /tmp/tb3_worlds/package.xml
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    source /turtlebot3_ws/install/setup.bash \
    && apt-get update \
    && rosdep install --from-paths /tmp/tb3_worlds \
        --ignore-src --rosdistro ${ROS_DISTRO} -y \
    && rm -rf /var/lib/apt/lists/*

COPY tb3_worlds/ ./src/tb3_worlds/
RUN source /turtlebot3_ws/install/setup.bash \
    && colcon build --symlink-install --parallel-workers ${COLCON_PARALLEL_WORKERS} \
        --event-handlers console_cohesion+

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh /overlay_ws/src/tb3_worlds/scripts/*.py
ENTRYPOINT ["/entrypoint.sh"]
