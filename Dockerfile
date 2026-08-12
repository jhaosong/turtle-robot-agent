FROM osrf/ros:humble-desktop AS rosa-ros2
LABEL authors="Rob Royce"

ENV DEBIAN_FRONTEND=noninteractive
ENV HEADLESS=false
ARG DEVELOPMENT=false

# Install linux packages
RUN apt-get update && apt-get install -y \
    ros-humble-turtlesim \
    locales \
    xvfb \
    python3-pip \
    curl \
    build-essential \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglu1-mesa \
    mesa-utils

# Install Rust (required for building tiktoken)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Cleanup disabled for development builds
# RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# Upgrade pip first, then install packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --break-system-packages python-dotenv
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "alias start='python3 /app/src/turtle_agent/scripts/turtle_agent.py'" >> /root/.bashrc

COPY . /app/
WORKDIR /app/

# Modify the RUN command to use ARG
RUN /bin/bash -c 'if [ "$DEVELOPMENT" = "true" ]; then \
    python3 -m pip install --break-system-packages --ignore-installed --user -e .; \
    else \
    python3 -m pip install --break-system-packages --ignore-installed -U "jpl-rosa>=1.0.8"; \
    fi'

CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && \
    export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1} && \
    export QT_X11_NO_MITSHM=${QT_X11_NO_MITSHM:-1} && \
    export MESA_LOADER_DRIVER_OVERRIDE=${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe} && \
    if [ \"$HEADLESS\" = \"false\" ]; then \
    ros2 run turtlesim turtlesim_node & \
    else \
    xvfb-run -a -s \"-screen 0 1920x1080x24\" ros2 run turtlesim turtlesim_node & \
    fi && \
    sleep 5 && \
    echo \"Run \\`start --streaming\\` or \\`start streaming:=true\\` to launch the ROSA-TurtleSim demo.\" && \
    /bin/bash"]
