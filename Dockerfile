FROM osrf/ros:noetic-desktop-full

# Cài đặt các công cụ cơ bản và thư viện cần thiết cho Navigation/MiR
RUN apt-get update && apt-get install -y \
    git \
    nano \
    alsa-utils \
    libasound2 \
    libasound2-plugins \
    pulseaudio-utils \
    libportaudio2 \
    portaudio19-dev \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-joy \
    ros-noetic-teleop-twist-joy \
    ros-noetic-teleop-twist-keyboard \
    ros-noetic-laser-geometry \
    ros-noetic-map-server \
    ros-noetic-amcl \
    ros-noetic-move-base \
    ros-noetic-dwa-local-planner \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-costmap-queue \
    ros-noetic-dwb-local-planner \
    ros-noetic-nav-core2 \
    ros-noetic-mbf-msgs \
    ros-noetic-mbf-costmap-core \
    ros-noetic-gazebo-ros-control \
    ros-noetic-hector-slam \
    ros-noetic-costmap-queue \
    ros-noetic-rospy-message-converter \
    ros-noetic-dwb-critics \
    ros-noetic-dwb-plugins \
    ros-noetic-robot-state-publisher \
    ros-noetic-rosbridge-suite \
    python3-websocket \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    numpy \
    sherpa-onnx \
    sounddevice \
    pygame \
    gTTS \
    ultralytics \
    onnxruntime-gpu==1.16.3 \
    opencv-python

# Cấu hình Workspace
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

# Source môi trường ROS tự động mỗi khi mở terminal mới
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "[ -f /root/catkin_ws/devel/setup.bash ] && source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# Copy entrypoint
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

ENTRYPOINT ["/root/entrypoint.sh"]