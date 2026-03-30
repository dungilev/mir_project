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

ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=20

# Cài pip mới + wheel tools để giảm lỗi resolver/download khi build
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Cài torch CUDA để Ultralytics/YOLO chạy GPU trong container
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 \
    torchvision==0.19.1

# Base image co the da co psutil (distutils), can cai de bang ignore-installed
# de tranh loi "Cannot uninstall psutil ... distutils installed project".
RUN python3 -m pip install --no-cache-dir --ignore-installed psutil==7.2.2

RUN python3 -m pip install --no-cache-dir \
    numpy \
    sherpa-onnx \
    sounddevice \
    pygame \
    gTTS \
    ultralytics \
    mediapipe==0.10.10 \
    pyrealsense2==2.54.1.5216 \
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