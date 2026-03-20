#!/bin/bash
set -e

# Source ROS Noetic
source /opt/ros/noetic/setup.bash

# Build catkin workspace nếu chưa build hoặc có thay đổi
cd /root/catkin_ws
if [ ! -f devel/setup.bash ]; then
    echo "============================================="
    echo "  Building catkin workspace lần đầu..."
    echo "============================================="
    catkin_make
fi

# Source workspace
source devel/setup.bash

echo "============================================="
echo "  MiR Robot ROS1 Noetic - Sẵn sàng!"
echo "  Workspace: /root/catkin_ws"
echo "============================================="

# Nếu có argument thì chạy nó, không thì mở bash
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    exec /bin/bash
fi

