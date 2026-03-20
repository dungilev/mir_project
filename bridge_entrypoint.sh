#!/bin/bash
set -e

# Source cả ROS1 và ROS2
source /opt/ros/noetic/setup.bash
source /opt/ros/galactic/setup.bash

echo "============================================="
echo "  ROS1 Bridge - Đang chờ ROS Master..."
echo "============================================="

# Chờ ROS Master (từ container mir_ros1) sẵn sàng
until python3 - <<'PY' > /dev/null 2>&1
import os
import xmlrpc.client

uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
proxy = xmlrpc.client.ServerProxy(uri)
code, _, _ = proxy.getUri("/ros1_bridge_wait")
raise SystemExit(0 if code == 1 else 1)
PY
do
    echo "Chờ ROS Master khởi động..."
    sleep 2
done

echo "============================================="
echo "  ROS1 Bridge - Đã kết nối ROS Master!"
echo "  Đang khởi động dynamic_bridge..."
echo "============================================="

# Chạy dynamic bridge - tự phát hiện topic và chuyển đổi
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics
