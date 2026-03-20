#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
  ALL-IN-ONE: Điều khiển MiR robot thật bằng tay cầm
  Kết nối TRỰC TIẾP đến MiR qua WebSocket (không cần mir_bridge)
  Sử dụng module rosbridge.py đã được chứng minh hoạt động.

  Cách chạy:
    roslaunch mir_driver mir_joystick_direct.launch mir_hostname:=192.168.0.177
=============================================================
"""
import sys
import json
import time
import rospy
from sensor_msgs.msg import Joy
from mir_driver.rosbridge import RosbridgeSetup
from rospy_message_converter import message_converter
import std_msgs.msg
import copy


class MirJoystickDirect:
    """Đọc tay cầm từ ROS /joy topic và gửi trực tiếp đến MiR."""

    def __init__(self, mir_hostname, mir_port=9090):
        # ===== Tham số tốc độ =====
        self.max_linear = rospy.get_param('~max_linear_speed', 0.3)    # m/s
        self.max_angular = rospy.get_param('~max_angular_speed', 0.3)   # rad/s
        self.current_linear = self.max_linear
        self.current_angular = self.max_angular

        # ===== Mapping nút tay cầm =====
        self.axis_linear = rospy.get_param('~axis_linear', 1)      # Cần trái ↑↓
        self.axis_angular = rospy.get_param('~axis_angular', 0)    # Cần trái ←→
        self.deadman_btn = rospy.get_param('~deadman_button', 5)   # R1/RB
        self.speed_up_btn = rospy.get_param('~speed_up_button', 3)   # Y/Triangle
        self.speed_down_btn = rospy.get_param('~speed_down_button', 0) # A/Cross

        # ===== Kết nối đến MiR bằng RosbridgeSetup (đã chứng minh hoạt động) =====
        rospy.loginfo("Đang kết nối đến MiR tại %s:%d ...", mir_hostname, mir_port)
        self.robot = RosbridgeSetup(mir_hostname, mir_port)

        # Chờ kết nối (timeout 15 giây)
        rate = rospy.Rate(10)
        for i in range(150):
            if rospy.is_shutdown():
                sys.exit(0)
            if self.robot.is_connected():
                break
            if self.robot.is_errored():
                rospy.logfatal("❌ Lỗi kết nối đến MiR %s:%d!", mir_hostname, mir_port)
                sys.exit(1)
            if i % 10 == 0 and i > 0:
                rospy.logwarn("Vẫn đang chờ kết nối đến MiR %s ...", mir_hostname)
            rate.sleep()

        if not self.robot.is_connected():
            rospy.logfatal("❌ Timeout - Không kết nối được đến MiR!")
            rospy.logfatal("  1. MiR đã bật và cùng mạng WiFi/LAN?")
            rospy.logfatal("  2. IP đúng chưa? (%s)", mir_hostname)
            rospy.logfatal("  3. Thử: ping %s", mir_hostname)
            sys.exit(1)

        rospy.loginfo("✅ Đã kết nối thành công đến MiR %s!", mir_hostname)

        # ===== Lắng nghe joystick =====
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        rospy.on_shutdown(self.shutdown)

        # ===== Trạng thái debug =====
        self.last_cmd_time = rospy.Time.now()

        rospy.loginfo("=" * 55)
        rospy.loginfo("  MiR Joystick Direct Controller")
        rospy.loginfo("  MiR IP:      %s:%d", mir_hostname, mir_port)
        rospy.loginfo("  Tốc độ:      linear=%.2f  angular=%.2f", self.max_linear, self.max_angular)
        rospy.loginfo("  Deadman:     Nút R1/RB (index %d)", self.deadman_btn)
        rospy.loginfo("  Tăng tốc:   Nút Y (index %d)", self.speed_up_btn)
        rospy.loginfo("  Giảm tốc:   Nút A (index %d)", self.speed_down_btn)
        rospy.loginfo("")
        rospy.loginfo("  ⚠️  GIỮ R1/RB + đẩy cần gạt trái để lái!")
        rospy.loginfo("=" * 55)

    def joy_callback(self, msg):
        """Nhận tín hiệu tay cầm và gửi lệnh vận tốc đến MiR."""

        # Kiểm tra nút tăng/giảm tốc
        if len(msg.buttons) > max(self.speed_up_btn, self.speed_down_btn):
            if msg.buttons[self.speed_up_btn]:
                self.current_linear = min(self.current_linear + 0.05, 1.0)
                self.current_angular = min(self.current_angular + 0.05, 1.0)
                rospy.loginfo("⬆️ Tốc độ TĂNG: linear=%.2f angular=%.2f",
                              self.current_linear, self.current_angular)
            elif msg.buttons[self.speed_down_btn]:
                self.current_linear = max(self.current_linear - 0.05, 0.05)
                self.current_angular = max(self.current_angular - 0.05, 0.05)
                rospy.loginfo("⬇️ Tốc độ GIẢM: linear=%.2f angular=%.2f",
                              self.current_linear, self.current_angular)

        # Deadman switch (R1/RB)
        linear_x = 0.0
        angular_z = 0.0

        if len(msg.buttons) > self.deadman_btn and msg.buttons[self.deadman_btn]:
            linear_x = msg.axes[self.axis_linear] * self.current_linear
            angular_z = msg.axes[self.axis_angular] * self.current_angular

            # Debug: in giá trị mỗi 0.5 giây
            now = rospy.Time.now()
            if (now - self.last_cmd_time).to_sec() > 0.5:
                rospy.loginfo("🚗 Gửi đến MiR: linear=%.3f angular=%.3f", linear_x, angular_z)
                self.last_cmd_time = now

        # Tạo TwistStamped message dict (MiR software >= 2.7 yêu cầu TwistStamped)
        header = std_msgs.msg.Header(frame_id='', stamp=rospy.Time.now())
        msg_dict = {
            'header': message_converter.convert_ros_message_to_dictionary(header),
            'twist': {
                'linear': {'x': linear_x, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': angular_z}
            }
        }

        # Gửi trực tiếp đến MiR qua WebSocket
        self.robot.publish('/cmd_vel', msg_dict)

    def shutdown(self):
        """Dừng robot khi tắt chương trình."""
        rospy.loginfo("🛑 Đang dừng robot...")
        header = std_msgs.msg.Header(frame_id='', stamp=rospy.Time.now())
        stop_msg = {
            'header': message_converter.convert_ros_message_to_dictionary(header),
            'twist': {
                'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            }
        }
        for _ in range(3):
            self.robot.publish('/cmd_vel', stop_msg)
            time.sleep(0.1)
        rospy.loginfo("✅ Đã dừng robot.")


def main():
    rospy.init_node('mir_joystick_direct')
    mir_ip = rospy.get_param('~mir_hostname', '192.168.12.20')
    mir_port = rospy.get_param('~mir_port', 9090)

    try:
        node = MirJoystickDirect(mir_ip, mir_port)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
