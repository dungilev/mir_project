#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class MirJoyTeleop:
    def __init__(self):
        rospy.init_node('mir_joy_teleop')
        
        # Đăng ký gửi lệnh xuống bánh xe
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Đăng ký nghe tín hiệu từ tay cầm
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        # Thông số tốc độ tối đa (Chỉnh nhỏ lại cho an toàn khi test lab)
        self.max_linear_speed = 0.3  # m/s
        self.max_angular_speed = 0.3 # rad/s
        
        # Mapping nút (Thường dùng cho tay Xbox/PS4)
        # Tuan có thể phải chỉnh lại số này dựa vào kết quả lệnh rostopic echo /joy
        self.axis_linear = 1   # Cần gạt trái (Lên/Xuống)
        self.axis_angular = 0  # Cần gạt trái (Trái/Phải)
        self.deadman_button = 5 # Nút R1 hoặc RB (Phải giữ nút này mới chạy được)

        rospy.loginfo("🎮 Bắt đầu điều khiển MiR bằng tay cầm!")
        rospy.loginfo("⚠️ NHỚ GIỮ NÚT R1/RB TRƯỚC KHI ĐẨY CẦN GẠT!")

    def joy_callback(self, joy_msg):
        twist = Twist()

        # Kiểm tra Nút An Toàn (Deadman Switch)
        if joy_msg.buttons[self.deadman_button] == 1:
            # Lấy giá trị từ trục cần gạt (-1.0 đến 1.0) nhân với tốc độ max
            twist.linear.x = joy_msg.axes[self.axis_linear] * self.max_linear_speed
            # Đảo dấu trục xoay cho thuận tay
            twist.angular.z = joy_msg.axes[self.axis_angular] * self.max_angular_speed
        else:
            # Nếu nhả nút an toàn -> Phanh gấp (gửi vận tốc 0)
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        # Gửi lệnh xuống robot
        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        MirJoyTeleop()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass