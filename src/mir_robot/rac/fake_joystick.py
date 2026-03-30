#!/usr/bin/env python3
"""
Script giả lập tín hiệu tay cầm PS5 để test GUI
"""
import rospy
from sensor_msgs.msg import Joy
import math
import time

def main():
    rospy.init_node('fake_joystick', anonymous=True)
    pub = rospy.Publisher('/joy', Joy, queue_size=10)
    
    rospy.loginfo("🎮 Bắt đầu giả lập tín hiệu tay cầm PS5...")
    rospy.loginfo("➤ Chạy GUI trong terminal khác: python3 guijoystick.py")
    
    rate = rospy.Rate(30)  # 30Hz
    t = 0
    
    while not rospy.is_shutdown():
        joy_msg = Joy()
        
        # Giả lập 8 axes (stick + trigger + dpad)
        joy_msg.axes = [
            0.8 * math.sin(t * 0.5),      # Left Stick X
            0.6 * math.cos(t * 0.3),      # Left Stick Y  
            1.0,                          # L2 trigger (released = +1)
            0.4 * math.sin(t * 0.7),      # Right Stick X
            0.5 * math.cos(t * 0.4),      # Right Stick Y
            1.0,                          # R2 trigger (released = +1)
            0.0,                          # D-pad X
            0.0                           # D-pad Y
        ]
        
        # Giả lập 14 buttons (0 = nhả, 1 = bấm)
        joy_msg.buttons = [
            int(t % 4 < 1),               # Cross (button 0)
            int((t+1) % 4 < 1),           # Circle (button 1) 
            int((t+2) % 4 < 1),           # Triangle (button 2)
            int((t+3) % 4 < 1),           # Square (button 3)
            int(t % 6 < 2),               # L1 (button 4)
            int(t % 8 < 3),               # R1 (button 5) - Deadman switch
            0, 0,                         # L2, R2 digital
            int(t % 10 < 1),              # Create (button 8)
            int(t % 12 < 1),              # Options (button 9)
            int(t % 20 < 1),              # PS (button 10)
            0,                            # L3 (button 11)
            0,                            # R3 (button 12)
            int(t % 15 < 1),              # Touchpad (button 13)
        ]
        
        joy_msg.header.stamp = rospy.Time.now()
        pub.publish(joy_msg)
        
        t += 0.033  # Tăng thời gian
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass