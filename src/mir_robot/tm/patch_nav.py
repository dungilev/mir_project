with open('/home/dung/mir_project/src/mir_robot/tm/testpc.py', 'r') as f:
    text = f.read()

rep1 = """            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
            print(f"🚀 [NAV] Bắt đầu di chuyển tới ({goal_w_x:.2f}, {goal_w_y:.2f})")
            rest_ok = False
            if hasattr(self, 'mir_headers') and self.mir_headers:
                rest_ok = nav.api_navigate(self.mir_headers, diem_dong, "diem_dong")
            if not rest_ok and self.robot:
                nav.ws_send_goal(self.robot, diem_dong)
            return"""

text = text.replace("""            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
            print(f"🚀 [NAV] Bắt đầu di chuyển tới ({goal_w_x:.2f}, {goal_w_y:.2f})")
            if self.robot:
                nav.ws_send_goal(self.robot, diem_dong)
            return""", rep1)


rep2 = """        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        print(f"🚀 [NAV] Bắt đầu di chuyển tới ({goal_w_x:.2f}, {goal_w_y:.2f})")
        rest_ok = False
        if hasattr(self, 'mir_headers') and self.mir_headers:
            rest_ok = nav.api_navigate(self.mir_headers, diem_dong, "diem_dong")
        if not rest_ok and self.robot:
            nav.ws_send_goal(self.robot, diem_dong)"""

text = text.replace("""        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        print(f"🚀 [NAV] Bắt đầu di chuyển tới ({goal_w_x:.2f}, {goal_w_y:.2f})")
        if self.robot:
            nav.ws_send_goal(self.robot, diem_dong)""", rep2)

with open('/home/dung/mir_project/src/mir_robot/tm/testpc.py', 'w') as f:
    f.write(text)

