import sys
import re

with open('/home/tuanminh/mir_project/src/mir_robot/tm/mainv3.py', 'r') as f:
    main_code = f.read()
    
# Extract calculate_geometry_safe_goal from mainv3
match = re.search(r'    def calculate_geometry_safe_goal\(self, target_x, target_y, original_yaw=0.0\):(.*?)\n    # ================= WORKER STATE MACHINE =================', main_code, re.DOTALL)
if match:
    geom_func = match.group(1)
else:
    print("Could not find calculate_geometry_safe_goal")
    sys.exit(1)

geom_code = """    def calculate_hybrid_safe_goal(self, target_x, target_y, obs_pt_map=None):""" + geom_func

# Now we modify geom_code to include the 3D check at the end
# find where goal_w_x is calculated at the end
end_insert = """
        goal_w_x = ox + goal_px_x * res
        goal_w_y = oy + goal_px_y * res

        # --- 3D HYBRID CHECK ---
        obs3d_px_tup = None
        if obs_pt_map:
            obs_px_x = int((obs_pt_map[0] - ox) / res)
            obs_px_y = int((obs_pt_map[1] - oy) / res)
            obs3d_px_tup = (obs_px_x, obs_px_y)
            self.map_label.obs3d_px = (obs_px_x, h - obs_px_y - 1)
        else:
            self.map_label.obs3d_px = None

        if obs3d_px_tup:
            rx = math.cos(goal_yaw)
            ry = math.sin(goal_yaw)
            
            dx = obs3d_px_tup[0] - goal_px_x
            dy = obs3d_px_tup[1] - goal_px_y
            
            d_along = dx * rx + dy * ry
            safe_dist_px = int(0.8 / res)
            
            if d_along > 0 and d_along < safe_dist_px:
                push_back_px = safe_dist_px - d_along
                goal_px_x -= push_back_px * rx
                goal_px_y -= push_back_px * ry
                print(f"[HYBRID] 3D lơ lửng quá gần! Lùi xe thêm {push_back_px * res:.2f}m")
                
                goal_w_x = ox + goal_px_x * res
                goal_w_y = oy + goal_px_y * res

        self.map_label.table_box_px = [(int(b[0]), h-int(b[1])-1) for b in box] if 'box' in locals() else None
        self.map_label.auto_target_px = (int(px_t), h - int(py_t) - 1)
        self.map_label.goal_yaw = goal_yaw
        self.map_label.goal_px = (int(goal_px_x), h - int(goal_px_y) - 1)
        self.map_label.update_view()
        
        print(f"[HYBRID] Điểm đỗ lý tưởng: ({goal_w_x:.2f}, {goal_w_y:.2f})")
        
        # Gọi lệnh di chuyển
        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        print(f"🚀 [NAV] Bắt đầu di chuyển tới ({goal_w_x:.2f}, {goal_w_y:.2f})")
        if self.robot:
            nav.ws_send_goal(self.robot, diem_dong)
"""

geom_code = re.sub(r'        goal_w_x = ox \+ goal_px_x \* res\n        goal_w_y = oy \+ goal_px_y \* res\n\n        self.map_label.table_box_px.*return goal_w_x, goal_w_y, goal_yaw', end_insert, geom_code, flags=re.DOTALL)

# Also replace the fallback return
geom_code = geom_code.replace(
    'return goal_w_x, goal_w_y, yaw',
    '''q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
            print(f"🚀 [NAV] Bắt đầu di chuyển tới ({goal_w_x:.2f}, {goal_w_y:.2f})")
            if self.robot:
                nav.ws_send_goal(self.robot, diem_dong)
            return'''
)


with open('/home/tuanminh/mir_project/src/mir_robot/tm/testpc.py', 'r') as f:
    testpc_code = f.read()

# Replace process_raycast_safe_goal logic
# We find process_raycast_safe_goal and remove it up to the end of the class
testpc_code = re.sub(r'    # ---------------- THUẬT TOÁN RAYCAST ----------------\n    def process_raycast_safe_goal\(self, target_x, target_y, obs_pt_map=None\):.*?(?=\n\nif __name__ ==)', '    # ---------------- THUẬT TOÁN LAI HYBRID ----------------\n' + geom_code, testpc_code, flags=re.DOTALL)

# Replace the connections
testpc_code = testpc_code.replace('self.map_label.clicked_signal.connect(self.process_raycast_safe_goal)', 'self.map_label.clicked_signal.connect(self.calculate_hybrid_safe_goal)')
testpc_code = testpc_code.replace('self.video_thread.target_locked_signal.connect(self.process_raycast_safe_goal)', 'self.video_thread.target_locked_signal.connect(self.calculate_hybrid_safe_goal)')


with open('/home/tuanminh/mir_project/src/mir_robot/tm/testpc.py', 'w') as f:
    f.write(testpc_code)
print("Patcher done.")
