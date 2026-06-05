#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import math
import time
import threading

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 TRÊN ROS NOETIC (PYTHON 3.8)
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA GPU...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_API'] = 'pyqt5'
os.environ['YOLO_OFFLINE'] = 'True'

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyrealsense2 as rs

_ros_sys_path = '/opt/ros/noetic/lib/python3/dist-packages'
if os.path.isdir(_ros_sys_path) and _ros_sys_path not in sys.path:
    sys.path.insert(1, _ros_sys_path)
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in ['geometry_msgs', 'nav_msgs', 'sensor_msgs',
                                                  'std_msgs', 'actionlib_msgs', 'tf2_msgs', 'move_base_msgs']):
            del sys.modules[mod_name]

import rospy
import tf
import tf.transformations
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped, Pose

# Thay đổi bằng file import nav của bạn
import navigationcacdiem as nav
from ultralytics import YOLO
import mediapipe as mp

# ================= Utils =================
def extract_3d_coordinates_from_pc(vertices, box, frame_w, frame_h):
    """Trích xuất cụm tọa độ 3D từ mảng Vertices của PointCloud nằm trong Bounding Box YOLO"""
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    
    # Cắt lấy nửa trên của bounding box để CHẮC CHẮN né cái bàn phía trước
    roi_x1 = max(0, int(x1 + width * 0.20))
    roi_x2 = min(frame_w, int(x2 - width * 0.20))
    roi_y1 = max(0, int(y1 + height * 0.05))
    roi_y2 = min(frame_h, int(y1 + height * 0.45))
    
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1: return None
        
    roi_pts = vertices[roi_y1:roi_y2, roi_x1:roi_x2]
    valid_mask = (roi_pts[:, :, 2] > 0.3) & (roi_pts[:, :, 2] < 6.0)
    valid_pts = roi_pts[valid_mask]
    
    if len(valid_pts) < 10: return None
        
    # GIẢI THUẬT LỌC FOREGROUND (Khách hàng):
    # Khi khách ngồi ngang, khung hình có rất nhiều khoảng trống xuyên qua tường phía sau.
    # Lấy Median toàn bộ sẽ bị kéo ra tường (làm điểm bị trôi xa).
    Z_values = valid_pts[:, 2]
    
    # Tìm khoảng cách của khối vật thể gần nhất trong ROI (chính là khách hàng)
    p15_Z = np.percentile(Z_values, 15) # Dùng phân vị 15% để bỏ qua nhiễu hạt
    
    # Lọc ra CHỈ các điểm thuộc về cơ thể người (dày khoảng 50cm từ mặt trước)
    person_mask = (Z_values >= p15_Z - 0.1) & (Z_values <= p15_Z + 0.5)
    person_pts = valid_pts[person_mask]
    
    if len(person_pts) < 5:
        person_pts = valid_pts # Fallback an toàn
        
    # Tính toán tọa độ chuẩn tâm của người
    median_pt = np.median(person_pts, axis=0)
    X, Y, Z = median_pt
    
    return {
        'Z': float(Z),
        'X_raw': float(X),
        'Y_raw': float(Y),
        'X': float(-X),
        'num_points': len(person_pts),
        'mask': valid_mask,
        'roi': (roi_x1, roi_y1, roi_x2, roi_y2)
    }

def find_3d_obstacle_in_path(vertices, target_Z, floor_y_thresh=1.6, width_m=0.35):
    """Quét dọc theo hành lang trước robot để tìm vật cản 3D lơ lửng"""
    mask_Z = (vertices[:, :, 2] > 1.0) & (vertices[:, :, 2] < target_Z - 0.2)
    mask_X = (vertices[:, :, 0] > -width_m) & (vertices[:, :, 0] < width_m)
    mask_Y = (vertices[:, :, 1] > 0.0) & (vertices[:, :, 1] < 1.4)
    
    valid_mask = mask_Z & mask_X & mask_Y
    valid_pts = vertices[valid_mask]
    
    if len(valid_pts) > 50:
        min_idx = np.argmin(valid_pts[:, 2])
        obs_pt = valid_pts[min_idx]
        return float(-obs_pt[0]), float(obs_pt[2])
    return None, None

# ================= GUI Map =================
class MapLabel(QLabel):
    clicked_signal = pyqtSignal(float, float, object)

    def __init__(self):
        super().__init__()
        self.setText("Đang chờ dữ liệu từ ROS topic /map ...")
        self.setStyleSheet("background-color: #333; color: white; font-size: 16px;")
        
        self.map_img = None
        self.map_info = None
        self.robot_px = None
        self.robot_yaw = 0.0
        self.map_data = None
        
        self.target_px = None
        self.goal_px = None
        self.ray_pixels = []
        self.obstacle_px = None
        self.obs3d_px = None
        self.table_box_px = []

    def set_robot_pose(self, wx, wy, yaw=0.0):
        if not self.map_info: return
        res, ox, oy, h = self.map_info.resolution, self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.height
        px = int((wx - ox) / res)
        py = h - int((wy - oy) / res) - 1
        if 0 <= px < self.map_info.width and 0 <= py < h:
            self.robot_px = (px, py)
            self.robot_yaw = yaw
            self.update_view()

    def set_map(self, occ_grid):
        self.map_info = occ_grid.info
        w, h = self.map_info.width, self.map_info.height
        data = np.array(occ_grid.data, dtype=np.int8).reshape((h, w))
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data == -1] = [220, 220, 220] 
        img[data == 0] = [255, 255, 255]  
        img[data > 0] = [0, 0, 0]         
        self.map_img = cv2.flip(img, 0)
        self.map_data = data
        self.update_view()

    def update_view(self):
        if self.map_img is None: return
        display_img = self.map_img.copy()

        # Vẽ Target (Khách hàng)
        if self.target_px:
            cv2.circle(display_img, self.target_px, 6, (0, 0, 255), -1) # Chấm xanh dương (RGB format)
            cv2.putText(display_img, "CUSTOMER", (self.target_px[0]+10, self.target_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Vẽ tia dò đường (Raycasting)
        # Đã ẩn theo yêu cầu của user để GUI nhìn rõ hơn
        # if hasattr(self, 'cone_pixels') and self.cone_pixels:
        #     for pt in self.cone_pixels:
        #         cv2.circle(display_img, pt, 1, (0, 255, 255), -1) 
        # if hasattr(self, 'end_pixels') and self.end_pixels:
        #     for pt in self.end_pixels:
        #         cv2.circle(display_img, pt, 1, (0, 0, 255), -1)
        # if hasattr(self, 'ray_pixels') and self.ray_pixels:
        #     for pt in self.ray_pixels:
        #         cv2.circle(display_img, pt, 1, (255, 255, 0), -1)

        # Vẽ vật cản 3D từ Camera
        if self.obs3d_px:
            cv2.circle(display_img, self.obs3d_px, 5, (255, 0, 255), -1) # Tím
            cv2.putText(display_img, "3D OBS", (self.obs3d_px[0]+10, self.obs3d_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Vẽ Goal an toàn
        if self.goal_px:
            cv2.circle(display_img, self.goal_px, 6, (0, 255, 0), -1) # Xanh lá
            cv2.putText(display_img, "SMART GOAL", (self.goal_px[0]+10, self.goal_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            
            # Vẽ đường đỗ từ Robot ra Smart Goal (Màu Vàng RGB: 255, 255, 0)
            if self.robot_px:
                cv2.line(display_img, self.robot_px, self.goal_px, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Vẽ mũi tên hướng đỗ (Yaw) của Goal
            if hasattr(self, 'goal_yaw'):
                gui_yaw = -self.goal_yaw
                ar_len = 35
                gx, gy = self.goal_px
                end_x = int(gx + ar_len * math.cos(gui_yaw))
                end_y = int(gy + ar_len * math.sin(gui_yaw))
                cv2.arrowedLine(display_img, (gx, gy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)

        # Vẽ Robot
        if self.robot_px and self.map_info:
            res = self.map_info.resolution
            rl, rw = (0.89 / res) / 2, (0.58 / res) / 2
            pts = []
            for dx, dy in [(-rl, -rw), (rl, -rw), (rl, rw), (-rl, rw)]:
                rx = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
                ry = dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)
                pts.append([int(self.robot_px[0] + rx), int(self.robot_px[1] + ry)])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(display_img, [pts], (0, 165, 255))
            cv2.polylines(display_img, [pts], True, (0, 0, 0), 2)

        h, w, ch = display_img.shape
        qImg = QImage(display_img.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qImg))

    def mouseReleaseEvent(self, event):
        if self.map_info is None: return
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        px, py = event.x(), event.y()
        wx = ox + px * res
        wy = oy + (h - py - 1) * res
        
        self.clicked_signal.emit(wx, wy, None)

# ================= Camera Thread =================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    target_locked_signal = pyqtSignal(float, float, object)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.is_scanning = False 
        self.model = YOLO("/home/dung/mir_project/yolo11n.pt") 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    def run(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.pc = rs.pointcloud()
            print("[INFO] Đã KẾT NỐI RealSense với PointCloud module!")
        except Exception as e:
            print(f"[ERROR] RealSense: {e}")
            return

        self.tf_listener = tf.TransformListener()

        while self._run_flag:
            try: frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except: continue
                
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame: continue

            # Hạn chế gọi tính toán PointCloud nếu không ở chế độ scanning để đỡ lag FPS
            frame = np.asanyarray(color_frame.get_data())
            results = self.model.track(frame, classes=[0], conf=0.45, iou=0.6, persist=True, tracker="bytetrack.yaml", verbose=False)
            annotated_frame = frame.copy()
            
            if self.is_scanning:
                self.pc.map_to(color_frame)
                points = self.pc.calculate(depth_frame)
                vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(480, 640, 3)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
                
                people = []
                if results[0].boxes:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    for box in boxes: people.append({"box": box})

                hand_detected_box = None
                if hand_results.multi_hand_landmarks:
                    best_person = None
                    min_dist = float('inf')
                    
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[0]
                        h, w, _ = frame.shape
                        hx, hy = int(wrist.x * w), int(wrist.y * h)
                        
                        for person in people:
                            x1, y1, x2, y2 = person["box"]
                            # Nới lỏng: Cổ tay trong vùng ngang của người và nằm ở nửa trên cơ thể
                            if (x1 - 30) <= hx <= (x2 + 30) and hy < y1 + (y2 - y1) * 0.50:
                                # Giải thuật chống Overlap: Tính khoảng cách từ tay đến ĐỈNH ĐẦU
                                # Ai có tay gần đỉnh đầu của mình nhất (trên ảnh 2D) thì người đó là chủ nhân
                                head_x = (x1 + x2) / 2
                                head_y = y1
                                dist = math.hypot(hx - head_x, hy - head_y)
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    best_person = person
                                    
                    if best_person is not None:
                        hand_detected_box = best_person["box"]

                if hand_detected_box is not None:
                    pc_data = extract_3d_coordinates_from_pc(vertices, hand_detected_box, 640, 480)
                    
                    if pc_data is not None:
                        self.is_scanning = False # Chỉ dừng quét khi đã lấy được tọa độ 3D thành công
                        rel_Z = pc_data['Z']
                        rel_X = pc_data['X']
                        X_raw = pc_data['X_raw']
                        Y_raw = pc_data['Y_raw']

                        # Chuyển đổi tọa độ Camera -> MiR -> Map (CÓ HIỆU CHỈNH GÓC NGHIÊNG CAMERA)
                        # Tính khoảng cách Euclidean thực tế từ camera đến mặt/vai khách hàng
                        euclid_d = math.sqrt(X_raw**2 + Y_raw**2 + rel_Z**2)
                        
                        # Độ chênh lệch chiều cao (Camera lắp ở 1.8m, ROI trích xuất vùng ngực/bụng người ngồi tầm 0.7m)
                        delta_h = 1.8 - 0.7 
                        
                        # Tính khoảng cách ngang thực tế trên mặt đất bằng định lý Pythagore
                        if euclid_d > delta_h:
                            horizontal_sq = euclid_d**2 - delta_h**2
                            d_ngang_toan_phan = math.sqrt(horizontal_sq)
                            # Trừ đi phần xê dịch trái/phải để lấy đúng khoảng cách tiến thẳng (forward)
                            if d_ngang_toan_phan**2 > rel_X**2:
                                d_forward = math.sqrt(d_ngang_toan_phan**2 - rel_X**2)
                            else:
                                d_forward = d_ngang_toan_phan
                        else:
                            d_forward = rel_Z # Fallback an toàn nếu tính toán vô lý
                            
                        msg = PointStamped()
                        msg.header.frame_id = "base_link"
                        msg.point.x, msg.point.y = d_forward - 0.475, rel_X
                        
                        obs_rel_X, obs_rel_Z = find_3d_obstacle_in_path(vertices, rel_Z, floor_y_thresh=1.6)
                        
                        try:
                            self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(0.05))
                            pt = self.tf_listener.transformPoint("/map", msg)
                            
                            obs_pt_map = None
                            if obs_rel_Z is not None:
                                obs_msg = PointStamped()
                                obs_msg.header.frame_id = "base_link"
                                obs_msg.point.x, obs_msg.point.y = obs_rel_Z - 0.475, obs_rel_X
                                obs_pt_tf = self.tf_listener.transformPoint("/map", obs_msg)
                                obs_pt_map = (obs_pt_tf.point.x, obs_pt_tf.point.y)

                            # Phát tín hiệu báo cho APP tính toán đỗ xe
                            self.target_locked_signal.emit(pt.point.x, pt.point.y, obs_pt_map)
                        except Exception as e: print(f"Lỗi TF: {e}")

                for person in people:
                    x1, y1, x2, y2 = map(int, person["box"])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            cv2.putText(annotated_frame, "SCANNING" if self.is_scanning else "IDLE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if self.is_scanning else (0,255,0), 2)
            self.change_pixmap_signal.emit(annotated_frame)
            
        self.pipeline.stop()

    def stop(self):
        self._run_flag = False
        self.wait()

# ================= Main App =================
class TestPCApp(QMainWindow):
    map_signal = pyqtSignal(object)
    pose_signal = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self.current_goal = None
        self.is_moving = False
        self.setWindowTitle("TEST SMART DOCKING (Dynamic Radius & Target Exclusion)")
        self.resize(1000, 500)
        
        self.central_widget = QWidget()
        self.layout = QHBoxLayout(self.central_widget)
        
        self.left_panel = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.left_panel.addWidget(self.camera_label, 1)
        
        self.btn_scan = QPushButton("BẮT ĐẦU SCAN NGƯỜI")
        self.btn_scan.setMinimumHeight(50)
        self.btn_scan.clicked.connect(self.start_scanning)
        self.left_panel.addWidget(self.btn_scan)
        
        self.layout.addLayout(self.left_panel, 1)
        
        self.map_label = MapLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.map_label, 1)
        
        self.setCentralWidget(self.central_widget)

        rospy.init_node('test_smart_raycast', anonymous=True)

        self.robot = nav.ws_connect()
        self.mir_headers = nav.api_login()

        self.map_signal.connect(self.map_label.set_map)
        self.pose_signal.connect(self.map_label.set_robot_pose)
        
        # Kết nối tín hiệu tính toán
        self.map_label.clicked_signal.connect(self.calculate_hybrid_safe_goal)

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.video_thread.target_locked_signal.connect(self.calculate_hybrid_safe_goal)
        self.video_thread.start()

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber('/robot_pose', Pose, self.pose_callback)

    def start_scanning(self):
        print("[TEST] Đang mở quét tay...")
        self.video_thread.is_scanning = True

    def update_camera_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qImg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.camera_label.setPixmap(QPixmap.fromImage(qImg).scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    def map_callback(self, msg):
        self.map_signal.emit(msg)

    def pose_callback(self, msg):
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        yaw = tf.transformations.euler_from_quaternion(q)[2]
        self.pose_signal.emit(msg.position.x, msg.position.y, yaw)
        
        if self.is_moving and self.current_goal:
            gx, gy = self.current_goal
            dist = math.hypot(msg.position.x - gx, msg.position.y - gy)
            if dist < 0.2: # Ngưỡng để xác định là đã tới (dưới 20cm)
                print("\n====================================")
                print("====== 🏆 TỚI ĐÍCH RỒI !!! ========")
                print("====================================\n")
                self.is_moving = False
                self.current_goal = None

    # ---------------- THUẬT TOÁN LAI HYBRID MỚI (Dynamic Radius + Target Exclusion) ----------------
    def calculate_hybrid_safe_goal(self, target_x, target_y, obs_pt_map=None):
        if not self.map_label.map_info or self.map_label.robot_px is None:
            return
            
        res = self.map_label.map_info.resolution
        ox = self.map_label.map_info.origin.position.x
        oy = self.map_label.map_info.origin.position.y
        w = self.map_label.map_info.width
        h = self.map_label.map_info.height
        
        px_t = int((target_x - ox) / res)
        py_t = int((target_y - oy) / res)
        
        px_r = self.map_label.robot_px[0]
        py_r = h - self.map_label.robot_px[1] - 1
        
        if not (0 <= px_t < w and 0 <= py_t < h):
            rospy.logwarn("[GEOM] Điểm đích vượt giới hạn bản đồ!")
            return

        # 1. TẠO LƯỚI TỔNG HỢP GLOBAL TỪ BẢN ĐỒ 2D (Giữ nguyên vật cản)
        obs_mask = np.where((self.map_label.map_data != 0) & (self.map_label.map_data != -1), 255, 0).astype(np.uint8)
        combined_obs = obs_mask.copy()
        
        # 2. THÊM VẬT CẢN 3D LƠ LỬNG (NẾU CÓ)
        if obs_pt_map:
            obs_px_x = int((obs_pt_map[0] - ox) / res)
            obs_px_y = int((obs_pt_map[1] - oy) / res)
            self.map_label.obs3d_px = (obs_px_x, h - obs_px_y - 1)
            
            if 0 <= obs_px_x < w and 0 <= obs_px_y < h:
                radius_px = int(0.15 / res) # Vật cản 3D lơ lửng bán kính 15cm
                cv2.circle(combined_obs, (obs_px_x, obs_px_y), radius_px, 255, -1)
        else:
            self.map_label.obs3d_px = None

        # 3. BƠM PHỒNG VẬT CẢN (THÂN XE 0.4m + 0.05m LỀ = 0.45m)
        safe_radius_px = int(0.45 / res)
        kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_radius_px*2+1, safe_radius_px*2+1))
        inflated_obs = cv2.dilate(combined_obs, kernel_safe)
        
        # === THUẬT TOÁN RAYCAST ĐỖ CHÉO (PLAN C) ===
        min_dist_normal = float('inf')
        theta_normal = 0.0
        max_ray_len = int(3.0 / res)
        
        self.map_label.cone_pixels = []
        
        # Bước 1: Quét 360 độ tìm Hướng Trực Diện (Pháp tuyến - Normal)
        ray_distances = []
        for angle in range(0, 360, 5):
            rad = math.radians(angle)
            dist = float('inf')
            for step in range(1, max_ray_len):
                cx = int(px_t + step * math.cos(rad))
                cy = int(py_t + step * math.sin(rad))
                if not (0 <= cx < w and 0 <= cy < h):
                    break
                self.map_label.cone_pixels.append((cx, h - cy - 1))
                if inflated_obs[cy, cx] == 0:
                    dist = step
                    break
            if dist != float('inf'):
                ray_distances.append((rad, dist))
                    
        if not ray_distances:
            print("[SMART NAV] ❌ THẤT BẠI: Kẹt hoàn toàn! Không có khoảng trống xung quanh khách hàng.")
            self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
            self.map_label.goal_px = None
            self.map_label.update_view()
            return
            
        # Tìm tâm của vùng không gian trống để có vector pháp tuyến chuẩn nhất
        min_dist_normal = min(d for r, d in ray_distances)
        valid_angles = [r for r, d in ray_distances if d <= min_dist_normal + 2] # sai số 2 pixels
        sum_x = sum(math.cos(r) for r in valid_angles)
        sum_y = sum(math.sin(r) for r in valid_angles)
        theta_normal = math.atan2(sum_y, sum_x)
            
        # Bước 2 & 3: Quét không gian để tìm bên nào thoáng hơn (Dựa vào lõi bàn)
        # Quét góc phần tư phía sau bên Trái và phía sau bên Phải để đếm mật độ bàn
        obs_left = 0
        obs_right = 0
        for step in range(1, int(1.5 / res)): # Quét xa 1.5m
            for offset_deg in range(90, 180, 5):
                # Bên Trái
                rad_l = theta_normal + math.radians(offset_deg)
                cx_l = int(px_t + step * math.cos(rad_l))
                cy_l = int(py_t + step * math.sin(rad_l))
                if 0 <= cx_l < w and 0 <= cy_l < h and combined_obs[cy_l, cx_l] > 0:
                    obs_left += 1
                
                # Bên Phải
                rad_r = theta_normal - math.radians(offset_deg)
                cx_r = int(px_t + step * math.cos(rad_r))
                cy_r = int(py_t + step * math.sin(rad_r))
                if 0 <= cx_r < w and 0 <= cy_r < h and combined_obs[cy_r, cx_r] > 0:
                    obs_right += 1
                    
        if obs_left > obs_right:
            # Bàn nằm nhiều ở bên Trái -> Bên Phải thoáng hơn -> Đỗ ra khoảng không bên Phải
            theta_dock = theta_normal - math.radians(45)
            print(f"[SMART NAV] ↪️ Không gian PHẢI thoáng hơn (L={obs_left}, R={obs_right}). Đỗ chéo ra PHẢI (góc 45 độ).")
        else:
            # Bàn nằm nhiều ở bên Phải -> Bên Trái thoáng hơn -> Đỗ ra khoảng không bên Trái
            theta_dock = theta_normal + math.radians(45)
            print(f"[SMART NAV] ↪️ Không gian TRÁI thoáng hơn (L={obs_left}, R={obs_right}). Đỗ chéo ra TRÁI (góc 45 độ).")
            
        # Bước 4: Tính toán target_dist_m động dựa trên khoảng trống (lidar map)
        free_start_step = None
        free_end_step = None
        self.map_label.ray_pixels = []
        
        for step in range(1, max_ray_len): # Quét tối đa 3.0m
            cx = int(px_t + step * math.cos(theta_dock))
            cy = int(py_t + step * math.sin(theta_dock))
            
            if not (0 <= cx < w and 0 <= cy < h):
                if free_start_step is not None and free_end_step is None:
                    free_end_step = step
                break
                
            self.map_label.ray_pixels.append((cx, h - cy - 1))
            
            if inflated_obs[cy, cx] == 0: # Không có vật cản
                if free_start_step is None:
                    free_start_step = step
            else: # Gặp vật cản khác phía sau
                if free_start_step is not None:
                    free_end_step = step
                    break
                    
        best_pose_px = None
        goal_yaw = 0.0
        
        if free_start_step is not None:
            if free_end_step is None:
                free_end_step = max_ray_len
                
            # Tính toán tâm của khoảng trống để robot đỗ cách xa cả khách và vật cản sau lưng
            target_step = int((free_start_step + free_end_step) / 2)
            target_dist_m = target_step * res
            
            # Ràng buộc khoảng cách từ 0.9m đến 1.5m
            if target_dist_m < 0.9:
                target_step = int(0.9 / res)
            elif target_dist_m > 1.5:
                target_step = int(1.5 / res)
                
            # Đảm bảo target_step không rơi vào vật cản
            if target_step >= free_end_step:
                target_step = max(free_start_step, free_end_step - 1)
            if target_step < free_start_step:
                target_step = free_start_step
                
            best_pose_px = (
                int(px_t + target_step * math.cos(theta_dock)),
                int(py_t + target_step * math.sin(theta_dock))
            )
            
            goal_yaw = math.atan2(py_t - best_pose_px[1], px_t - best_pose_px[0])
            actual_dist = math.hypot(px_t - best_pose_px[0], py_t - best_pose_px[1])
            print(f"[SMART NAV] ✅ Đã chốt điểm đỗ chéo ở vùng trống an toàn, cách khách {actual_dist * res:.2f}m")
        else:
            print("[SMART NAV] ❌ THẤT BẠI: Bị kẹt trên tia chéo, không thể tìm thấy chỗ lách vào an toàn!")
            self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
            self.map_label.goal_px = None
            self.map_label.update_view()
            return
                
        # 5. GỬI LỆNH ĐIỀU HƯỚNG
            
        final_px_x, final_px_y = best_pose_px
        goal_w_x = ox + final_px_x * res
        goal_w_y = oy + final_px_y * res
        
        # Cập nhật GUI (Chuyển ngược tọa độ Y lên GUI)
        self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
        self.map_label.goal_yaw = goal_yaw
        self.map_label.goal_px = (int(final_px_x), h - int(final_px_y) - 1)
        self.map_label.update_view()
        
        print(f"[SMART NAV] 🎯 Tọa độ đỗ cuối cùng (X,Y) = ({goal_w_x:.2f}, {goal_w_y:.2f}), Hướng Yaw = {math.degrees(goal_yaw):.1f}°")
        
        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        print(f"🚀 [NAV] Bắn lệnh tới MiR Fleet / MoveBase!")
        
        # Bắt đầu theo dõi hành trình để thông báo khi tới nơi
        self.current_goal = (goal_w_x, goal_w_y)
        self.is_moving = True
        
        rest_ok = False
        if hasattr(self, 'mir_headers') and self.mir_headers:
            rest_ok = nav.api_navigate(self.mir_headers, diem_dong, "diem_dong")
        if not rest_ok and self.robot:
            nav.ws_send_goal(self.robot, diem_dong)

    def closeEvent(self, event):
        print("[INFO] Đang đóng luồng Camera an toàn...")
        self.video_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = TestPCApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
