#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

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

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import time
import math
import pyrealsense2 as rs
import threading

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

import navigationcacdiem as nav

from ultralytics import YOLO
import mediapipe as mp

# ================= Utils =================
def extract_3d_coordinates_from_pc(vertices, box, frame_w, frame_h):
    """Trích xuất cụm tọa độ 3D từ mảng Vertices của PointCloud nằm trong Bounding Box YOLO"""
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    
    # Lấy vùng trung tâm của người (15% theo chiều ngang, 10-60% theo chiều dọc từ trên xuống)
    roi_x1 = max(0, int(x1 + width * 0.15))
    roi_x2 = min(frame_w, int(x2 - width * 0.15))
    roi_y1 = max(0, int(y1 + height * 0.10))
    roi_y2 = min(frame_h, int(y1 + height * 0.60))
    
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None
        
    # Cắt lấy đám mây điểm khu vực này
    roi_pts = vertices[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Lọc phần tử (Z phải hợp lý, từ 0.3m tới 6.0m) - Z ở PC của Realsense là chiều sâu
    valid_mask = (roi_pts[:, :, 2] > 0.3) & (roi_pts[:, :, 2] < 6.0)
    valid_pts = roi_pts[valid_mask]
    
    if len(valid_pts) == 0:
        return None
        
    # Lấy giá trị trung vị (trung bình đám đông) để chống nhiễu
    median_pt = np.median(valid_pts, axis=0)
    
    X, Y, Z = median_pt
    
    # Trả về cả dictionary chứa thông tin mask để vẽ và số lượng điểm 3D
    return {
        'Z': float(Z),
        'X': float(-X),
        'num_points': len(valid_pts),
        'mask': valid_mask,
        'roi': (roi_x1, roi_y1, roi_x2, roi_y2)
    }

def find_3d_obstacle_in_path(vertices, target_Z, floor_y_thresh=1.6, width_m=0.35):
    """Quét dọc theo hành lang trước robot để tìm vật cản 3D lơ lửng"""
    # 1. Mask khoảng cách: Lấy điểm nằm xa hơn đầu xe MiR (Z > 1.0m) để không nhận nhầm chính robot
    mask_Z = (vertices[:, :, 2] > 1.0) & (vertices[:, :, 2] < target_Z - 0.2)
    # 2. Mask bề ngang: Hành lang rộng bằng thân robot (+- width_m)
    mask_X = (vertices[:, :, 0] > -width_m) & (vertices[:, :, 0] < width_m)
    # 3. Mask chiều cao: Bỏ qua sàn nhà (Y > 1.4) và bỏ qua trần (Y < 0.0)
    mask_Y = (vertices[:, :, 1] > 0.0) & (vertices[:, :, 1] < 1.4)
    
    valid_mask = mask_Z & mask_X & mask_Y
    valid_pts = vertices[valid_mask]
    
    if len(valid_pts) > 50: # Yêu cầu ít nhất 50 điểm để chống nhiễu
        # Tìm Z nhỏ nhất (vật cản gần nhất)
        min_idx = np.argmin(valid_pts[:, 2])
        obs_pt = valid_pts[min_idx]
        return float(-obs_pt[0]), float(obs_pt[2]) # Trả về X, Z (Camera frame)
    return None, None

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points

# ================= GUI Map =================
class MapLabel(QLabel):
    clicked_signal = pyqtSignal(float, float, object)

    def __init__(self):
        super().__init__()
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

        # Vẽ tia dò đường
        if len(self.ray_pixels) > 0:
            for p in self.ray_pixels:
                cv2.circle(display_img, p, 1, (0, 255, 255), -1) # Vàng

        # Vẽ Target (Người)
        if self.target_px:
            cv2.drawMarker(display_img, self.target_px, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

        # Vẽ vật cản (Mép bàn)
        if self.obstacle_px:
            cv2.circle(display_img, self.obstacle_px, 4, (0, 0, 255), -1) # Đỏ

        # Vẽ vật cản 3D từ Camera
        if self.obs3d_px:
            cv2.circle(display_img, self.obs3d_px, 5, (255, 0, 255), -1) # Tím
            cv2.putText(display_img, "3D OBS", (self.obs3d_px[0]+10, self.obs3d_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Vẽ viền bàn nhận diện được (Xanh dương)
        if hasattr(self, 'table_box_px') and self.table_box_px and len(self.table_box_px) == 4:
            pts = np.array(self.table_box_px, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, (255, 0, 0), 2)

        # Vẽ Goal an toàn
        if self.goal_px:
            cv2.circle(display_img, self.goal_px, 6, (0, 255, 0), -1) # Xanh lá
            cv2.putText(display_img, "SAFE GOAL", (self.goal_px[0]+10, self.goal_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            
            # Vẽ đường lùi màu vàng từ người ra Safe Goal
            if self.target_px:
                cv2.line(display_img, self.target_px, self.goal_px, (0, 255, 255), 2, cv2.LINE_AA)
            
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

# ================= Camera Thread =================
    def mouseReleaseEvent(self, event):
        if self.map_info is None: return
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        px, py = event.x(), event.y()
        wx = ox + px * res
        wy = oy + (h - py - 1) * res
        
        # Bắn ra toạ độ y như việc camera đã quét thấy người
        self.clicked_signal.emit(wx, wy, None)
        
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
            depth_profile = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
            
            # Khởi tạo đối tượng xử lý PointCloud
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

            # ---> PointCloud Processing <---
            # Khớp texture màu vào điểm ảnh 3D
            self.pc.map_to(color_frame)
            points = self.pc.calculate(depth_frame)
            # Trích xuất dạng NumPy shape (480, 640, 3) 
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(480, 640, 3)

            frame = np.asanyarray(color_frame.get_data())
            results = self.model.track(frame, classes=[0], conf=0.45, iou=0.6, persist=True, tracker="bytetrack.yaml", verbose=False)
            annotated_frame = frame.copy()
            
            if self.is_scanning:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
                
                people = []
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    for box, tid in zip(boxes, track_ids): people.append({"id": tid, "box": box})

                hand_detected_box = None
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[0]
                        h, w, _ = frame.shape
                        hx, hy = int(wrist.x * w), int(wrist.y * h)
                        
                        for person in people:
                            x1, y1, x2, y2 = person["box"]
                            if (x1 - 30) <= hx <= (x2 + 30) and hy < y1 + (y2 - y1) * 0.25:
                                hand_detected_box = person["box"]
                                break
                        if hand_detected_box is not None: break

                if hand_detected_box is not None:
                    self.is_scanning = False
                    
                    # Gọi hàm PointCloud 3D thay vì Depth 2D
                    pc_data = extract_3d_coordinates_from_pc(vertices, hand_detected_box, 640, 480)
                    
                    if pc_data is not None:
                        rel_Z = pc_data['Z']
                        rel_X = pc_data['X']
                        num_pts = pc_data['num_points']
                        mask = pc_data['mask']
                        rx1, ry1, rx2, ry2 = pc_data['roi']

                        print(f"=========================================")
                        print(f"[POINTCLOUD] Xác nhận xử lý từ Mây Điểm 3D (PointCloud)!")
                        print(f"[POINTCLOUD] Đã tính toán gộp từ {num_pts} điểm ảnh 3D hơp lệ.")
                        print(f"[POINTCLOUD] Kết quả tọa độ: Trục Z (Tiến)={rel_Z:.2f}m, Trục X={rel_X:.2f}m")
                        print(f"=========================================")

                        # === XUẤT LƯỚI POINTCLOUD RA FILE 3D (.ply) ===
                        try:
                            # Sẽ xuất nguyên mây điểm và áp map màu rgb
                            ply_path = "/home/dung/mir_project/pointcloud_mesh.ply"
                            # SỬA LỖI: dùng object 'points' thay vì 'self.pc'
                            points.export_to_ply(ply_path, color_frame)
                            print(f"[POINTCLOUD] Đã xuất thành công lưới 3D (Mesh) ra file: {ply_path}")
                        except Exception as e:
                            print(f"[POINTCLOUD_WARN] Không thể export ply: {e}")

                        # === HIGHLIGHT MÂY ĐIỂM NGAY TRÊN 2D ===
                        # Tô màu xanh lá (Tint Green) cho chính xác điểm ảnh thuộc về 3D PC hợp lệ
                        roi_color = annotated_frame[ry1:ry2, rx1:rx2].copy()
                        roi_color[mask] = roi_color[mask] * 0.4 + np.array([0, 255, 0], dtype=np.uint8) * 0.6
                        annotated_frame[ry1:ry2, rx1:rx2] = roi_color

                        msg = PointStamped()
                        msg.header.frame_id = "base_link"
                        # Trừ đi 0.475m bù trừ tâm quay robot
                        msg.point.x, msg.point.y = rel_Z - 0.475, rel_X
                        
                        # --- THÊM: Quét vật cản 3D lơ lửng ---
                        obs_rel_X, obs_rel_Z = find_3d_obstacle_in_path(vertices, rel_Z, floor_y_thresh=1.6)
                        
                        try:
                            self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(0.05))
                            pt = self.tf_listener.transformPoint("/map", msg)
                            
                            obs_pt_map = None
                            if obs_rel_Z is not None:
                                print(f"[POINTCLOUD] CẢNH BÁO: Thấy vật cản 3D lơ lửng ở Z={obs_rel_Z:.2f}m")
                                obs_msg = PointStamped()
                                obs_msg.header.frame_id = "base_link"
                                obs_msg.point.x, obs_msg.point.y = obs_rel_Z - 0.475, obs_rel_X
                                obs_pt_tf = self.tf_listener.transformPoint("/map", obs_msg)
                                obs_pt_map = (obs_pt_tf.point.x, obs_pt_tf.point.y)

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
        self.setWindowTitle("TEST RAYCAST (Camera + LiDAR)")
        self.resize(1000, 500)
        
        self.central_widget = QWidget()
        self.layout = QHBoxLayout(self.central_widget)
        
        # Left Panel (Camera + Buttons)
        self.left_panel = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.left_panel.addWidget(self.camera_label, 1)
        
        self.btn_scan = QPushButton("BẮT ĐẦU SCAN NGƯỜI")
        self.btn_scan.setMinimumHeight(50)
        self.btn_scan.clicked.connect(self.start_scanning)
        self.left_panel.addWidget(self.btn_scan)
        
        self.layout.addLayout(self.left_panel, 1)
        
        # Right Panel (Map)
        self.map_label = MapLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.map_label, 1)
        
        self.setCentralWidget(self.central_widget)

        rospy.init_node('test_pc_raycast', anonymous=True)

        self.robot = nav.ws_connect()
        self.mir_headers = nav.api_login()

        self.map_signal.connect(self.map_label.set_map)
        self.pose_signal.connect(self.map_label.set_robot_pose)
        
        # Nhận diện Click mô phỏng
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

    # ---------------- THUẬT TOÁN LAI HYBRID (Pointcloud + Lidar 2D + testlui) ----------------
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
        py_r = h - self.map_label.robot_px[1] - 1 # Chuyển ngược lại tọa độ numpy
        
        if not (0 <= px_t < w and 0 <= py_t < h):
            rospy.logwarn("[GEOM] Điểm vượt giới hạn map")
            return

        # 1. TẠO LƯỚI TỔNG HỢP GLOBAL (Combined Scan Grid)
        obs_mask = np.where((self.map_label.map_data != 0) & (self.map_label.map_data != -1), 255, 0).astype(np.uint8)
        
        combined_obs = obs_mask.copy()
        
        # Thêm 3D Pointcloud vào Lưới
        obs3d_px_tup = None
        if obs_pt_map:
            obs_px_x = int((obs_pt_map[0] - ox) / res)
            obs_px_y = int((obs_pt_map[1] - oy) / res)
            obs3d_px_tup = (obs_px_x, obs_px_y)
            self.map_label.obs3d_px = (obs_px_x, h - obs_px_y - 1)
            
            if 0 <= obs_px_x < w and 0 <= obs_px_y < h:
                radius_px = int(0.15 / res) # Vật cản 3D mở rộng 15cm
                cv2.circle(combined_obs, (obs_px_x, obs_px_y), radius_px, 255, -1)
        else:
            self.map_label.obs3d_px = None

        # Mask vùng phình to của vật cản để kiểm tra an toàn (bán kính xe ~40cm)
        safe_radius_px = int(0.4 / res)
        kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_radius_px*2+1, safe_radius_px*2+1))
        inflated_obs = cv2.dilate(combined_obs, kernel_safe)

        # 2. TÌM KIẾM BÀN XUNG QUANH AI TRÊN BẢN ĐỒ NORMAL (theo cục bộ testlui)
        win_m = 6.0
        win_px = int(win_m / res)
        half_win = win_px // 2
        
        x1 = max(0, px_t - half_win)
        x2 = min(w, px_t + half_win)
        y1 = max(0, py_t - half_win)
        y2 = min(h, py_t + half_win)
        
        local_mask = obs_mask[y1:y2, x1:x2].copy()
        contours, _ = cv2.findContours(local_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        global_contours = [cnt + np.array([[[x1, y1]]]) for cnt in contours]
            
        best_contour = None
        min_dist = float('inf')
        pt = (px_t, py_t)
        
        for cnt in global_contours:
            if cv2.contourArea(cnt) < 2 and len(cnt) < 5:
                continue
            
            dist = cv2.pointPolygonTest(cnt, pt, True)
            if dist >= 0:
                best_contour = cnt
                break
            else:
                abs_dist = abs(dist)
                if abs_dist < min_dist:
                    min_dist = abs_dist
                    best_contour = cnt
                
        # 3. TÍNH TOÁN ĐIỂM "SAFE GOAL" THEO TESTLUI.PY
        goal_px_x, goal_px_y = px_t, py_t
        goal_yaw = 0.0

        if best_contour is None:
            print("[TEST] Trống rỗng, fallback lùi đều.")
            goal_yaw = math.atan2(py_t - py_r, px_t - px_r)
            safe_dist_px = int(0.7 / res) # Lùi 0.7m
            goal_px_x = px_t - safe_dist_px * math.cos(goal_yaw)
            goal_px_y = py_t - safe_dist_px * math.sin(goal_yaw)
        else:
            rect = cv2.minAreaRect(best_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            
            # Lưu lại viền bàn để vẽ trên GUI
            self.map_label.table_box_px = [(int(b[0]), h-int(b[1])-1) for b in box]
            
            edges = []
            for i in range(4):
                p1 = box[i]
                p2 = box[(i+1)%4]
                length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                center = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)
                edges.append({'p1': p1, 'p2': p2, 'len': length, 'center': center})
                
            edges.sort(key=lambda e: e['len'], reverse=True)
            long_edges = edges[0:2]
            short_edges = edges[2:4]
            
            best_long_edge = min(long_edges, key=lambda e: math.hypot(e['center'][0] - px_t, e['center'][1] - py_t))
                    
            p1 = np.array(best_long_edge['p1'], dtype=float)
            p2 = np.array(best_long_edge['p2'], dtype=float)
            
            vec_edge = p2 - p1
            length_edge = best_long_edge['len']
            vec_edge_unit = vec_edge / length_edge if length_edge > 0 else np.array([1, 0])
            
            vec_pt = np.array([px_t - p1[0], py_t - p1[1]], dtype=float)
            proj_length = np.dot(vec_pt, vec_edge_unit)
            
            t = proj_length / length_edge if length_edge > 0 else 0.5
            t = max(0.0, min(1.0, t))
            
            proj_pt = p1 + vec_edge_unit * proj_length
            
            rect_center = np.array(rect[0])
            vec_center_to_edge = np.array(best_long_edge['center']) - rect_center
            normal_long = np.array([-vec_edge_unit[1], vec_edge_unit[0]]) # Vuông góc hướng ra
            if np.dot(vec_center_to_edge, normal_long) < 0:
                normal_long = -normal_long
                
            if t < 0.2 or t > 0.8:
                # Kịch bản A (Đầu Bàn)
                best_short_edge = min(short_edges, key=lambda e: math.hypot(e['center'][0] - proj_pt[0], e['center'][1] - proj_pt[1]))
                safe_dist_px = int(0.7 / res) # Lùi 0.7m
                
                sp1 = np.array(best_short_edge['p1'], dtype=float)
                sp2 = np.array(best_short_edge['p2'], dtype=float)
                vec_se = sp2 - sp1
                vec_se_unit = vec_se / best_short_edge['len'] if best_short_edge['len'] > 0 else np.array([1,0])
                normal_short = np.array([-vec_se_unit[1], vec_se_unit[0]])
                if np.dot(np.array(best_short_edge['center']) - rect_center, normal_short) < 0:
                    normal_short = -normal_short
                    
                ideal_px_x = best_short_edge['center'][0] + normal_short[0] * safe_dist_px
                ideal_px_y = best_short_edge['center'][1] + normal_short[1] * safe_dist_px
                
                dir_yaw = np.array([proj_pt[0] - ideal_px_x, proj_pt[1] - ideal_px_y])
                goal_yaw = math.atan2(dir_yaw[1], dir_yaw[0])
                goal_px_x, goal_px_y = ideal_px_x, ideal_px_y
                
            else:
                # Kịch bản B (Giữa bàn)
                offset_px = int(0.7 / res)
                goal_p2 = proj_pt + normal_long * offset_px + vec_edge_unit * offset_px
                goal_p1 = proj_pt + normal_long * offset_px - vec_edge_unit * offset_px
                
                preferred_goal = goal_p1 if t < 0.5 else goal_p2
                fallback_goal = goal_p2 if t < 0.5 else goal_p1
                    
                def check_safe(g):
                    gx, gy = int(g[0]), int(g[1])
                    if 0 <= gx < w and 0 <= gy < h:
                        return inflated_obs[gy, gx] == 0
                    return False
                    
                if check_safe(preferred_goal):
                    chosen = preferred_goal
                elif check_safe(fallback_goal):
                    chosen = fallback_goal
                else:
                    chosen = proj_pt + normal_long * offset_px
                    
                goal_px_x, goal_px_y = chosen[0], chosen[1]
                dir_yaw = np.array([proj_pt[0] - goal_px_x, proj_pt[1] - goal_px_y])
                goal_yaw = math.atan2(dir_yaw[1], dir_yaw[0])

        print(f"[TEST] 1. Safe goal lý tưởng tính toán được: ({ox + goal_px_x*res:.2f}, {oy + goal_px_y*res:.2f})")

        # 4. KIỂM TRA LƯỚI TỔNG HỢP (POINTCLOUD + LIDAR): TÌM ĐIỂM AN TOÀN
        final_px_x, final_px_y = int(goal_px_x), int(goal_px_y)
        
        # Kiểm tra xem điểm đích phân tích được có đang đè thẳng vào Lidar / 3D OBS hay không
        if 0 <= final_px_x < w and 0 <= final_px_y < h and inflated_obs[final_px_y, final_px_x] > 0:
            print("[COMBINED GRID] ⚠️ Điểm đỗ lý tưởng bị đè lên vật cản Lidar/PointCloud. Đang đẩy lùi thoát ra xa...")
            
            # Thay vì bắn tia từ Robot đâm xuyên bàn (gây lỗi đi gần), 
            # Ta dùng đà góc lùi (goal_yaw) để đẩy lùi Safe goal tiếp tục ra xa bức tường hay vật cản!
            escape_distance_px = int(2.0 / res) # Dự kiến cho lui tối đa thêm 2 mét để tìm khoảng trống
            
            # Tính điểm xa nhất dọc theo tia lùi:
            escape_end_x = int(final_px_x - escape_distance_px * math.cos(goal_yaw))
            escape_end_y = int(final_px_y - escape_distance_px * math.sin(goal_yaw))
            
            ray_escape = bresenham_line(final_px_x, final_px_y, escape_end_x, escape_end_y)
            self.map_label.ray_pixels = [(pt[0], h - pt[1] - 1) for pt in ray_escape] # Vẽ tia màu vàng (Cyan) là tia thoát hiểm
            
            found_safe = False
            for pt in ray_escape:
                gx, gy = pt
                if 0 <= gx < w and 0 <= gy < h:
                    if inflated_obs[gy, gx] == 0:
                        final_px_x, final_px_y = gx, gy
                        print("[COMBINED GRID] Đã tìm thấy điểm đỗ an toàn sau khi đẩy lùi khỏi vùng 3D OBS!")
                        found_safe = True
                        break
            if not found_safe:
                print("[COMBINED GRID] ❌ QUÁ DÀY ĐẶC VẬT CẢN: KHÔNG THỂ TÌM THẤY KHOẢNG TRỐNG!")
        else:
            print("[COMBINED GRID] ✅ Điểm đỗ lý tưởng trọn vẹn, không có Lidar hay 3D OBS cản đường.")
            self.map_label.ray_pixels = []
            
        goal_w_x = ox + final_px_x * res
        goal_w_y = oy + final_px_y * res
        print(f"[HYBRID] 🎯 Điểm đỗ Cuối Cùng đã cập nhật an toàn: ({goal_w_x:.2f}, {goal_w_y:.2f})")

        # Cập nhật GUI
        self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
        self.map_label.goal_yaw = goal_yaw
        self.map_label.goal_px = (int(final_px_x), h - int(final_px_y) - 1)
        self.map_label.ray_pixels = [(pt[0], h - pt[1] - 1) for pt in ray_pts]
        self.map_label.update_view()
        
        # Gọi lệnh di chuyển
        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        print(f"🚀 [NAV] Bắt đầu di chuyển tới ({goal_w_x:.2f}, {goal_w_y:.2f})")
        rest_ok = False
        if hasattr(self, 'mir_headers') and self.mir_headers:
            rest_ok = nav.api_navigate(self.mir_headers, diem_dong, "diem_dong")
        if not rest_ok and self.robot:
            nav.ws_send_goal(self.robot, diem_dong)



def main():
    app = QApplication(sys.argv)
    window = TestPCApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
