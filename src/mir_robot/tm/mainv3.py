import os
import sys

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 (Blackwell - sm_120) TRÊN ROS NOETIC (PYTHON 3.8)
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA RTX 5060 (sm_120) GPU...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_API'] = 'pyqt5'
os.environ['YOLO_OFFLINE'] = 'True'

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
import time
import math
import pyrealsense2 as rs
import requests
import queue
import threading
import json

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
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, PoseWithCovarianceStamped
from std_msgs.msg import String

import navigationcacdiem as nav
import mir_tts

try:
    from ultralytics import YOLO
except ImportError:
    print("Vui lòng cài ultralytics")
    sys.exit()

try:
    import mediapipe as mp
except ImportError:
    print("Vui lòng cài mediapipe")
    sys.exit()

MIR_IP = "192.168.0.177"
MIR_API_URL = f"http://{MIR_IP}/api/v2.0.0"
MIR_AUTH = "Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=="

def get_depth_distance_m(depth_frame, box, frame_w, frame_h):
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    roi_x1 = int(x1 + width * 0.15)
    roi_x2 = int(x2 - width * 0.15)
    roi_y1 = int(y1 + height * 0.10)
    roi_y2 = int(y1 + height * 0.60)
    
    distances = []
    step_x = max(1, (roi_x2 - roi_x1) // 10)
    step_y = max(1, (roi_y2 - roi_y1) // 10)
    for px in range(roi_x1, roi_x2 + 1, step_x):
        for py in range(roi_y1, roi_y2 + 1, step_y):
            if 0 <= px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(px, py)
                if 0.3 < d < 6.0: distances.append(d)
    if not distances: return -1.0
    return float(np.percentile(distances, 30))

def get_person_relative_position_m(box, frame_w, depth_intrinsics, distance_m):
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    if distance_m <= 0: return None
    if depth_intrinsics is None:
        hfov_rad = math.radians(69.0)
        angle = ((center_x - frame_w / 2.0) / frame_w) * hfov_rad
        x_cam = distance_m * math.tan(angle)
    else:
        x_cam = (center_x - depth_intrinsics.ppx) / depth_intrinsics.fx * distance_m
    return (distance_m, -x_cam)

# ==============================================================================
class MapLabel(QLabel):
    clicked_signal = pyqtSignal(float, float, float)
    def __init__(self):
        super().__init__()
        self.map_img = None
        self.map_info = None
        self.goal_px = None
        self.goal_yaw = 0.0
        self.auto_target_px = None
        self.table_box_px = None
        self.path_px = []
        self.robot_px = None
        self.robot_yaw = 0.0
        self.map_data = None

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
        img[data == -1] = [220, 220, 220] # Xám nhạt cho vùng Unknown
        img[data == 0] = [255, 255, 255]  # Trắng cho vùng Free
        img[data > 0] = [0, 0, 0]         # Đen cho MỌI vật cản (kể cả xác suất thấp)
        self.map_img = cv2.flip(img, 0)
        self.map_data = data
        self.update_view()

    def set_path(self, path_msg):
        if not self.map_info: return
        self.path_px = []
        res, ox, oy, h = self.map_info.resolution, self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.height
        for pose in path_msg.poses:
            wx, wy = pose.pose.position.x, pose.pose.position.y
            px = int((wx - ox) / res)
            py = h - int((wy - oy) / res) - 1
            if 0 <= px < self.map_info.width and 0 <= py < h:
                self.path_px.append((px, py))
        self.update_view()

    def update_view(self):
        if self.map_img is None: return
        display_img = self.map_img.copy()
        if len(self.path_px) > 1:
            for i in range(len(self.path_px)-1):
                cv2.line(display_img, self.path_px[i], self.path_px[i+1], (255, 0, 0), 2)
                
        if getattr(self, 'table_box_px', None):
            pts = np.array(self.table_box_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, (255, 0, 0), 2) # Xanh dương giống testlui

        if getattr(self, 'auto_target_px', None):
            tx, ty = self.auto_target_px
            cv2.drawMarker(display_img, (tx, ty), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            cv2.circle(display_img, (tx, ty), 6, (0, 0, 255), 1)

        if self.goal_px and getattr(self, 'auto_target_px', None):
            gx, gy = self.goal_px
            cv2.line(display_img, self.auto_target_px, self.goal_px, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(display_img, (gx, gy), 8, (0, 255, 0), -1)
            cv2.putText(display_img, "SAFE GOAL", (gx+10, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            
            ar_len = 35
            gui_yaw = -self.goal_yaw
            end_x = int(gx + ar_len * math.cos(gui_yaw))
            end_y = int(gy + ar_len * math.sin(gui_yaw))
            cv2.arrowedLine(display_img, (gx, gy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)

        if self.robot_px and self.map_info:
            res = self.map_info.resolution
            rl, rw = (0.89 / res) / 2, (0.58 / res) / 2
            pts = []
            for dx, dy in [(-rl, -rw), (rl, -rw), (rl, rw), (-rl, rw)]:
                rx = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
                ry = dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)
                pts.append([int(self.robot_px[0] + rx), int(self.robot_px[1] + ry)])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [pts], (0, 165, 255))
            cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)
            cv2.polylines(display_img, [pts], True, (0, 0, 0), 2)

        h, w, ch = display_img.shape
        qImg = QImage(display_img.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qImg))

# ==============================================================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    target_locked_signal = pyqtSignal(float, float, int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.is_scanning_for_hand = False 
        
        print("[INFO] Đang tải mô hình cảnh báo người YOLO11n...")
        self.model = YOLO("/home/tuanminh/mir_project/yolo11n.pt") 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

        self.locked_track_id = None
        self.target_candidate_id = None
        self.open_hand_start_time = 0

    def is_hand_open(self, hand_landmarks):
        open_fingers = sum(1 for tip, pip in [(8,6), (12,10), (16,14), (20,18)] if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x: open_fingers += 1
        return open_fingers >= 4

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
            print("[INFO] Đã KẾT NỐI RealSense!")
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

            frame = np.asanyarray(color_frame.get_data())
            results = self.model.track(frame, classes=[0], conf=0.45, iou=0.6, persist=True, tracker="bytetrack.yaml", verbose=False)
            annotated_frame = frame.copy()
            
            # Chỉ xử lý nhận diện tay khi bật cờ
            if self.is_scanning_for_hand:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
                current_time = time.time()
                
                people = []
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    for box, tid in zip(boxes, track_ids): people.append({"id": tid, "box": box})

                hand_owner_id = None
                hand_state = None
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[0]
                        h, w, _ = frame.shape
                        hx, hy = int(wrist.x * w), int(wrist.y * h)
                        if self.is_hand_open(hand_landmarks): hand_state = "open"
                        
                        for person in people:
                            x1, y1, x2, y2 = person["box"]
                            if (x1 - 30) <= hx <= (x2 + 30) and hy < y1 + (y2 - y1) * 0.25:
                                hand_owner_id = person["id"]
                                break
                        if hand_owner_id is not None: break

                if hand_owner_id is not None and self.locked_track_id is None:
                    if hand_state == "open":
                        if self.target_candidate_id != hand_owner_id:
                            self.target_candidate_id = hand_owner_id
                            self.open_hand_start_time = current_time
                        elif current_time - self.open_hand_start_time >= 2.0: # Giảm thời gian khóa xuống 2s cho nhanh
                            self.locked_track_id = hand_owner_id
                    else:
                        self.target_candidate_id = None
                elif self.locked_track_id is None:
                    self.target_candidate_id = None
                            
                for person in people:
                    tid, box = person["id"], person["box"]
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Cập nhật vẽ Bounding Box cho tất cả mọi người phát hiện được để người dùng nhìn thấy
                    color = (0, 255, 0) if self.locked_track_id == tid else (255, 0, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated_frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if self.locked_track_id == tid:
                        d_m_raw = get_depth_distance_m(depth_frame, box, 640, 480)
                        if d_m_raw > 0:
                            d_ngang_m = math.sqrt(d_m_raw**2 - 1.2**2) if d_m_raw > 1.2 else d_m_raw
                            rel = get_person_relative_position_m(box, 640, self.depth_intrinsics, d_ngang_m)
                            if rel is not None:
                                msg = PointStamped()
                                msg.header.frame_id = "base_link"
                                msg.point.x, msg.point.y = rel[0] - 0.475, rel[1]
                                try:
                                    self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(0.05))
                                    pt = self.tf_listener.transformPoint("/map", msg)
                                    self.target_locked_signal.emit(pt.point.x, pt.point.y, tid)
                                    self.locked_track_id = None # Tắt khóa ngay sau khi emit
                                    self.target_candidate_id = None
                                except Exception as e: rospy.logerr(f"Lỗi ESP: {e}")

            cv2.putText(annotated_frame, "SCANNING" if self.is_scanning_for_hand else "IDLE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            self.change_pixmap_signal.emit(annotated_frame)
            
        self.pipeline.stop()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==============================================================================
class MainApp(QMainWindow):
    map_signal = pyqtSignal(object)
    pose_signal = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiR Auto Navigation - V3 STATE MACHINE")
        self.resize(1280, 600)
        
        self.central_widget = QWidget()
        self.layout = QHBoxLayout(self.central_widget)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.camera_label, 1)
        self.map_label = MapLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.map_label, 1)
        self.setCentralWidget(self.central_widget)

        rospy.init_node('main_control_v3', anonymous=True)
        
        # Load Laptop YOLO Model
        rospy.loginfo("Đang tải YOLO Laptop (Đồ uống)...")
        abs_path = '/home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt'
        rel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best', 'best.pt')
        model_path = abs_path if os.path.exists(abs_path) else rel_path
        
        if os.path.exists(model_path):
            self.laptop_yolo = YOLO(model_path)
        else:
            self.laptop_yolo = None
            rospy.logwarn("❌ Lỗi: Không tìm thấy model laptop.")

        self.task_queue = queue.Queue()
        self.active_orders = {} 
        self.saved_locations = {} # LƯU TỌA ĐỘ VẪY TAY { "ban 1": (x, y) }
        self.current_location = "sac"
        
        self.wait_event = threading.Event()
        self.scanning_event = threading.Event()
        self.charging_cancel_event = threading.Event()
        self.target_locked_coords = None

        self.robot = nav.ws_connect()
        self.mir_headers = nav.api_login()
        if self.mir_headers:
            try:
                nav.api_ensure_ready(self.mir_headers)
                rospy.loginfo("🔓 Đã mở phanh Dashboard (State 3) thành công!")
            except Exception as e:
                rospy.logwarn(f"Lỗi khi mở phanh: {e}")

        # Kết nối Signal để tránh xung đột Thread giữa PyQt và ROS
        self.map_signal.connect(self.map_label.set_map)
        self.pose_signal.connect(self.map_label.set_robot_pose)

        # Video thread (camera + hand tracking)
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.video_thread.target_locked_signal.connect(self.on_hand_locked)
        self.video_thread.start()

        # Publishers
        self.pub_arrived = rospy.Publisher('/robot_arrived_table', String, queue_size=10)

        # Subscribers
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber('/robot_pose', Pose, self.pose_callback)
        rospy.Subscriber('/table_call_buttons', String, self.on_guest_call)
        rospy.Subscriber('/robot_orders', String, self.on_web_order)
        rospy.Subscriber('/kitchen_commands', String, self.on_kitchen_cmd)

        rospy.on_shutdown(self.on_shutdown_hook)

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()
        rospy.loginfo("✅ MainApp v3 KHỞI ĐỘNG XONG. Đang lắng nghe /table_call_buttons...")

    def on_shutdown_hook(self):
        rospy.logwarn("[!!!] ĐANG DỪNG KHẨN CẤP ROBOT VÀ XÓA LỆNH...")
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
        try:
            while not self.task_queue.empty():
                try: self.task_queue.get_nowait()
                except: pass
            if self.mir_headers:
                requests.delete(f"{nav.API_URL}/mission_queue", headers=self.mir_headers, timeout=3)
                nav.api_set_state(self.mir_headers, 4)
            rospy.loginfo("Đã xóa hàng đợi. Robot dừng an toàn.")
        except Exception as e:
            rospy.logerr(f"Lỗi khi dừng khẩn cấp: {e}")

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

    def on_guest_call(self, msg):
        try:
            data = json.loads(msg.data)
            ban = str(data.get("ban")).strip()
            if ban.isdigit(): ban = f"ban {ban}"
            rospy.loginfo(f"[Queue] Khách gọi từ {ban}, đẩy vào hàng đợi.")
            self.task_queue.put({"type": "GUEST_CALL", "target": ban})
            self.charging_cancel_event.set()
        except Exception as e: rospy.logerr(f"Lỗi ESP: {e}")

    def on_web_order(self, msg):
        try:
            data = json.loads(msg.data)
            ban = str(data.get("ban", "")).strip()
            if ban.isdigit(): ban = f"ban {ban}"
            self.active_orders[ban] = {"coca": int(data.get("coca", 0)), "lavie": int(data.get("lavie", 0))}
            self.wait_event.set() # Ngắt chờ order
        except Exception as e: rospy.logerr(f"Lỗi ESP: {e}")

    def on_kitchen_cmd(self, msg):
        try:
            data = json.loads(msg.data)
            action = data.get("action")
            if action == "call_robot":
                rospy.loginfo("[Queue] Bếp gọi robot.")
                self.task_queue.put({"type": "KITCHEN_CALL", "target": "bep"})
                self.charging_cancel_event.set()
            elif action == "deliver":
                ban = str(data.get("table", "")).strip()
                if ban.isdigit(): ban = f"ban {ban}"
                if self.current_location != "bep":
                    self.task_queue.put({"type": "KITCHEN_CALL", "target": "bep"})
                self.task_queue.put({"type": "DELIVER", "target": ban})
                self.charging_cancel_event.set()
        except Exception as e: rospy.logerr(f"Lỗi ESP: {e}")

    def on_hand_locked(self, mx, my, tid):
        self.target_locked_coords = (mx, my)
        self.scanning_event.set()

    # ================= GEOMETRY LOGIC =================
    def calculate_geometry_safe_goal(self, target_x, target_y, original_yaw=0.0):
        if not self.map_label.map_info or self.map_label.robot_px is None:
            return target_x, target_y, original_yaw
            
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
            return target_x, target_y, original_yaw

        obs_mask = np.where((self.map_label.map_data != 0) & (self.map_label.map_data != -1), 255, 0).astype(np.uint8)
        
        win_m = 6.0
        win_px = int(win_m / res)
        half_win = win_px // 2
        
        x1 = max(0, px_t - half_win)
        x2 = min(w, px_t + half_win)
        y1 = max(0, py_t - half_win)
        y2 = min(h, py_t + half_win)
        
        local_mask = obs_mask[y1:y2, x1:x2].copy()
        contours, _ = cv2.findContours(local_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        global_contours = []
        for cnt in contours:
            global_cnt = cnt + np.array([[[x1, y1]]])
            global_contours.append(global_cnt)
            
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
                
        if best_contour is None:
            # Fallback y hệt mainv3 cũ (khi ko thấy bàn)
            rx, ry = target_x, target_y
            if self.map_label.robot_px:
                rx = self.map_label.robot_px[0] * res + ox
                ry = (h - 1 - self.map_label.robot_px[1]) * res + oy
            yaw = math.atan2(target_y - ry, target_x - rx)
            goal_w_x = target_x - 0.7 * math.cos(yaw)
            goal_w_y = target_y - 0.7 * math.sin(yaw)
            
            self.map_label.auto_target_px = (int((target_x - ox)/res), h - int((target_y - oy)/res) - 1)
            self.map_label.goal_px = (int((goal_w_x - ox)/res), h - int((goal_w_y - oy)/res) - 1)
            self.map_label.goal_yaw = yaw
            self.map_label.update_view()
            return goal_w_x, goal_w_y, yaw
            
        rect = cv2.minAreaRect(best_contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
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
        
        best_long_edge = None
        min_ed = float('inf')
        for edge in long_edges:
            d = math.hypot(edge['center'][0] - px_t, edge['center'][1] - py_t)
            if d < min_ed:
                min_ed = d
                best_long_edge = edge
                
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
        normal_long = np.array([-vec_edge_unit[1], vec_edge_unit[0]]) # Vuông góc
        if np.dot(vec_center_to_edge, normal_long) < 0:
            normal_long = -normal_long # Đảm bảo hướng ra ngoài
            
        goal_px_x, goal_px_y = None, None
        goal_yaw = 0.0
        
        if t < 0.2 or t > 0.8:
            best_short_edge = None
            min_sd = float('inf')
            for edge in short_edges:
                d = math.hypot(edge['center'][0] - proj_pt[0], edge['center'][1] - proj_pt[1])
                if d < min_sd:
                    min_sd = d
                    best_short_edge = edge
                    
            safe_dist_px = int(0.7 / res) # Lùi 0.7m
            
            sp1 = np.array(best_short_edge['p1'], dtype=float)
            sp2 = np.array(best_short_edge['p2'], dtype=float)
            vec_se = sp2 - sp1
            vec_se_unit = vec_se / best_short_edge['len'] if best_short_edge['len'] > 0 else np.array([1,0])
            normal_short = np.array([-vec_se_unit[1], vec_se_unit[0]])
            if np.dot(np.array(best_short_edge['center']) - rect_center, normal_short) < 0:
                normal_short = -normal_short
                
            goal_px_x = best_short_edge['center'][0] + normal_short[0] * safe_dist_px
            goal_px_y = best_short_edge['center'][1] + normal_short[1] * safe_dist_px
            
            dir_yaw = np.array([proj_pt[0] - goal_px_x, proj_pt[1] - goal_px_y])
            goal_yaw = math.atan2(dir_yaw[1], dir_yaw[0])
            
        else:
            offset_px = int(0.7 / res)
            goal_p2 = proj_pt + normal_long * offset_px + vec_edge_unit * offset_px
            goal_p1 = proj_pt + normal_long * offset_px - vec_edge_unit * offset_px
            
            if t < 0.5:
                preferred_goal = goal_p1
                fallback_goal = goal_p2
            else:
                preferred_goal = goal_p2
                fallback_goal = goal_p1
                
            free_space = (self.map_label.map_data == 0).astype(np.uint8)
            safe_radius_px = int(0.4 / res)
            kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_radius_px*2+1, safe_radius_px*2+1))
            safe_mask = cv2.erode(free_space, kernel_safe)
            
            def is_safe(g):
                gx, gy = int(g[0]), int(g[1])
                if 0 <= gx < w and 0 <= gy < h:
                    return safe_mask[gy, gx] == 1
                return False
                
            if is_safe(preferred_goal):
                chosen = preferred_goal
            elif is_safe(fallback_goal):
                chosen = fallback_goal
            else:
                chosen = proj_pt + normal_long * offset_px
                
            goal_px_x = chosen[0]
            goal_px_y = chosen[1]
            
            dir_yaw = np.array([proj_pt[0] - goal_px_x, proj_pt[1] - goal_px_y])
            goal_yaw = math.atan2(dir_yaw[1], dir_yaw[0])

        goal_w_x = ox + goal_px_x * res
        goal_w_y = oy + goal_px_y * res

        self.map_label.table_box_px = [(int(b[0]), h-int(b[1])-1) for b in box]
        self.map_label.auto_target_px = (int(px_t), h - int(py_t) - 1)
        self.map_label.goal_yaw = goal_yaw
        self.map_label.goal_px = (int(goal_px_x), h - int(goal_px_y) - 1)
        self.map_label.update_view()
        
        return goal_w_x, goal_w_y, goal_yaw

    # ================= WORKER STATE MACHINE =================
    def worker_loop(self):
        while not rospy.is_shutdown():
            try:
                task = self.task_queue.get(timeout=2.0)
                try:
                    if task["type"] == "RETURN_HOME":
                        self.charging_cancel_event.clear()
                        self.execute_task(task, cancel_event=self.charging_cancel_event)
                    else:
                        self.execute_task(task)
                except Exception as e:
                    rospy.logerr(f"Lỗi khi thực thi task: {e}")
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                if self.current_location not in ["sac", "moving_to_sac", "bep"]:
                    rospy.loginfo("Rảnh rỗi -> Về sạc")
                    self.current_location = "moving_to_sac"
                    self.task_queue.put({"type": "RETURN_HOME", "target": "sac"})

    def move_to_pose(self, x, y, yaw):
        rospy.loginfo(f"🚀 Bắt đầu di chuyển tới toạ độ động ({x:.2f}, {y:.2f})")
            
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        diem_dong = {"x": x, "y": y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        # SỬ DỤNG REST API ĐỂ CÓ THỂ QUAY ĐÚNG GÓC YAW THAY VÌ WS_SEND_GOAL GÂY TRƯỢT GÓC
        rest_ok = False
        if self.mir_headers:
            rest_ok = nav.api_navigate(self.mir_headers, diem_dong, "diem_dong")
            
        if not rest_ok:
            rospy.logwarn("REST API thất bại, thử dùng WebSocket dự phòng...")
            if self.robot:
                nav.ws_send_goal(self.robot, diem_dong)
        
        result = nav.wait_arrival(self.robot, diem_dong, self.mir_headers, timeout=60, rest_mode=rest_ok)
        if result:
            rospy.loginfo(f"✅ Đã tới toạ độ động với góc yaw chuẩn xác")
            return True
        else:
            rospy.logwarn(f"❌ Di chuyển toạ độ động thất bại hoặc bị kẹt!")
            return False

    def move_to_static_goal(self, target_name, cancel_event=None):
        if target_name == self.current_location:
            rospy.loginfo(f"Khỏi cần đi, Robot đang ở sẵn '{target_name}'.")
            return True
        if target_name not in nav.DIEM: 
            rospy.logwarn(f"⚠️ Điểm đến '{target_name}' không tồn tại trong map!")
            return False
            
        rospy.loginfo(f"🚀 Bắt đầu di chuyển tới điểm tĩnh: {target_name}")
        nav.handle_command(target_name, self.robot, self.mir_headers, non_interactive=True, cancel_event=cancel_event)
        
        if cancel_event and cancel_event.is_set():
            rospy.loginfo(f"⚠️ Hủy di chuyển đến '{target_name}' do có lệnh mới.")
            return False
            
        self.current_location = target_name
        return True

    def verify_tray(self, exp_coca, exp_lavie, check_empty=False):
        if not self.laptop_yolo: return True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            rospy.logwarn("⚠️ Không mở được Camera kiểm đồ! Bỏ qua AI check.")
            return True
        start = rospy.Time.now()
        success = 0
        while (rospy.Time.now() - start).to_sec() < 30.0:
            ret, frame = cap.read()
            if not ret: break
            res = self.laptop_yolo.track(frame, persist=True, stream=True, conf=0.40, verbose=False)
            coca, lavie = 0, 0
            for r in res:
                if r.boxes:
                    for b in r.boxes:
                        if int(b.cls[0]) == 0: coca += 1
                        else: lavie += 1
            
            if check_empty:
                if coca == 0 and lavie == 0: success += 1
                else: success = 0
            else:
                ec, el = max(0, exp_coca), max(0, exp_lavie)
                if ec==0 and el==0: el=1
                if coca >= ec and lavie >= el: success += 1
                else: success = 0
                
            if success >= 5:
                cap.release()
                return True
        cap.release()
        return False

    def execute_task(self, task, cancel_event=None):
        ttype, target = task["type"], task["target"]
        
        if ttype == "GUEST_CALL":
            ok = self.move_to_static_goal(target, cancel_event=cancel_event)
            if not ok: return
            
            mir_tts.speak_on_mir("Chào quý khách, khách nào order thì giơ tay lên.")
            
            self.target_locked_coords = None
            self.scanning_event.clear()
            self.video_thread.is_scanning_for_hand = True
            
            if self.scanning_event.wait(timeout=20.0):
                self.video_thread.is_scanning_for_hand = False
                tx, ty = self.target_locked_coords
                self.saved_locations[target] = (tx, ty)
                
                safe_x, safe_y, safe_yaw = self.calculate_geometry_safe_goal(tx, ty, 0.0)
                self.move_to_pose(safe_x, safe_y, safe_yaw)
                self.current_location = "specific_" + target
                
                mir_tts.speak_on_mir("Mời khách order.")
                self.wait_event.clear()
                self.pub_arrived.publish(json.dumps({"action": "popup_menu", "ban": target}))
                
                start_wait = time.time()
                ordered = False
                while time.time() - start_wait < 45.0:
                    if target in self.active_orders:
                        ordered = True
                        break
                    self.wait_event.wait(timeout=1.0)
                    self.wait_event.clear()
                
                if ordered:
                    mir_tts.speak_on_mir("Đã nhận order, vui lòng đợi món.")
                else:
                    mir_tts.speak_on_mir("Hết thời gian order, robot xin phép rời đi.")
            else:
                self.video_thread.is_scanning_for_hand = False
                mir_tts.speak_on_mir("Không thấy khách vẫy tay, robot xin phép rời đi.")

        elif ttype == "KITCHEN_CALL":
            ok = self.move_to_static_goal(target, cancel_event=cancel_event)
            if not ok: return
            mir_tts.speak_on_mir("Đã tới bếp, yêu cầu đặt món lên khay.")
            rospy.sleep(10.0) # Đợi bếp đặt đồ

        elif ttype == "DELIVER":
            if self.current_location != "bep":
                ok = self.move_to_static_goal("bep", cancel_event=cancel_event)
                if not ok: return
            mir_tts.speak_on_mir("Đã tới bếp, yêu cầu để đồ ăn lên.")
            
            order = self.active_orders.get(target, {"coca":0, "lavie":0})
            if not self.verify_tray(order["coca"], order["lavie"], check_empty=False):
                mir_tts.speak_on_mir("Đồ ăn chưa đủ, xin thử lại sau.")
                return
                
            mir_tts.speak_on_mir(f"Đang giao món tới {target}.")
            
            if target in self.saved_locations:
                tx, ty = self.saved_locations[target]
                safe_x, safe_y, safe_yaw = self.calculate_geometry_safe_goal(tx, ty, 0.0)
                ok = self.move_to_pose(safe_x, safe_y, safe_yaw)
                if ok: self.current_location = "specific_" + target
                else: return
            else:
                ok = self.move_to_static_goal(target, cancel_event=cancel_event)
                if not ok: return
                self.current_location = target
                
            mir_tts.speak_on_mir("Đồ ăn của quý khách đã tới, yêu cầu quý khách lấy đồ ăn.")
            
            if self.verify_tray(0, 0, check_empty=True):
                mir_tts.speak_on_mir("Cảm ơn quý khách, chúc quý khách bữa ăn ngon miệng.")
            else:
                mir_tts.speak_on_mir("Khách chưa lấy hết đồ, robot xin phép rời đi.")
                
            if target in self.saved_locations: del self.saved_locations[target]
            if target in self.active_orders: del self.active_orders[target]

        elif ttype == "RETURN_HOME":
            self.move_to_static_goal(target, cancel_event=cancel_event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
