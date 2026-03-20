#!/usr/bin/env python3
"""
=============================================================================
HỆ THỐNG ROBOT PHỤC VỤ NƯỚC - MiR Robot
=============================================================================
Chức năng:
  1. Nhận diện giọng nói tiếng Việt (Whisper)
  2. Nhận diện & ghi nhớ khuôn mặt khách hàng (InsightFace)
  3. Ghi nhớ vị trí khách hàng trên bản đồ
  4. Di chuyển tự động về kho lấy nước (move_base)
  5. Nhận diện số chai nước đã đặt lên robot (YOLO)
  6. Mang nước về đúng vị trí khách hàng

Yêu cầu môi trường:
  - ROS 1 Noetic
  - Python 3.8+
  - GPU NVIDIA (CUDA)
  - Camera laptop/USB (OpenCV)
=============================================================================
"""

# ===========================================================================
# PHẦN 1: IMPORT THƯ VIỆN
# ===========================================================================

import os
import sys
import time
import json
import threading
import queue
import numpy as np
import cv2

# --- ROS ---
import rospy
import actionlib
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from tf.transformations import euler_from_quaternion

# --- AI / Computer Vision ---
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# --- Giọng nói ---
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile

# --- GUI ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ===========================================================================
# PHẦN 2: HẰNG SỐ CẤU HÌNH
# ===========================================================================

# --- Điểm waypoint cố định trên bản đồ (đơn vị: mét) ---
# !! QUAN TRỌNG: Bạn phải chỉnh 2 giá trị này sau khi tạo bản đồ thực tế !!
WATER_STATION = {
    "x": 2.0,   # Tọa độ X của kho lấy nước (đo từ RViz)
    "y": 1.0,   # Tọa độ Y của kho lấy nước (đo từ RViz)
    "yaw": 0.0  # Hướng robot khi đến kho (radians, 0 = nhìn về phía +X)
}

# --- Nhận diện khuôn mặt ---
FACE_MATCH_THRESHOLD = 0.4   # Cosine distance < 0.4 → cùng 1 người
                              # Giảm xuống (vd 0.3) = khắt khe hơn
                              # Tăng lên (vd 0.5) = dễ nhận hơn nhưng hay nhầm

# --- Nhận diện chai nước ---
BOTTLE_CLASS_ID = 39          # Class ID của "bottle" trong COCO dataset (dùng khi chưa train custom)
                              # Nếu đã train custom model thì đổi thành 0 (class đầu tiên)
BOTTLE_CUSTOM_MODEL = False   # True = dùng model đã train riêng trên chai nước
                              # False = dùng COCO pretrained (class 39)
BOTTLE_CONFIDENCE = 0.45      # Ngưỡng confidence tối thiểu
BOTTLE_IOU_THRESHOLD = 0.5    # Ngưỡng NMS IoU (loại bỏ box trùng nhau)
BOTTLE_CHECK_FRAMES = 10      # Đếm đủ chai trong 10 frame liên tiếp → xác nhận
BOTTLE_CHECK_INTERVAL = 0.3   # Kiểm tra mỗi 0.3 giây (nhanh hơn)

# --- Voice ---
WAKE_WORD = "robot"           # Từ khoá kích hoạt robot lắng nghe
RECORD_SECONDS = 5            # Ghi âm tối đa 5 giây mỗi lần
SAMPLE_RATE = 16000           # Sample rate chuẩn cho Whisper

# --- Camera ---
CAMERA_INDEX = 0              # 0 = webcam mặc định, thử 1,2 nếu không được
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Robot ---
NAVIGATION_TIMEOUT = 60       # Timeout 60 giây cho mỗi lần di chuyển

# --- Bàn / Vị trí khách hàng ---
# !! QUAN TRỌNG: Bạn phải đo tọa độ các bàn từ RViz rồi điền vào đây !!
# Mỗi bàn gồm: tọa độ tâm bàn (x, y) và 4 vị trí tiếp cận (cho 4 ghế)
#
# Giải thích về approach_positions (các điểm tiếp cận):
#   Mỗi bàn có 4 ghế ở 4 phía. Robot cần đứng đối diện khách để
#   camera nhìn thẳng vào mặt khách. Ví dụ bàn 1:
#
#        Ghế Bắc (robot đứng phía nam nhìn lên)
#    Ghế Tây [   BÀN   ] Ghế Đông
#        Ghế Nam (robot đứng phía bắc nhìn xuống)
#
# approach_offset: khoảng cách từ tâm bàn đến vị trí robot đứng (mét)
TABLE_APPROACH_OFFSET = 1.0  # Robot đứng cách tâm bàn 1 mét

TABLE_POSITIONS = {
    1: {
        "x": 3.0, "y": 2.0, "name": "Bàn 1",
        # 4 vị trí tiếp cận quanh bàn:
        # Robot đứng ĐỐI DIỆN ghế khách để camera nhìn vào mặt
        # (x, y, yaw) - yaw là hướng robot nhìn vào bàn
        "approach_positions": [
            {"x": 3.0, "y": 1.0, "yaw": 1.57,  "seat": "Nam"},   # Đứng phía nam, nhìn lên bắc
            {"x": 3.0, "y": 3.0, "yaw": -1.57, "seat": "Bắc"},  # Đứng phía bắc, nhìn xuống nam
            {"x": 2.0, "y": 2.0, "yaw": 0.0,   "seat": "Tây"},   # Đứng phía tây, nhìn sang đông
            {"x": 4.0, "y": 2.0, "yaw": 3.14,  "seat": "Đông"},  # Đứng phía đông, nhìn sang tây
        ]
    },
    2: {
        "x": 3.0, "y": 4.0, "name": "Bàn 2",
        "approach_positions": [
            {"x": 3.0, "y": 3.0, "yaw": 1.57,  "seat": "Nam"},
            {"x": 3.0, "y": 5.0, "yaw": -1.57, "seat": "Bắc"},
            {"x": 2.0, "y": 4.0, "yaw": 0.0,   "seat": "Tây"},
            {"x": 4.0, "y": 4.0, "yaw": 3.14,  "seat": "Đông"},
        ]
    },
    3: {
        "x": 5.0, "y": 2.0, "name": "Bàn 3",
        "approach_positions": [
            {"x": 5.0, "y": 1.0, "yaw": 1.57,  "seat": "Nam"},
            {"x": 5.0, "y": 3.0, "yaw": -1.57, "seat": "Bắc"},
            {"x": 4.0, "y": 2.0, "yaw": 0.0,   "seat": "Tây"},
            {"x": 6.0, "y": 2.0, "yaw": 3.14,  "seat": "Đông"},
        ]
    },
    4: {
        "x": 5.0, "y": 4.0, "name": "Bàn 4",
        "approach_positions": [
            {"x": 5.0, "y": 3.0, "yaw": 1.57,  "seat": "Nam"},
            {"x": 5.0, "y": 5.0, "yaw": -1.57, "seat": "Bắc"},
            {"x": 4.0, "y": 4.0, "yaw": 0.0,   "seat": "Tây"},
            {"x": 6.0, "y": 4.0, "yaw": 3.14,  "seat": "Đông"},
        ]
    },
    5: {
        "x": 7.0, "y": 2.0, "name": "Bàn 5",
        "approach_positions": [
            {"x": 7.0, "y": 1.0, "yaw": 1.57,  "seat": "Nam"},
            {"x": 7.0, "y": 3.0, "yaw": -1.57, "seat": "Bắc"},
            {"x": 6.0, "y": 2.0, "yaw": 0.0,   "seat": "Tây"},
            {"x": 8.0, "y": 2.0, "yaw": 3.14,  "seat": "Đông"},
        ]
    },
    6: {
        "x": 7.0, "y": 4.0, "name": "Bàn 6",
        "approach_positions": [
            {"x": 7.0, "y": 3.0, "yaw": 1.57,  "seat": "Nam"},
            {"x": 7.0, "y": 5.0, "yaw": -1.57, "seat": "Bắc"},
            {"x": 6.0, "y": 4.0, "yaw": 0.0,   "seat": "Tây"},
            {"x": 8.0, "y": 4.0, "yaw": 3.14,  "seat": "Đông"},
        ]
    },
}

# --- Gaze detection (nhận diện hướng nhìn) ---
GAZE_SCORE_THRESHOLD = 0.6   # Người nhìn vào camera có gaze score > 0.6 → đang nhìn robot
LOCATE_TIMEOUT = 90           # Timeout tối đa 90 giây để tìm khách quanh bàn
LOCATE_DWELL_TIME = 3         # Đứng mỗi vị trí 3 giây để quét mặt


# ===========================================================================
# PHẦN 3: STATE MACHINE - TRẠNG THÁI ROBOT
# ===========================================================================
# Robot hoạt động theo trình tự state machine (máy trạng thái):
#
#   IDLE (chờ khách gọi)
#     ↓ khách gọi "robot bàn 3 cho tôi 2 chai nước"
#   LISTENING (ghi âm + nhận diện giọng nói)
#     ↓ trích xuất số bàn + số chai
#   NAVIGATING_TO_CUSTOMER_FIRST (đi tới bàn khách)
#     ↓ đến nơi
#   LOCATING_CUSTOMER (xoay quanh bàn tìm đúng người gọi)
#     ↓ tìm thấy người nhìn vào robot
#   FACE_REGISTER (chụp mặt khách 3 góc)
#     ↓ đăng ký/nhận diện xong
#   NAVIGATING_TO_WATER (đi tới kho lấy nước)
#     ↓ đến kho
#   WAITING_FOR_BOTTLES (chờ bếp đặt chai lên robot)
#     ↓ đủ chai
#   NAVIGATING_TO_CUSTOMER (quay lại chỗ khách giao nước)
#     ↓ đến nơi
#   DELIVERING (chờ khách lấy nước)
#     ↓ xong
#   IDLE (quay lại chờ)

class RobotState:
    IDLE                       = "IDLE"                       # Chờ khách gọi
    LISTENING                  = "LISTENING"                  # Đang nghe giọng nói
    NAVIGATING_TO_CUSTOMER_FIRST = "NAVIGATING_TO_CUSTOMER_FIRST"  # Đi tới bàn khách
    LOCATING_CUSTOMER          = "LOCATING_CUSTOMER"          # Xoay quanh bàn tìm đúng người gọi
    FACE_REGISTER              = "FACE_REGISTER"              # Đang chụp khuôn mặt khách
    NAVIGATING_TO_WATER        = "NAVIGATING_TO_WATER"        # Đang đi tới kho nước
    WAITING_FOR_BOTTLES        = "WAITING_FOR_BOTTLES"        # Chờ bếp đặt chai lên robot
    NAVIGATING_TO_CUSTOMER     = "NAVIGATING_TO_CUSTOMER"     # Đang đi về chỗ khách (giao nước)
    DELIVERING                 = "DELIVERING"                 # Đến nơi, chờ khách lấy
    ERROR                      = "ERROR"                      # Lỗi


# ===========================================================================
# PHẦN 4: DATABASE KHÁCH HÀNG (lưu vào file JSON)
# ===========================================================================

class CustomerDatabase:
    """
    Lưu thông tin khách hàng vào file customers.json
    Mỗi khách có: id, tên, embedding khuôn mặt, lịch sử đặt nước
    """

    def __init__(self, db_path="/app/data/customers.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.data = self._load()

    def _load(self):
        """Đọc database từ file, nếu không có thì tạo mới"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"customers": [], "next_id": 1}

    def _save(self):
        """Ghi database xuống file"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def find_customer(self, face_embedding, face_app_threshold=FACE_MATCH_THRESHOLD):
        """
        Tìm khách hàng trong database theo khuôn mặt
        Trả về customer dict nếu tìm thấy, None nếu khách mới
        """
        best_match = None
        best_distance = float('inf')

        for customer in self.data["customers"]:
            for stored_embedding in customer["embeddings"]:
                # Cosine distance: 0 = giống nhau, 2 = khác hoàn toàn
                stored = np.array(stored_embedding)
                distance = 1 - np.dot(face_embedding, stored) / (
                    np.linalg.norm(face_embedding) * np.linalg.norm(stored) + 1e-8
                )
                if distance < best_distance:
                    best_distance = distance
                    best_match = customer

        if best_distance < face_app_threshold:
            return best_match
        return None

    def register_customer(self, embeddings, name=None):
        """
        Đăng ký khách hàng mới vào database
        embeddings: list 3 vector khuôn mặt (thẳng, trái, phải)
        """
        customer_id = self.data["next_id"]
        customer = {
            "id": customer_id,
            "name": name or f"Khách {customer_id}",
            "embeddings": [e.tolist() for e in embeddings],  # numpy → list để lưu JSON
            "order_history": []
        }
        self.data["customers"].append(customer)
        self.data["next_id"] += 1
        self._save()
        rospy.loginfo(f"[DB] Đã đăng ký khách mới: {customer['name']} (ID={customer_id})")
        return customer

    def add_order(self, customer_id, bottle_count, position):
        """Ghi lịch sử đặt nước của khách"""
        for customer in self.data["customers"]:
            if customer["id"] == customer_id:
                customer["order_history"].append({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "bottles": bottle_count,
                    "position": {"x": position[0], "y": position[1]}
                })
                self._save()
                break


# ===========================================================================
# PHẦN 5: MODULE NHẬN DIỆN GIỌNG NÓI (WHISPER)
# ===========================================================================

class VoiceRecognizer:
    """
    Nhận diện giọng nói tiếng Việt offline bằng OpenAI Whisper
    Chạy trong thread riêng để không block GUI
    """

    def __init__(self, model_size="small"):
        # model_size options: "tiny"(nhanh, kém), "small"(cân bằng), "medium"(chậm, tốt)
        # Với GPU NVIDIA, "small" chạy real-time được
        rospy.loginfo("[Voice] Đang load Whisper model...")
        self.model = whisper.load_model(model_size)
        rospy.loginfo(f"[Voice] Đã load Whisper {model_size}")

    def record_audio(self, duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
        """
        Ghi âm từ microphone trong N giây
        Trả về đường dẫn file .wav tạm thời
        """
        rospy.loginfo(f"[Voice] 🎤 Đang ghi âm {duration} giây...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,              # Mono
            dtype='int16'            # 16-bit PCM, Whisper yêu cầu
        )
        sd.wait()  # Chờ ghi xong

        # Lưu vào file tạm
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav.write(tmp_file.name, sample_rate, audio_data)
        return tmp_file.name

    def transcribe(self, audio_path):
        """
        Chuyển file âm thanh → văn bản tiếng Việt
        Trả về string hoặc "" nếu không nhận được gì
        """
        result = self.model.transcribe(
            audio_path,
            language="vi",       # Tiếng Việt
            fp16=True,           # Dùng float16 trên GPU, nhanh hơn
            temperature=0.0      # Deterministic output (không random)
        )
        text = result["text"].strip().lower()
        rospy.loginfo(f"[Voice] Nhận diện được: '{text}'")
        os.unlink(audio_path)  # Xoá file tạm
        return text

    def extract_bottle_count(self, text):
        """
        Trích xuất số chai nước từ câu nói
        Ví dụ: "cho tôi 2 chai nước" → 2
                "lấy ba chai" → 3
        """
        # Mapping chữ số tiếng Việt → số
        viet_numbers = {
            "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
            "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
            "6": 6, "7": 7, "8": 8, "9": 9, "10": 10
        }
        for word, number in viet_numbers.items():
            if word in text:
                return number
        return 1  # Mặc định 1 chai nếu không nghe được số

    def contains_wake_word(self, text):
        """Kiểm tra câu nói có chứa từ khoá kích hoạt không"""
        wake_words = [WAKE_WORD, "robot", "rô bốt", "robô"]
        return any(w in text for w in wake_words)

    def extract_table_number(self, text):
        """
        Trích xuất số bàn từ câu nói
        Ví dụ: "robot bàn số 3 cho tôi nước" → 3
                "bàn hai lấy nước" → 2
        Trả về số bàn (int) hoặc None nếu không nói bàn
        """
        # Mapping chữ số tiếng Việt → số
        viet_table_numbers = {
            "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
            "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
            "6": 6, "7": 7, "8": 8, "9": 9, "10": 10
        }
        # Tìm pattern "bàn [số/chữ]" hoặc "bàn số [số/chữ]"
        import re
        # Thử tìm "bàn số X" hoặc "bàn X"
        match = re.search(r'bàn\s*(?:số\s*)?(\w+)', text)
        if match:
            word = match.group(1)
            if word in viet_table_numbers:
                table_num = viet_table_numbers[word]
                if table_num in TABLE_POSITIONS:
                    return table_num
        return None


# ===========================================================================
# PHẦN 6: MODULE NHẬN DIỆN KHUÔN MẶT (InsightFace)
# ===========================================================================

class FaceRecognizer:
    """
    Nhận diện khuôn mặt dùng InsightFace
    - buffalo_l: chính xác cao (mặc định, khuyên dùng khi có GPU)
    - buffalo_s: nhẹ hơn, dùng khi chạy CPU
    """

    def __init__(self, model_name=None):
        # Tự chọn model theo phần cứng
        if model_name is None:
            model_name = "buffalo_l" if torch.cuda.is_available() else "buffalo_s"
        rospy.loginfo(f"[Face] Đang khởi tạo InsightFace ({model_name})...")
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1,
            det_size=(640, 640)
        )
        rospy.loginfo(f"[Face] InsightFace {model_name} ready!")

    def get_face_embedding(self, frame_bgr, box=None):
        """
        Lấy embedding của khuôn mặt trong frame
        box: (x1,y1,x2,y2) nếu muốn crop vùng cụ thể
        Trả về numpy array (512,) hoặc None nếu không có mặt
        """
        if box is not None:
            x1, y1, x2, y2 = box
            # Chỉ lấy nửa trên của bounding box (vùng đầu/mặt)
            y_mid = y1 + (y2 - y1) // 2
            region = frame_bgr[max(0,y1):y_mid, max(0,x1):min(FRAME_WIDTH,x2)]
        else:
            region = frame_bgr

        if region.size == 0:
            return None

        # InsightFace cần RGB
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        faces = self.app.get(region_rgb)

        if not faces:
            return None

        # Chọn khuôn mặt có diện tích lớn nhất (người đứng gần nhất)
        largest_face = max(faces, key=lambda f: (
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))
        return largest_face.embedding  # numpy array (512,)

    def compare_embeddings(self, emb1, emb2):
        """
        So sánh 2 embedding khuôn mặt
        Trả về cosine distance (0=giống nhau, 2=khác nhau)
        """
        return 1 - np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )

    def detect_all_faces(self, frame_bgr):
        """
        Phát hiện TẤT CẢ khuôn mặt trong frame.
        Trả về danh sách các face objects của InsightFace,
        mỗi face có: bbox, embedding, landmark, pose (chiếu quay đầu).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        region_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(region_rgb)
        return faces if faces else []

    def find_person_looking_at_camera(self, frame_bgr, speak_confirm_callback=None):
        """
        Tìm người đang NHÌN THẲNG vào camera (robot).

        Nguyên lý:
        - InsightFace trả về pose (pitch, yaw, roll) của mỗi khuôn mặt
        - Người nhìn thẳng vào camera có yaw ≈ 0 và pitch ≈ 0
        - Ta chọn người có góc lệch nhỏ nhất (gaze score cao nhất)

        Trả về:
          - (face, score) nếu tìm thấy người đang nhìn robot
          - (None, 0) nếu không tìm thấy
        """
        faces = self.detect_all_faces(frame_bgr)
        if not faces:
            return None, 0

        best_face = None
        best_score = 0

        for face in faces:
            # InsightFace buffalo_l có thuộc tính pose (3 góc: pitch, yaw, roll)
            # Nếu không có pose, dùng bbox position để ước lượng
            if hasattr(face, 'pose'):
                pitch, yaw, roll = face.pose
            else:
                # Fallback: ước lượng từ landmarks nếu có
                # Người nhìn thẳng → landmark mũi ở giữa 2 mắt
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    lmk = face.landmark_2d_106
                    # Mũi = point 86, mắt trái = 33, mắt phải = 87 (xấp xỉ)
                    nose_x = lmk[86][0]
                    left_eye_x = lmk[33][0]
                    right_eye_x = lmk[87][0]
                    # Nếu mũi ở giữa 2 mắt → đang nhìn thẳng
                    mid_eye_x = (left_eye_x + right_eye_x) / 2
                    yaw = (nose_x - mid_eye_x) / max(abs(right_eye_x - left_eye_x), 1) * 45
                    pitch = 0
                elif hasattr(face, 'kps') and face.kps is not None:
                    # Fallback 2: dùng 5-point landmarks (mắt trái, mắt phải, mũi, miệng trái, miệng phải)
                    kps = face.kps
                    nose_x = kps[2][0]          # Mũi
                    left_eye_x = kps[0][0]      # Mắt trái
                    right_eye_x = kps[1][0]     # Mắt phải
                    mid_eye_x = (left_eye_x + right_eye_x) / 2
                    eye_dist = max(abs(right_eye_x - left_eye_x), 1)
                    yaw = (nose_x - mid_eye_x) / eye_dist * 45  # Ước lượng góc yaw
                    pitch = 0
                else:
                    # Không có thông tin hướng mặt, giả sử nhìn thẳng
                    yaw = 0
                    pitch = 0

            # Tính gaze score: càng gần 0 độ càng tốt (người nhìn thẳng camera)
            # Score = 1 khi nhìn thẳng, giảm khi quay đầu
            angle_deviation = np.sqrt(yaw**2 + pitch**2)  # Tổng góc lệch
            gaze_score = max(0, 1.0 - angle_deviation / 45.0)  # Normalize về [0, 1]

            # Chỉ chọn người thực sự đang nhìn vào robot
            if gaze_score > GAZE_SCORE_THRESHOLD and gaze_score > best_score:
                best_score = gaze_score
                best_face = face

        return best_face, best_score


# ===========================================================================
# PHẦN 6.5: MODULE TÌM KHÁCH QUANH BÀN (CustomerLocator)
# ===========================================================================

class CustomerLocator:
    """
    Tìm đúng người đã gọi robot khi bàn có nhiều khách.

    Nguyên lý hoạt động:
    ┌─────────────────────────────────────────────────────────┐
    │  1. Robot đến bàn, dừng ở vị trí tiếp cận đầu tiên    │
    │  2. Phát loa: "Xin hãy nhìn vào tôi"                  │
    │  3. Dùng camera + gaze detection tìm ai đang nhìn      │
    │     → Nếu tìm thấy → xác nhận đúng người              │
    │     → Nếu không tìm thấy → di chuyển sang ghế kế tiếp │
    │  4. Lặp lại cho tất cả 4 vị trí quanh bàn             │
    │  5. Nếu quét hết 4 phía vẫn không thấy → nhờ khách    │
    │     nói lại hoặc vẫy tay                               │
    └─────────────────────────────────────────────────────────┘

    Điều kiện phần cứng:
    - Camera phải gắn trên robot (hướng ra trước)
    - Loa để phát hướng dẫn bằng giọng nói (tùy chọn, dùng espeak/gTTS)
    """

    def __init__(self, face_recognizer, navigator, camera, camera_lock):
        """
        Khởi tạo CustomerLocator.
        - face_recognizer: đối tượng FaceRecognizer đã được khởi tạo
        - navigator: đối tượng Navigator để di chuyển robot
        - camera: cv2.VideoCapture
        - camera_lock: threading.Lock() bảo vệ camera
        """
        self.face_rec = face_recognizer
        self.navigator = navigator
        self.camera = camera
        self.camera_lock = camera_lock

    def locate_customer_at_table(self, table_number, state_callback=None, gui_callback=None):
        """
        Quét quanh bàn để tìm người đang nhìn vào robot.

        Thuật toán:
        1. Ở mỗi vị trí tiếp cận (4 ghế):
           a. Robot di chuyển tới vị trí đối diện ghế đó
           b. Phát thông báo "Xin hãy nhìn vào camera"
           c. Quét camera trong LOCATE_DWELL_TIME giây
           d. Nếu phát hiện người nhìn vào camera → trả về face + vị trí
        2. Nếu quét hết 4 phía không thấy → thử thêm 1 vòng nữa
        3. Nếu vẫn không → trả về None

        Tham số:
        - table_number: số bàn cần quét
        - state_callback: hàm để cập nhật state label trên GUI
        - gui_callback: hàm để cập nhật thông tin trên GUI

        Trả về:
        - (face_embedding, seat_position, seat_index) nếu tìm thấy
        - (None, None, -1) nếu không tìm thấy
        """
        if table_number not in TABLE_POSITIONS:
            rospy.logerr(f"[Locate] Bàn {table_number} không tồn tại!")
            return None, None, -1

        table = TABLE_POSITIONS[table_number]
        positions = table["approach_positions"]
        total_positions = len(positions)
        start_time = time.time()

        rospy.loginfo(f"[Locate] 🔍 Bắt đầu quét {total_positions} vị trí quanh {table['name']}")

        # ============================================
        # VÒNG LẶP: Quét từng ghế quanh bàn
        # ============================================
        for attempt in range(2):  # Thử tối đa 2 vòng
            for seat_idx, pos in enumerate(positions):
                # Kiểm tra timeout tổng
                if time.time() - start_time > LOCATE_TIMEOUT:
                    rospy.logwarn("[Locate] ⚠️ Hết thời gian tìm khách!")
                    return None, None, -1

                # Cập nhật GUI
                seat_label = pos["seat"]
                msg = f"🔍 Quét ghế {seat_label} ({seat_idx+1}/{total_positions}) - Vòng {attempt+1}"
                rospy.loginfo(f"[Locate] {msg}")
                if gui_callback:
                    gui_callback(msg)

                # ---- Di chuyển tới vị trí đối diện ghế này ----
                # Vòng 1 seat 0: đã ở vị trí này rồi (navigate_to_table đã đưa đến)
                if attempt == 0 and seat_idx == 0:
                    rospy.loginfo("[Locate] Đang ở vị trí đầu tiên, bắt đầu quét")
                else:
                    success = self.navigator.navigate_to_seat(table_number, seat_idx)
                    if not success:
                        rospy.logwarn(f"[Locate] Không thể đi tới ghế {seat_label}, bỏ qua")
                        continue

                # ---- Phát thông báo bằng loa (nếu có espeak) ----
                try:
                    import subprocess
                    subprocess.Popen(
                        ["espeak", "-v", "vi", "Xin hãy nhìn vào camera"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except FileNotFoundError:
                    pass  # espeak không có sẵn, bỏ qua

                # ---- Quét camera trong LOCATE_DWELL_TIME giây ----
                scan_start = time.time()
                best_face_at_seat = None
                best_score_at_seat = 0

                while time.time() - scan_start < LOCATE_DWELL_TIME:
                    with self.camera_lock:
                        ret, frame = self.camera.read()
                    if not ret:
                        time.sleep(0.1)
                        continue

                    # Tìm người đang nhìn vào camera
                    face, score = self.face_rec.find_person_looking_at_camera(frame)

                    if face is not None and score > best_score_at_seat:
                        best_face_at_seat = face
                        best_score_at_seat = score

                    time.sleep(0.1)  # Kiểm tra 10 lần/giây

                # ---- Kiểm tra kết quả quét tại ghế này ----
                if best_face_at_seat is not None:
                    rospy.loginfo(
                        f"[Locate] ✅ TÌM THẤY khách tại ghế {seat_label}! "
                        f"Gaze score: {best_score_at_seat:.2f}"
                    )
                    # Trả về embedding, vị trí ghế, index
                    seat_position = (pos["x"], pos["y"])
                    return best_face_at_seat.embedding, seat_position, seat_idx
                else:
                    rospy.loginfo(
                        f"[Locate] ❌ Không thấy ai nhìn robot tại ghế {seat_label}"
                    )

            # Hết 1 vòng, thông báo
            if attempt == 0:
                rospy.loginfo("[Locate] 🔄 Chưa tìm thấy, quét lại vòng 2...")
                if gui_callback:
                    gui_callback("🔄 Chưa tìm thấy khách, quét lại...")
                # Phát loa nhờ khách nhìn vào robot
                try:
                    import subprocess
                    subprocess.Popen(
                        ["espeak", "-v", "vi",
                         "Xin quý khách đã gọi nước vui lòng nhìn vào tôi"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except FileNotFoundError:
                    pass

        # Quét hết 2 vòng vẫn không tìm thấy
        rospy.logwarn("[Locate] ⚠️ Không tìm thấy khách sau 2 vòng quét!")
        return None, None, -1


# ===========================================================================
# PHẦN 7: MODULE NHẬN DIỆN CHAI NƯỚC (YOLO)
# ===========================================================================

class BottleDetector:
    """
    Dùng YOLOv11m để đếm số chai nước đã đặt lên robot
    Hỗ trợ 2 chế độ:
      - COCO pretrained: dùng class 39 ("bottle") - nhanh, không cần train
      - Custom model: train riêng trên chai nước - chính xác hơn nhiều
    """

    def __init__(self, model_path=None):
        # Tự động chọn model path
        if model_path is None:
            custom_path = "/app/assets/models/water_bottle_best.pt"
            if BOTTLE_CUSTOM_MODEL and os.path.exists(custom_path):
                model_path = custom_path
                rospy.loginfo("[Bottle] Sử dụng custom model (water_bottle)")
            else:
                model_path = "yolo11m.pt"  # YOLOv11 Medium - tốt hơn s (51.5 vs 47.0 mAP)
                rospy.loginfo("[Bottle] Sử dụng COCO pretrained (yolo11m)")

        rospy.loginfo(f"[Bottle] Đang load YOLO model: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        # Warm up model (inference 1 lần để load toàn bộ vào GPU, tránh lag lần đầu)
        dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device)
        rospy.loginfo(f"[Bottle] YOLO ready trên {self.device}")

        # Xác định class ID cần detect
        if BOTTLE_CUSTOM_MODEL:
            self.target_class_id = 0      # Custom model: class 0 = chai nước
        else:
            self.target_class_id = BOTTLE_CLASS_ID  # COCO: class 39 = bottle

    def count_bottles(self, frame_bgr):
        """
        Đếm số chai nước trong frame
        Trả về (count, annotated_frame, confidences)
        """
        results = self.model(
            frame_bgr,
            verbose=False,
            conf=BOTTLE_CONFIDENCE,
            iou=BOTTLE_IOU_THRESHOLD,
            device=self.device,
            classes=[self.target_class_id]  # Chỉ detect class cần thiết, nhanh hơn
        )
        bottle_count = 0
        confidences = []
        annotated = frame_bgr.copy()

        for result in results:
            for box in result.boxes:
                bottle_count += 1
                conf = float(box.conf[0])
                confidences.append(conf)

                # Vẽ bounding box với màu theo confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Xanh lá khi confidence cao, vàng khi thấp
                color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"Chai {bottle_count} ({conf:.0%})"
                # Vẽ nền cho text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(annotated, label,
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1)

        # Hiển thị tổng số chai ở góc trên
        avg_conf = np.mean(confidences) if confidences else 0
        cv2.putText(annotated, f"Tong: {bottle_count} chai (TB: {avg_conf:.0%})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return bottle_count, annotated, confidences

    def verify_bottle_count(self, camera, camera_lock, required_count, timeout=60, gui_ref=None):
        """
        Kiểm tra xem số chai đã đủ chưa trong N giây
        Dùng voting: đếm đủ trong BOTTLE_CHECK_FRAMES frame liên tiếp → xác nhận
        camera_lock: threading.Lock() bảo vệ camera giữa các thread
        gui_ref: tham chiếu MainWindow để cập nhật _last_bottle_count
        Trả về True khi đủ chai, False khi timeout
        """
        rospy.loginfo(f"[Bottle] Chờ đủ {required_count} chai nước...")
        start_time = time.time()
        consecutive_correct = 0  # Số frame liên tiếp đếm đúng

        while time.time() - start_time < timeout:
            with camera_lock:
                ret, frame = camera.read()
            if not ret:
                continue

            count, _, confs = self.count_bottles(frame)

            # Cập nhật số đếm để GUI hiển thị
            if gui_ref is not None:
                gui_ref._last_bottle_count = count
                gui_ref._last_bottle_confs = confs

            if count >= required_count:
                consecutive_correct += 1
                rospy.loginfo(f"[Bottle] Đếm được {count}/{required_count} chai ({consecutive_correct}/{BOTTLE_CHECK_FRAMES})")
                if consecutive_correct >= BOTTLE_CHECK_FRAMES:
                    rospy.loginfo("[Bottle] ✅ Đã đủ số chai!")
                    return True
            else:
                consecutive_correct = 0  # Reset nếu đếm sai

            time.sleep(BOTTLE_CHECK_INTERVAL)

        rospy.logwarn("[Bottle] ⚠️ Timeout! Không đủ chai sau 60 giây")
        return False


# ===========================================================================
# PHẦN 8: MODULE ĐIỀU HƯỚNG (Navigation)
# ===========================================================================

class Navigator:
    """
    Điều khiển robot di chuyển dùng move_base action server
    move_base đã có sẵn trong MiR robot, tự động né vật cản
    """

    def __init__(self):
        rospy.loginfo("[Nav] Kết nối move_base action server...")
        # SimpleActionClient giao tiếp với move_base
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server(timeout=rospy.Duration(10))
        rospy.loginfo("[Nav] move_base ready!")

        # TF listener để lấy vị trí robot
        self.tf_listener = tf.TransformListener()
        self.robot_pose = None  # (x, y, yaw)

        # Subscribe vị trí robot từ TF liên tục
        self._pose_thread = threading.Thread(target=self._update_pose_loop, daemon=True)
        self._pose_thread.start()

    def _update_pose_loop(self):
        """Cập nhật vị trí robot từ TF mỗi 200ms"""
        rate = rospy.Rate(5)  # 5Hz
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.tf_listener.lookupTransform(
                    '/map',       # Frame đích (bản đồ)
                    '/base_link', # Frame nguồn (thân robot)
                    rospy.Time(0) # Lấy transform mới nhất
                )
                yaw = euler_from_quaternion(rot)[2]
                self.robot_pose = (trans[0], trans[1], yaw)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
            rate.sleep()

    def get_robot_pose(self):
        """Trả về vị trí robot hiện tại (x, y, yaw) hoặc None"""
        return self.robot_pose

    def _yaw_to_quaternion(self, yaw):
        """Chuyển góc yaw (radians) → quaternion (x, y, z, w) để gửi cho move_base"""
        return 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)

    def navigate_to(self, x, y, yaw=None, timeout=NAVIGATION_TIMEOUT):
        """
        Di chuyển robot đến tọa độ (x, y) trên bản đồ
        yaw: hướng nhìn khi đến nơi (radians), None = tự quyết
        Trả về True nếu thành công, False nếu thất bại
        """
        rospy.loginfo(f"[Nav] 🚗 Đang di chuyển tới ({x:.2f}, {y:.2f})")

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        # Tính hướng: nếu không chỉ định thì hướng về phía goal từ vị trí hiện tại
        if yaw is None and self.robot_pose:
            dx = x - self.robot_pose[0]
            dy = y - self.robot_pose[1]
            yaw = np.arctan2(dy, dx)
        elif yaw is None:
            yaw = 0.0

        qx, qy, qz, qw = self._yaw_to_quaternion(yaw)
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw

        self.client.send_goal(goal)
        finished = self.client.wait_for_result(rospy.Duration(timeout))

        if not finished:
            rospy.logwarn("[Nav] ⚠️ Timeout! Hủy goal.")
            self.client.cancel_goal()
            return False

        state = self.client.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("[Nav] ✅ Đã đến nơi!")
            return True
        else:
            rospy.logwarn(f"[Nav] ❌ Thất bại, state={state}")
            return False

    def navigate_to_water_station(self):
        """Di chuyển về kho lấy nước"""
        return self.navigate_to(
            WATER_STATION["x"],
            WATER_STATION["y"],
            WATER_STATION["yaw"]
        )

    def navigate_to_customer(self, customer_position):
        """Di chuyển về vị trí khách hàng"""
        x, y = customer_position
        return self.navigate_to(x, y)

    def navigate_to_table(self, table_number):
        """
        Di chuyển robot đến vị trí tiếp cận ĐẦU TIÊN của bàn (approach_positions[0]).
        Lấy tọa độ từ TABLE_POSITIONS đã cấu hình sẵn.
        Trả về True nếu thành công, False nếu thất bại.
        """
        if table_number not in TABLE_POSITIONS:
            rospy.logerr(f"[Nav] ❌ Không tìm thấy bàn số {table_number}!")
            return False
        table = TABLE_POSITIONS[table_number]
        # Đi tới approach position đầu tiên
        first_approach = table["approach_positions"][0]
        rospy.loginfo(f"[Nav] 🚗 Đi tới {table['name']} - ghế {first_approach['seat']}")
        return self.navigate_to(first_approach["x"], first_approach["y"], first_approach["yaw"])

    def navigate_to_seat(self, table_number, seat_index):
        """
        Di chuyển robot đến một ghế cụ thể của bàn.
        seat_index: 0-3 (tương ứng 4 ghế quanh bàn)
        Robot sẽ đứng đối diện ghế đó, camera nhìn thẳng vào khách.
        """
        if table_number not in TABLE_POSITIONS:
            return False
        table = TABLE_POSITIONS[table_number]
        positions = table["approach_positions"]
        if seat_index >= len(positions):
            return False
        pos = positions[seat_index]
        rospy.loginfo(f"[Nav] 🚗 Đi tới {table['name']} - ghế {pos['seat']} (vị trí {seat_index+1}/4)")
        return self.navigate_to(pos["x"], pos["y"], pos["yaw"])

    def stop(self):
        """Dừng robot ngay lập tức"""
        self.client.cancel_all_goals()


# ===========================================================================
# PHẦN 9: CLASS BẢN ĐỒ (MapCanvas)
# ===========================================================================

class MapCanvas(FigureCanvas):
    """Hiển thị bản đồ OccupancyGrid trong PyQt5"""

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=80)
        self.fig.patch.set_facecolor('#2c2c2c')
        self.ax.set_facecolor('#2c2c2c')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Dữ liệu bản đồ
        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin_x = 0
        self.map_origin_y = 0

        # Các điểm hiển thị
        self.robot_pose = None
        self.customer_position = None
        self.water_station = (WATER_STATION["x"], WATER_STATION["y"])
        self.goal_position = None

    def update_map(self, map_data, resolution, origin_x, origin_y):
        self.map_data = map_data
        self.map_resolution = resolution
        self.map_origin_x = origin_x
        self.map_origin_y = origin_y
        self.draw_map()

    def _world_to_pixel(self, wx, wy):
        """Chuyển tọa độ thế giới (mét) → pixel trên bản đồ"""
        px = (wx - self.map_origin_x) / self.map_resolution
        py = (wy - self.map_origin_y) / self.map_resolution
        return px, py

    def draw_map(self):
        """Vẽ lại toàn bộ bản đồ và các marker"""
        if self.map_data is None:
            return
        try:
            self.ax.clear()
            self.ax.set_aspect('equal')
            self.ax.set_facecolor('#2c2c2c')

            # Vẽ bản đồ nền
            self.ax.imshow(self.map_data, cmap='gray', origin='lower',
                           extent=[0, self.map_data.shape[1],
                                   0, self.map_data.shape[0]])

            # Vẽ kho nước (ngôi sao vàng)
            wx, wy = self._world_to_pixel(self.water_station[0], self.water_station[1])
            self.ax.plot(wx, wy, 'y*', markersize=15, label='Kho nước')

            # Vẽ vị trí robot (hình chữ nhật xanh + mũi tên)
            if self.robot_pose:
                rx, ry = self._world_to_pixel(self.robot_pose[0], self.robot_pose[1])
                self.ax.plot(rx, ry, 'bs', markersize=10, label='Robot')
                # Mũi tên chỉ hướng
                arrow_len = 10
                dx = arrow_len * np.cos(self.robot_pose[2])
                dy = arrow_len * np.sin(self.robot_pose[2])
                self.ax.arrow(rx, ry, dx, dy,
                              head_width=4, head_length=3, fc='cyan', ec='cyan')

            # Vẽ vị trí khách hàng (chấm xanh lá)
            if self.customer_position:
                cx, cy = self._world_to_pixel(self.customer_position[0], self.customer_position[1])
                self.ax.plot(cx, cy, 'go', markersize=10, label='Khách hàng')

            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.legend(loc='upper right', fontsize=7,
                           facecolor='#3c3c3c', labelcolor='white')
            self.draw()
        except Exception as e:
            rospy.logerr(f"[Map] Lỗi vẽ bản đồ: {e}")


# ===========================================================================
# PHẦN 10: CỬA SỔ CHÍNH (MainWindow)
# ===========================================================================

class MainWindow(QMainWindow):
    """
    Giao diện chính của hệ thống
    Bố cục: [Camera Feed | Thông tin đơn hàng] [Bản đồ | Nút điều khiển]
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Robot Phục Vụ Nước - MiR100")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #1a1a2e; color: white;")

        # ---- Khởi tạo các module ----
        self.db = CustomerDatabase()
        self.face_recognizer = FaceRecognizer()
        self.bottle_detector = BottleDetector()
        self.navigator = Navigator()
        self.voice_recognizer = VoiceRecognizer(model_size="small")

        # ---- Trạng thái hệ thống ----
        self.state = RobotState.IDLE
        self.current_customer = None        # Thông tin khách hàng hiện tại
        self.customer_position = None       # Vị trí khách (x, y) trên bản đồ
        self.customer_table = None          # Số bàn khách đang ngồi
        self.customer_seat_index = -1       # Index ghế khách ngồi (0-3)
        self.customer_initial_embedding = None  # Embedding ban đầu từ gaze detection
        self.required_bottles = 0           # Số chai khách yêu cầu
        self.face_embeddings = []           # List embedding đang thu thập (tối đa 3)
        self.current_frame = None           # Frame camera hiện tại

        # ---- Queue thread-safe ----
        self.map_queue = queue.Queue()      # Queue nhận dữ liệu bản đồ từ ROS callback

        # ---- Khởi tạo camera ----
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.camera_lock = threading.Lock()  # Lock bảo vệ camera giữa các thread

        # ---- Khởi tạo CustomerLocator (tìm khách quanh bàn) ----
        self.customer_locator = CustomerLocator(
            self.face_recognizer,
            self.navigator,
            self.camera,
            self.camera_lock
        )

        # ---- Build UI ----
        self._build_ui()

        # ---- ROS Subscribers ----
        rospy.Subscriber('/map', OccupancyGrid, self._map_callback)

        # ---- Timers ----
        # Timer cập nhật camera (30ms ≈ 33fps)
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._update_camera)
        self.camera_timer.start(30)

        # Timer cập nhật bản đồ từ queue (100ms)
        self.map_timer = QTimer()
        self.map_timer.timeout.connect(self._process_map_queue)
        self.map_timer.start(100)

        # Timer cập nhật vị trí robot trên bản đồ (500ms)
        self.pose_timer = QTimer()
        self.pose_timer.timeout.connect(self._update_robot_pose)
        self.pose_timer.start(500)

        # ---- ROS spin trong thread riêng ----
        self.ros_thread = threading.Thread(target=rospy.spin, daemon=True)
        self.ros_thread.start()

        rospy.loginfo("[Main] Hệ thống khởi động xong! Đang chờ khách...")

    # =========================================================================
    # BUILD UI
    # =========================================================================

    def _build_ui(self):
        """Xây dựng giao diện người dùng"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ---- Panel trái: Camera ----
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #16213e; border-radius: 8px;")
        left_panel.setMinimumWidth(650)
        left_layout = QVBoxLayout(left_panel)

        # Tiêu đề
        title = QLabel("📷 Camera Robot")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00d4ff; padding: 5px;")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # Hiển thị video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #0f0f1a; border: 2px solid #00d4ff; border-radius: 4px;")
        self.video_label.mousePressEvent = self._on_video_click  # Click để chụp mặt
        left_layout.addWidget(self.video_label)

        # Trạng thái hệ thống
        self.state_label = QLabel("🟢 Trạng thái: Đang chờ khách")
        self.state_label.setStyleSheet("font-size: 12pt; color: #00ff88; padding: 5px;")
        self.state_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.state_label)

        # Thông tin đơn hàng
        self.order_label = QLabel("📦 Đơn hàng: Chưa có")
        self.order_label.setStyleSheet("font-size: 11pt; color: #ffcc00; padding: 3px;")
        self.order_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.order_label)

        # Trạng thái nhận diện mặt
        self.face_label = QLabel("👤 Khuôn mặt: Chưa đăng ký")
        self.face_label.setStyleSheet("font-size: 11pt; color: #aaaaaa; padding: 3px;")
        self.face_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.face_label)

        # Nút điều khiển
        btn_layout = QHBoxLayout()

        self.voice_btn = QPushButton("🎤 Gọi Robot")
        self.voice_btn.setStyleSheet(self._btn_style("#00aa44"))
        self.voice_btn.clicked.connect(self._start_voice_listening)
        btn_layout.addWidget(self.voice_btn)

        self.cancel_btn = QPushButton("❌ Hủy")
        self.cancel_btn.setStyleSheet(self._btn_style("#cc2200"))
        self.cancel_btn.clicked.connect(self._cancel_current_task)
        btn_layout.addWidget(self.cancel_btn)

        self.manual_nav_btn = QPushButton("🏠 Về kho nước")
        self.manual_nav_btn.setStyleSheet(self._btn_style("#0066cc"))
        self.manual_nav_btn.clicked.connect(self._manual_go_to_water)
        btn_layout.addWidget(self.manual_nav_btn)

        left_layout.addLayout(btn_layout)
        main_layout.addWidget(left_panel)

        # ---- Panel phải: Bản đồ ----
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #16213e; border-radius: 8px;")
        right_layout = QVBoxLayout(right_panel)

        map_title = QLabel("🗺️ Bản đồ")
        map_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00d4ff; padding: 5px;")
        map_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(map_title)

        self.map_canvas = MapCanvas(self)
        right_layout.addWidget(self.map_canvas)

        # Thông tin vị trí
        self.pos_label = QLabel("📍 Vị trí robot: --")
        self.pos_label.setStyleSheet("font-size: 10pt; color: #aaaaaa; padding: 3px;")
        self.pos_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.pos_label)

        self.customer_pos_label = QLabel("👤 Vị trí khách: --")
        self.customer_pos_label.setStyleSheet("font-size: 10pt; color: #aaaaaa; padding: 3px;")
        self.customer_pos_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.customer_pos_label)

        main_layout.addWidget(right_panel)

    def _btn_style(self, color):
        return (
            f"QPushButton {{"
            f"  background-color: {color}; color: white;"
            f"  border: none; padding: 8px 12px;"
            f"  font-size: 11pt; font-weight: bold; border-radius: 4px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {color}cc; }}"
            f"QPushButton:disabled {{ background-color: #555555; }}"
        )

    # =========================================================================
    # CẬP NHẬT CAMERA
    # =========================================================================

    def _update_camera(self):
        """Đọc frame từ camera và hiển thị lên GUI (chạy mỗi 30ms)"""
        with self.camera_lock:
            ret, frame = self.camera.read()
        if not ret:
            return

        self.current_frame = frame.copy()

        # Tùy trạng thái, vẽ thông tin khác nhau lên frame
        annotated = self._annotate_frame(frame)

        # Chuyển BGR → RGB → QImage → QPixmap để hiển thị PyQt5
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def _annotate_frame(self, frame):
        """Vẽ thông tin trạng thái lên frame camera"""
        annotated = frame.copy()
        state_colors = {
            RobotState.IDLE:                       (0, 255, 0),
            RobotState.LISTENING:                  (0, 255, 255),
            RobotState.NAVIGATING_TO_CUSTOMER_FIRST: (255, 200, 0),
            RobotState.LOCATING_CUSTOMER:          (0, 200, 255),
            RobotState.FACE_REGISTER:              (255, 255, 0),
            RobotState.NAVIGATING_TO_WATER:        (255, 165, 0),
            RobotState.WAITING_FOR_BOTTLES:        (255, 0, 255),
            RobotState.NAVIGATING_TO_CUSTOMER:     (0, 165, 255),
            RobotState.DELIVERING:                 (0, 255, 0),
            RobotState.ERROR:                      (0, 0, 255),
        }
        color = state_colors.get(self.state, (255, 255, 255))

        # Vẽ thanh trạng thái ở đỉnh
        cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 40), (0, 0, 0), -1)
        cv2.putText(annotated, f"STATE: {self.state}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Nếu đang đợi chụp mặt → hiển thị hướng dẫn
        if self.state == RobotState.FACE_REGISTER:
            steps = ["Nhìn thẳng vào camera → Click", "Quay trái 45° → Click", "Quay phải 45° → Click"]
            step = len(self.face_embeddings)
            if step < 3:
                msg = steps[step]
                cv2.putText(annotated, msg, (10, FRAME_HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Nếu đang tìm khách quanh bàn → vẽ tất cả khuôn mặt + gaze info
        if self.state == RobotState.LOCATING_CUSTOMER:
            cv2.putText(annotated, "Xin hay nhin vao camera!",
                        (10, FRAME_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            # Vẽ bounding box cho mỗi khuôn mặt phát hiện được
            try:
                faces = self.face_recognizer.detect_all_faces(annotated)
                for face in faces:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
            except Exception:
                pass

        # Nếu đang kiểm tra chai → hiển thị thông tin (YOLO chạy riêng trong thread)
        if self.state == RobotState.WAITING_FOR_BOTTLES:
            count = getattr(self, '_last_bottle_count', 0)
            cv2.putText(annotated, f"Chai: {count}/{self.required_bottles}",
                        (10, FRAME_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        return annotated

    # =========================================================================
    # XỬ LÝ BẢN ĐỒ
    # =========================================================================

    def _map_callback(self, msg):
        """
        ROS callback khi nhận được bản đồ mới từ /map topic
        KHÔNG xử lý trực tiếp ở đây vì đây là ROS thread, không phải GUI thread
        → Đẩy vào queue, GUI thread sẽ xử lý sau
        """
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        data = np.array(msg.data).reshape((height, width)).astype(np.float32)
        # Chuyển OccupancyGrid (-1=unknown, 0=free, 100=occupied) → grayscale
        data[data == -1] = 128   # Unknown → xám
        data = 100 - data        # Đảo: tường đen, tự do trắng
        self.map_queue.put((data, resolution, origin_x, origin_y))

    def _process_map_queue(self):
        """Xử lý dữ liệu bản đồ từ queue (chạy trong GUI thread)"""
        try:
            while not self.map_queue.empty():
                data, res, ox, oy = self.map_queue.get_nowait()
                self.map_canvas.update_map(data, res, ox, oy)
        except queue.Empty:
            pass

    def _update_robot_pose(self):
        """Cập nhật vị trí robot lên bản đồ (chạy mỗi 500ms)"""
        pose = self.navigator.get_robot_pose()
        if pose:
            self.map_canvas.robot_pose = pose
            self.map_canvas.draw_map()
            self.pos_label.setText(f"📍 Robot: ({pose[0]:.2f}, {pose[1]:.2f}), yaw={np.degrees(pose[2]):.1f}°")

    # =========================================================================
    # LUỒNG CHÍNH: STATE MACHINE
    # =========================================================================

    def _start_voice_listening(self):
        """Bắt đầu lắng nghe giọng nói (chạy trong thread riêng)"""
        if self.state != RobotState.IDLE:
            rospy.logwarn("[Main] Robot đang bận, không thể nhận lệnh mới")
            return
        threading.Thread(target=self._voice_workflow, daemon=True).start()

    def _voice_workflow(self):
        """
        Luồng xử lý giọng nói (ĐÃ SỬA ĐÚNG FLOW):
        1. Ghi âm
        2. Nhận diện văn bản
        3. Trích xuất số chai + số bàn
        4. ĐI TỚI CHỖ KHÁCH TRƯỚC (quan trọng!)
        5. Sau khi đến nơi → đăng ký khuôn mặt
        """
        self._set_state(RobotState.LISTENING)

        audio_path = self.voice_recognizer.record_audio()
        text = self.voice_recognizer.transcribe(audio_path)

        if not text:
            rospy.logwarn("[Voice] Không nhận được âm thanh")
            self._set_state(RobotState.IDLE)
            return

        # Kiểm tra từ khoá kích hoạt
        if not self.voice_recognizer.contains_wake_word(text):
            rospy.loginfo(f"[Voice] Không chứa từ khoá kích hoạt, bỏ qua: '{text}'")
            self._set_state(RobotState.IDLE)
            return

        # Trích xuất số chai
        bottle_count = self.voice_recognizer.extract_bottle_count(text)
        self.required_bottles = bottle_count
        rospy.loginfo(f"[Voice] Khách yêu cầu {bottle_count} chai nước")

        # Trích xuất số bàn → xác định vị trí khách
        table_number = self.voice_recognizer.extract_table_number(text)

        if table_number is None:
            # Khách không nói số bàn → yêu cầu nói lại
            rospy.logwarn("[Voice] ⚠️ Không xác định được số bàn!")
            QTimer.singleShot(0, lambda: self.state_label.setText(
                "⚠️ Vui lòng nói lại kèm số bàn (VD: 'robot bàn 3 cho tôi 2 chai nước')"
            ))
            self._set_state(RobotState.IDLE)
            return

        self.customer_table = table_number
        table_info = TABLE_POSITIONS[table_number]
        self.customer_position = (table_info["x"], table_info["y"])

        rospy.loginfo(f"[Voice] 📍 Khách ở {table_info['name']} → vị trí ({table_info['x']}, {table_info['y']})")

        # Cập nhật GUI
        QTimer.singleShot(0, lambda: self.order_label.setText(
            f"📦 {table_info['name']}: {bottle_count} chai nước"
        ))
        QTimer.singleShot(0, lambda: self.customer_pos_label.setText(
            f"👤 Khách: {table_info['name']} ({table_info['x']:.1f}, {table_info['y']:.1f})"
        ))

        # Cập nhật vị trí khách trên bản đồ
        self.map_canvas.customer_position = self.customer_position
        QTimer.singleShot(0, lambda: self.map_canvas.draw_map())

        # ======================================================
        # BƯỚC QUAN TRỌNG: ĐI TỚI CHỖ KHÁCH TRƯỚC
        # Robot phải đến nơi khách ngồi để:
        #   - Chụp khuôn mặt khách (cần camera nhìn thấy mặt)
        #   - Xác nhận đúng người đặt nước
        # ======================================================
        self._set_state(RobotState.NAVIGATING_TO_CUSTOMER_FIRST)
        rospy.loginfo(f"[Main] 🚗 Đi tới {table_info['name']} để gặp khách...")

        success = self.navigator.navigate_to_table(table_number)
        if not success:
            rospy.logerr(f"[Main] ❌ Không thể đi tới {table_info['name']}!")
            self._set_state(RobotState.ERROR)
            return

        rospy.loginfo(f"[Main] ✅ Đã đến {table_info['name']}, bắt đầu TÌM KHÁCH")

        # ======================================================
        # BƯỚC MỚI: TÌM ĐÚNG NGƯỜI GỌI QUANH BÀN
        # Robot xoay quanh bàn, dùng gaze detection để tìm
        # người đang nhìn vào camera (= người đã gọi robot)
        # ======================================================
        self._set_state(RobotState.LOCATING_CUSTOMER)

        def gui_update_locate(msg):
            QTimer.singleShot(0, lambda: self.face_label.setText(f"🔍 {msg}"))

        # Gọi CustomerLocator để quét quanh bàn
        found_embedding, seat_pos, seat_idx = self.customer_locator.locate_customer_at_table(
            table_number,
            state_callback=None,
            gui_callback=gui_update_locate
        )

        if found_embedding is not None:
            # ✅ Tìm thấy đúng người gọi!
            self.customer_seat_index = seat_idx
            self.customer_initial_embedding = found_embedding

            # Cập nhật vị trí khách chính xác = ghế khách ngồi (không phải tâm bàn)
            seat_info = TABLE_POSITIONS[table_number]["approach_positions"][seat_idx]
            self.customer_position = seat_pos  # Vị trí robot đứng đối diện khách
            rospy.loginfo(
                f"[Main] ✅ Xác định khách ngồi ghế {seat_info['seat']} - "
                f"Bàn {table_number}"
            )
            QTimer.singleShot(0, lambda: self.face_label.setText(
                f"✅ Tìm thấy khách tại ghế {seat_info['seat']}! Đang chụp mặt..."
            ))
            QTimer.singleShot(0, lambda: self.customer_pos_label.setText(
                f"👤 Khách: {table_info['name']} ghế {seat_info['seat']}"
            ))

            # Đã tìm thấy → bắt đầu chụp mặt
            # Embedding đầu tiên đã có từ gaze detection
            self.face_embeddings = [found_embedding]
            self._set_state(RobotState.FACE_REGISTER)
            QTimer.singleShot(0, lambda: self.face_label.setText(
                "👤 Đã có ảnh thẳng mặt. Quay trái 45° → Click (bước 2/3)"
            ))
        else:
            # ❌ Không tìm thấy ai nhìn vào robot
            # Fallback: vẫn chụp mặt nhưng thông báo cho user
            rospy.logwarn("[Main] ⚠️ Không xác định được ai là người gọi!")
            QTimer.singleShot(0, lambda: self.face_label.setText(
                "⚠️ Chưa tìm thấy khách. Click vào camera khi thấy đúng người (bước 1/3)"
            ))
            self.face_embeddings = []
            self._set_state(RobotState.FACE_REGISTER)

    def _on_video_click(self, event):
        """
        Xử lý khi người dùng click vào video
        Dùng để chụp khuôn mặt khách hàng qua 3 góc
        """
        if self.state != RobotState.FACE_REGISTER:
            return
        if self.current_frame is None:
            return

        # Lấy embedding khuôn mặt từ frame hiện tại
        embedding = self.face_recognizer.get_face_embedding(self.current_frame)

        if embedding is None:
            self.face_label.setText("❌ Không tìm thấy khuôn mặt! Thử lại.")
            self.face_label.setStyleSheet("font-size: 11pt; color: red;")
            return

        self.face_embeddings.append(embedding)
        step = len(self.face_embeddings)
        rospy.loginfo(f"[Face] Đã chụp bước {step}/3")

        if step == 1:
            self.face_label.setText("✅ Bước 1 xong. Quay trái 45° → Click")
            self.face_label.setStyleSheet("font-size: 11pt; color: yellow;")
        elif step == 2:
            self.face_label.setText("✅ Bước 2 xong. Quay phải 45° → Click")
            self.face_label.setStyleSheet("font-size: 11pt; color: yellow;")
        elif step == 3:
            # Đủ 3 góc → tìm trong database hoặc đăng ký mới
            self._complete_face_registration()

    def _complete_face_registration(self):
        """
        Hoàn tất đăng ký khuôn mặt:
        - Tìm khách trong database
        - Nếu khách mới → tạo hồ sơ mới
        - Vị trí khách đã được xác định từ trước (qua số bàn)
        - Bắt đầu luồng giao nước (đi lấy nước rồi quay lại)
        """
        rospy.loginfo("[Face] Đang tìm khách trong database...")

        # Tìm khách qua embedding đầu tiên (nhìn thẳng)
        customer = self.db.find_customer(self.face_embeddings[0])

        if customer:
            rospy.loginfo(f"[Face] ✅ Nhận ra khách: {customer['name']}")
            self.face_label.setText(f"✅ Xin chào {customer['name']}!")
            self.face_label.setStyleSheet("font-size: 11pt; color: #00ff88;")
        else:
            rospy.loginfo("[Face] Khách mới, đang đăng ký...")
            customer = self.db.register_customer(self.face_embeddings)
            self.face_label.setText(f"✅ Đã đăng ký: {customer['name']}")
            self.face_label.setStyleSheet("font-size: 11pt; color: #00ff88;")

        self.current_customer = customer

        # Vị trí khách đã được xác định từ bước voice_workflow
        # (thông qua số bàn trong câu nói)
        # Không cần lấy vị trí robot nữa vì đã có customer_position từ TABLE_POSITIONS
        if self.customer_position:
            # Lưu lịch sử đặt hàng
            self.db.add_order(customer["id"], self.required_bottles, self.customer_position)
            table_name = TABLE_POSITIONS.get(self.customer_table, {}).get("name", "Không rõ")
            rospy.loginfo(f"[Face] Khách {customer['name']} tại {table_name}, cần {self.required_bottles} chai")

        # Bắt đầu luồng giao nước trong thread riêng
        # Flow: đi kho nước → chờ đặt chai → quay lại chỗ khách
        threading.Thread(target=self._delivery_workflow, daemon=True).start()

    def _delivery_workflow(self):
        """
        Luồng giao nước chính (chạy trong thread riêng để không block GUI):
        1. Đi tới kho lấy nước
        2. Chờ bếp đặt chai lên
        3. Xác nhận đủ chai
        4. Đi về chỗ khách
        5. Hoàn tất
        """
        # ---- Bước 1: Đi tới kho nước ----
        self._set_state(RobotState.NAVIGATING_TO_WATER)
        rospy.loginfo(f"[Delivery] 🚗 Đi tới kho nước ({WATER_STATION['x']}, {WATER_STATION['y']})")

        success = self.navigator.navigate_to_water_station()
        if not success:
            rospy.logerr("[Delivery] ❌ Không thể đến kho nước!")
            self._set_state(RobotState.ERROR)
            return

        # ---- Bước 2: Chờ đủ chai nước ----
        self._set_state(RobotState.WAITING_FOR_BOTTLES)
        rospy.loginfo(f"[Delivery] ⏳ Chờ {self.required_bottles} chai nước được đặt lên...")

        self._last_bottle_count = 0
        bottles_ready = self.bottle_detector.verify_bottle_count(
            self.camera,
            self.camera_lock,
            self.required_bottles,
            timeout=120,  # Chờ tối đa 2 phút
            gui_ref=self
        )

        if not bottles_ready:
            rospy.logwarn("[Delivery] ⚠️ Hết thời gian chờ chai nước!")
            # Vẫn tiếp tục giao với số chai hiện có (hoặc có thể return về IDLE)

        # ---- Bước 3: Đi về chỗ khách ----
        if self.customer_position is None:
            rospy.logerr("[Delivery] ❌ Không có vị trí khách hàng!")
            self._set_state(RobotState.ERROR)
            return

        self._set_state(RobotState.NAVIGATING_TO_CUSTOMER)
        rospy.loginfo(f"[Delivery] 🚗 Đi về chỗ khách tại {self.customer_position}")

        success = self.navigator.navigate_to_customer(self.customer_position)
        if not success:
            rospy.logerr("[Delivery] ❌ Không thể về chỗ khách!")
            self._set_state(RobotState.ERROR)
            return

        # ---- Bước 4: Hoàn tất ----
        self._set_state(RobotState.DELIVERING)
        rospy.loginfo("[Delivery] ✅ Đã giao nước! Chờ khách lấy...")

        # Chờ 15 giây để khách lấy nước
        time.sleep(15)

        # Reset về trạng thái chờ
        self._reset_to_idle()

    # =========================================================================
    # CÁC HÀM HỖ TRỢ
    # =========================================================================

    def _set_state(self, new_state):
        """Cập nhật trạng thái robot và UI"""
        self.state = new_state
        state_messages = {
            RobotState.IDLE:                       "🟢 Đang chờ khách",
            RobotState.LISTENING:                  "🎤 Đang nghe...",
            RobotState.NAVIGATING_TO_CUSTOMER_FIRST: "🚗 Đang đi tới bàn khách",
            RobotState.LOCATING_CUSTOMER:          "🔍 Đang tìm khách quanh bàn...",
            RobotState.FACE_REGISTER:              "📸 Đang chụp khuôn mặt",
            RobotState.NAVIGATING_TO_WATER:        "🚗 Đang đi lấy nước",
            RobotState.WAITING_FOR_BOTTLES:        "⏳ Chờ bếp đặt chai",
            RobotState.NAVIGATING_TO_CUSTOMER:     "🚗 Đang giao nước",
            RobotState.DELIVERING:                 "✅ Đã đến! Vui lòng lấy nước",
            RobotState.ERROR:                      "❌ Lỗi hệ thống",
        }
        msg = state_messages.get(new_state, new_state)
        # Cập nhật GUI phải từ main thread, dùng QTimer.singleShot
        QTimer.singleShot(0, lambda: self.state_label.setText(f"Trạng thái: {msg}"))
        rospy.loginfo(f"[State] → {new_state}")

    def _reset_to_idle(self):
        """Reset toàn bộ về trạng thái chờ ban đầu"""
        self.current_customer = None
        self.customer_position = None
        self.customer_table = None
        self.required_bottles = 0
        self.face_embeddings = []
        self.map_canvas.customer_position = None
        QTimer.singleShot(0, lambda: self.order_label.setText("📦 Đơn hàng: Chưa có"))
        QTimer.singleShot(0, lambda: self.face_label.setText("👤 Khuôn mặt: Chưa đăng ký"))
        QTimer.singleShot(0, lambda: self.face_label.setStyleSheet("font-size: 11pt; color: #aaaaaa;"))
        self._set_state(RobotState.IDLE)

    def _cancel_current_task(self):
        """Hủy task hiện tại và về trạng thái IDLE"""
        rospy.loginfo("[Main] Hủy task hiện tại")
        self.navigator.stop()
        self._reset_to_idle()

    def _manual_go_to_water(self):
        """Nút test thủ công: đi về kho nước"""
        threading.Thread(
            target=self.navigator.navigate_to_water_station,
            daemon=True
        ).start()

    def closeEvent(self, event):
        """Dọn dẹp khi đóng cửa sổ"""
        self.navigator.stop()
        self.camera.release()
        rospy.signal_shutdown("GUI closed")
        event.accept()


# ===========================================================================
# PHẦN 11: ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # Khởi tạo ROS node trước khi tạo QApplication
    rospy.init_node('water_delivery_robot', anonymous=False)

    # Tạo Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Water Delivery Robot")

    # Tạo và hiển thị cửa sổ chính
    window = MainWindow()
    window.show()

    # Chạy event loop Qt
    sys.exit(app.exec_())