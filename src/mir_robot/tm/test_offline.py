#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
import math
import yaml

os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_API'] = 'pyqt5'

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class MapLabelOffline(QLabel):
    def __init__(self, pgm_path, yaml_path):
        super().__init__()
        self.map_img = None
        self.map_data = None
        self.res = 0.05
        self.ox = 0.0
        self.oy = 0.0
        
        self.robot_px = None
        self.target_px = None
        self.goal_px = None
        self.goal_yaw = 0.0
        self.ray_pixels = []
        
        self.load_map(pgm_path, yaml_path)

    def load_map(self, pgm_path, yaml_path):
        if not os.path.exists(pgm_path):
            print(f"Không tìm thấy {pgm_path}")
            return
            
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                info = yaml.safe_load(f)
                self.res = info.get('resolution', 0.05)
                self.ox = info.get('origin', [0,0,0])[0]
                self.oy = info.get('origin', [0,0,0])[1]
                
        # Đọc PGM
        raw_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        self.h, self.w = raw_img.shape
        
        # Chuyển đổi PGM sang định dạng OccupancyGrid giả lập: 255 (obstacle), 0 (free)
        self.map_data = np.zeros_like(raw_img, dtype=np.uint8)
        self.map_data[raw_img < 200] = 255 # Vật cản (Đen trong PGM)
        self.map_data[raw_img >= 240] = 0  # Trống (Trắng trong PGM)
        
        # Ảnh để hiển thị GUI (Lật lại cho đúng chiều trục Y hướng lên)
        self.display_base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.display_base[self.map_data == 0] = [255, 255, 255]
        self.display_base[self.map_data == 255] = [0, 0, 0]
        self.display_base[raw_img > 200] = [220, 220, 220] # Unknown
        self.display_base[self.map_data == 0] = [255, 255, 255] # Ghi đè lại Free
        self.display_base[self.map_data == 255] = [0, 0, 0]     # Ghi đè lại Obstacle
        
        self.display_base = cv2.flip(self.display_base, 0)
        # Tạo cờ cho phép bắt sự kiện di chuyển chuột
        self.setMouseTracking(True)
        self.update_view()

    def update_view(self):
        if self.display_base is None: return
        img = self.display_base.copy()
        
        # 1. Vẽ Robot (Xanh lam)
        if self.robot_px:
            rx, ry = self.robot_px
            gui_rx, gui_ry = rx, self.h - ry - 1
            cv2.rectangle(img, (gui_rx-8, gui_ry-8), (gui_rx+8, gui_ry+8), (255, 100, 0), -1)
            cv2.putText(img, "ROBOT", (gui_rx-20, gui_ry-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

        # 2. Vẽ chùm tia 360 độ (Vàng)
        if hasattr(self, 'cone_pixels'):
            for pt in self.cone_pixels:
                gui_pt = (pt[0], self.h - pt[1] - 1)
                cv2.circle(img, gui_pt, 1, (0, 255, 255), -1) 
                
        # 2b. Vẽ 2 tia dò đầu bàn (Đỏ)
        if hasattr(self, 'end_pixels'):
            for pt in self.end_pixels:
                gui_pt = (pt[0], self.h - pt[1] - 1)
                cv2.circle(img, gui_pt, 1, (0, 0, 255), -1)
                
        # 2c. Vẽ tia trượt đỗ chéo (Cyan)
        for pt in self.ray_pixels:
            gui_pt = (pt[0], self.h - pt[1] - 1)
            cv2.circle(img, gui_pt, 1, (255, 255, 0), -1) 
            
        # 3. Vẽ Target (Đỏ)
        if self.target_px:
            tx, ty = self.target_px
            gui_tx, gui_ty = tx, self.h - ty - 1
            cv2.drawMarker(img, (gui_tx, gui_ty), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            cv2.putText(img, "TARGET", (gui_tx+10, gui_ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 4. Vẽ Goal (Xanh lá)
        if self.goal_px:
            gx, gy = self.goal_px
            gui_gx, gui_gy = gx, self.h - gy - 1
            cv2.circle(img, (gui_gx, gui_gy), 6, (0, 255, 0), -1)
            cv2.putText(img, "SMART GOAL", (gui_gx+10, gui_gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            
            # Mũi tên Yaw
            gui_yaw = -self.goal_yaw
            end_x = int(gui_gx + 30 * math.cos(gui_yaw))
            end_y = int(gui_gy + 30 * math.sin(gui_yaw))
            cv2.arrowedLine(img, (gui_gx, gui_gy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)

        qImg = QImage(img.data, self.w, self.h, self.w * 3, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qImg))

    def mousePressEvent(self, event):
        self.handle_mouse(event)
        
    def mouseMoveEvent(self, event):
        # Chỉ xử lý khi đang giữ chuột
        if event.buttons() != Qt.NoButton:
            self.handle_mouse(event)

    def handle_mouse(self, event):
        px, py = event.x(), event.y()
        map_py = self.h - py - 1 # Chuyển đổi Y từ GUI sang hệ tọa độ Map (0 ở dưới cùng)
        
        if event.buttons() & Qt.LeftButton:
            # Chuột Trái -> Đặt Khách Hàng (Target)
            self.target_px = (px, map_py)
            self.calculate_goal()
        elif event.buttons() & Qt.RightButton:
            # Chuột Phải -> Đặt Robot
            self.robot_px = (px, map_py)
            self.calculate_goal()
        elif event.buttons() & Qt.MiddleButton:
            # Chuột Giữa -> VẼ VẬT CẢN ĐỘNG (Ví dụ: Túi xách, cái ghế mọc ra giữa đường)
            if 0 <= px < self.w and 0 <= map_py < self.h:
                # Vẽ một cục vật cản bán kính 4 pixel (~20cm) lên map_data
                cv2.circle(self.map_data, (px, map_py), 4, 255, -1)
                
                # Cập nhật lại base hiển thị
                gui_py = self.h - map_py - 1
                cv2.circle(self.display_base, (px, gui_py), 4, (0, 0, 0), -1)
                
                # Tính lại ngay lập tức -> ĐIỂM TRƯỢT SẼ XUẤT HIỆN Ở ĐÂY!
                self.calculate_goal()

    def calculate_goal(self):
        self.update_view()
        if not self.robot_px or not self.target_px:
            return
            
        px_r, py_r = self.robot_px
        px_t, py_t = self.target_px
        
        # Bơm phồng vật cản (Thân xe 0.45m)
        safe_radius_px = int(0.45 / self.res)
        kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_radius_px*2+1, safe_radius_px*2+1))
        inflated_obs = cv2.dilate(self.map_data, kernel_safe)
        
        # === THUẬT TOÁN RAYCAST ĐỖ CHÉO (PLAN C) ===
        
        # Bước 1: Quét 360 độ tìm Hướng Trực Diện (Pháp tuyến - Normal)
        ray_distances = []
        max_ray_len = int(3.0 / self.res)
        self.cone_pixels = []
        
        for angle in range(0, 360, 5):
            rad = math.radians(angle)
            dist = float('inf')
            for step in range(1, max_ray_len):
                cx = int(px_t + step * math.cos(rad))
                cy = int(py_t + step * math.sin(rad))
                if not (0 <= cx < self.w and 0 <= cy < self.h):
                    break
                self.cone_pixels.append((cx, cy))
                if inflated_obs[cy, cx] == 0:
                    dist = step
                    break
            if dist != float('inf'):
                ray_distances.append((rad, dist))
                    
        if not ray_distances:
            print("[TEST] ❌ Kẹt hoàn toàn! Không có khoảng trống xung quanh khách hàng.")
            self.update_view()
            return
            
        min_dist_normal = min(d for r, d in ray_distances)
        valid_angles = [r for r, d in ray_distances if d <= min_dist_normal + 2]
        sum_x = sum(math.cos(r) for r in valid_angles)
        sum_y = sum(math.sin(r) for r in valid_angles)
        theta_normal = math.atan2(sum_y, sum_x)
            
        # Bước 2 & 3: Quét không gian để tìm bên nào thoáng hơn (Dựa vào lõi bàn)
        obs_left = 0
        obs_right = 0
        for step in range(1, int(1.5 / self.res)):
            for offset_deg in range(90, 180, 5):
                # Bên Trái
                rad_l = theta_normal + math.radians(offset_deg)
                cx_l = int(px_t + step * math.cos(rad_l))
                cy_l = int(py_t + step * math.sin(rad_l))
                if 0 <= cx_l < self.w and 0 <= cy_l < self.h and self.map_data[cy_l, cx_l] == 0: # map_data=0 is obstacle in original offline logic
                    obs_left += 1
                
                # Bên Phải
                rad_r = theta_normal - math.radians(offset_deg)
                cx_r = int(px_t + step * math.cos(rad_r))
                cy_r = int(py_t + step * math.sin(rad_r))
                if 0 <= cx_r < self.w and 0 <= cy_r < self.h and self.map_data[cy_r, cx_r] == 0:
                    obs_right += 1
                    
        if obs_left > obs_right:
            theta_dock = theta_normal - math.radians(45)
            print(f"[TEST] ↪️ Không gian PHẢI thoáng hơn (L={obs_left}, R={obs_right}). Đỗ chéo ra PHẢI (góc 45 độ).")
        else:
            theta_dock = theta_normal + math.radians(45)
            print(f"[TEST] ↪️ Không gian TRÁI thoáng hơn (L={obs_left}, R={obs_right}). Đỗ chéo ra TRÁI (góc 45 độ).")
            
        # Bước 4: Trượt điểm đỗ trên tia chéo (Mục tiêu: Cách khách 1.1m)
        target_dist_m = 1.1 
        target_step = int(target_dist_m / self.res)
        
        best_pose_px = None
        self.ray_pixels = []
        
        for step in range(1, target_step + 1):
            cx = int(px_t + step * math.cos(theta_dock))
            cy = int(py_t + step * math.sin(theta_dock))
            
            if not (0 <= cx < self.w and 0 <= cy < self.h):
                break
                
            self.ray_pixels.append((cx, cy))
            
            if inflated_obs[cy, cx] == 0:
                # Đang trong vùng an toàn, trượt điểm đỗ ra xa thêm
                best_pose_px = (cx, cy)
            else:
                # Đang trượt an toàn mà lại đụng vật cản khác (balo, tường) -> Dừng trượt!
                if best_pose_px is not None:
                    break
                    
        self.goal_px = best_pose_px
        if best_pose_px:
            self.goal_yaw = math.atan2(py_t - best_pose_px[1], px_t - best_pose_px[0])
            actual_dist = math.hypot(px_t - best_pose_px[0], py_t - best_pose_px[1])
            print(f"[TEST] ✅ Đã chốt điểm đỗ chéo, cách khách {actual_dist * self.res:.2f}m")
        else:
            print("[TEST] ❌ Bị kẹt trên tia chéo, thử nghiệm thất bại.")
            
        self.update_view()

class OfflineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OFFLINE TEST: TIA QUẠT NGƯỢC")
        
        # Đường dẫn tương đối từ tm/test_offline.py lùi ra thư mục src/
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pgm = os.path.join(base_dir, "612.png")
        yaml = os.path.join(base_dir, "612.yaml")
        
        self.map_label = MapLabelOffline(pgm, yaml)
        self.setCentralWidget(self.map_label)
        self.resize(self.map_label.w, self.map_label.h)
        
        print("\n--- HƯỚNG DẪN TEST ĐIỂM TRƯỢT (DYNAMIC GOAL) ---")
        print("1. Click CHUỘT PHẢI: Đặt vị trí Robot (Xanh lam).")
        print("2. Click CHUỘT TRÁI : Đặt Khách hàng (Đỏ) lên mặt bàn.")
        print("3. Kéo thả CHUỘT GIỮA (Nút cuộn): Vẽ các cục Balo/Vật cản vào sát mép bàn.")
        print("-> Nhìn điểm SMART GOAL tự động trượt ra xa để né cục Balo!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OfflineApp()
    window.show()
    sys.exit(app.exec_())
