#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
# Ép YOLO chạy offline để tránh bị treo khi kiểm tra update
os.environ['YOLO_OFFLINE'] = 'True'

import time
import math
import numpy as np
import requests
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
import navigationcacdiem as nav



# Đăng nhập REST API tự động qua file navigationcacdiem (sử dụng IP 192.168.0.177 và mật khẩu SHA-256)
HEADERS = nav.api_login()
BASE_URL = nav.API_URL

def get_register(reg_id):
    """Đọc giá trị của một thanh ghi PLC từ MiR"""
    if not HEADERS:
        print("[ERR] Chưa đăng nhập được REST API!")
        return None
    try:
        url = f"{BASE_URL}/registers/{reg_id}"
        response = requests.get(url, headers=HEADERS, timeout=2)
        if response.status_code == 200:
            return response.json().get("value")
        else:
            print(f"[ERR] Lỗi HTTP {response.status_code} khi đọc thanh ghi {reg_id}: {response.text}")
    except Exception as e:
        print(f"Lỗi kết nối khi đọc thanh ghi {reg_id}: {e}")
    return None

def set_register(reg_id, value):
    """Ghi giá trị vào một thanh ghi PLC trên MiR"""
    if not HEADERS:
        print("[ERR] Chưa đăng nhập được REST API!")
        return False
    try:
        url = f"{BASE_URL}/registers/{reg_id}"
        data = {"value": value}
        response = requests.put(url, headers=HEADERS, json=data, timeout=2)
        if response.status_code == 200:
            return True
        else:
            print(f"[ERR] Lỗi HTTP {response.status_code} khi ghi thanh ghi {reg_id}: {response.text}")
    except Exception as e:
        print(f"Lỗi kết nối khi ghi thanh ghi {reg_id}: {e}")
    return False

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

def run_visual_servoing(is_test=False):
    """Kích hoạt RealSense + YOLO, tính toán góc và khoảng cách đến người, ghi vào Reg 102/103 để MiR làm Relative Move"""
    print("[ACTIVE] Kích hoạt Camera 3D và YOLO. Đang dò tìm vị trí khách hàng...")
    if is_test:
        print("👉 ĐANG CHẠY CHẾ ĐỘ TEST (Camera mở liên tục). Nhấn 'q' trên cửa sổ camera để thoát.")
    
    # 1. Khởi tạo Camera RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        depth_profile = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()
        print("[INFO] Đã kết nối RealSense thành công.")
    except Exception as e:
        print(f"[ERR] Không thể kết nối RealSense: {e}")
        if not is_test:
            set_register(102, 0.0)
            set_register(103, 0.0)
        return False, 0.0, 0.0


    # 2. Tải model YOLO
    try:
        model = YOLO("/home/tuanminh/mir_project/yolo11n.pt")
    except Exception as e:
        print(f"[ERR] Không thể tải model YOLO: {e}")
        pipeline.stop()
        if not is_test:
            set_register(102, 0.0)
            set_register(103, 0.0)
        return False, 0.0, 0.0


    start_time = time.time()
    detected = False
    d_move = 0.0
    theta_deg = 0.0

    # Tạo cửa sổ hiển thị
    cv2.namedWindow("RealSense + YOLO Tracking", cv2.WINDOW_AUTOSIZE)

    # Quét vô hạn ở chế độ test, hoặc tối đa 8 giây ở chế độ thường
    while is_test or (time.time() - start_time < 8.0):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except:
            continue

        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        results = model.track(frame, classes=[0], conf=0.45, verbose=False)
        
        # Vẽ các kết quả nhận diện lên frame
        annotated_frame = results[0].plot() if (results and results[0].boxes) else frame.copy()
        
        if results and results[0].boxes:
            # Lấy đối tượng người đầu tiên phát hiện được
            box = results[0].boxes.xyxy.cpu().numpy()[0]
            
            # Tính toán khoảng cách
            distance_m = get_depth_distance_m(depth_frame, box, 640, 480)
            if distance_m > 0:
                # Bù trừ độ cao camera để lấy khoảng cách mặt đất
                d_ngang_m = math.sqrt(distance_m**2 - 1.2**2) if distance_m > 1.2 else distance_m
                
                rel = get_person_relative_position_m(box, 640, depth_intrinsics, d_ngang_m)
                if rel:
                    rel_Z, rel_X = rel[0], rel[1]
                    
                    # Tính toán tọa độ tương đối đối với tâm quay robot base_link
                    x_rel = rel_Z - 0.475
                    y_rel = rel_X
                    
                    D = math.sqrt(x_rel**2 + y_rel**2)
                    theta_rad = math.atan2(y_rel, x_rel)
                    theta_deg = math.degrees(theta_rad)
                    
                    # Khoảng cách cần đi để đỗ cách khách đúng 0.5m
                    d_move = D - 0.5
                    
                    # Các giới hạn an toàn để tránh robot di chuyển bất thường
                    d_move = max(-1.0, min(2.0, d_move))       # Giới hạn lùi tối đa 1m, tiến tối đa 2m
                    theta_deg = max(-90.0, min(90.0, theta_deg)) # Giới hạn góc quay tối đa 90 độ
                    
                    print(f"[FOUND] Tìm thấy khách hàng:")
                    print(f"        - Khoảng cách thực tế: {D:.2f}m")
                    print(f"        - Góc lệch: {theta_deg:.1f} độ")
                    print(f"        - Khoảng cách cần dịch chuyển: {d_move:.2f}m")
                    
                    # Vẽ thông tin tính toán lên màn hình preview
                    cv2.putText(annotated_frame, f"Dist: {D:.2f}m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(annotated_frame, f"Angle: {theta_deg:.1f} deg", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(annotated_frame, f"Move: {d_move:.2f}m", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(annotated_frame, "Target Locked! Moving...", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    if not is_test:
                        cv2.imshow("RealSense + YOLO Tracking", annotated_frame)
                        cv2.waitKey(1500)  # Hiển thị kết quả tĩnh trong 1.5 giây để kiểm tra trực quan
                        detected = True
                        break
                    else:
                        # Ở chế độ test, in giá trị ra màn hình để xem liên tục
                        cv2.putText(annotated_frame, "TEST MODE - Press 'q' to Quit", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("RealSense + YOLO Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Đã nhấn 'q' để thoát quét camera.")
            break

    if not is_test and not detected:
        # Nếu hết 8 giây không tìm thấy ai, hiển thị thông báo lỗi
        frame_err = frame.copy() if 'frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame_err, "No Customer Detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("RealSense + YOLO Tracking", frame_err)
        cv2.waitKey(1500)

    cv2.destroyAllWindows()
    pipeline.stop()

    if is_test:
        return True, 0.0, 0.0

    if detected:
        # Ghi các tham số tính toán được vào thanh ghi 102 và 103
        set_register(102, round(d_move, 3))
        set_register(103, round(theta_deg, 1))
        print(f"[SUCCESS] Đã ghi tham số di chuyển: Reg 102 = {d_move:.3f}m, Reg 103 = {theta_deg:.1f} deg.")
        return True, d_move, theta_deg
    else:
        print("[WARN] Không nhận diện được khách hàng. Đặt giá trị mặc định là 0 để an toàn.")
        set_register(102, 0.0)
        set_register(103, 0.0)
        return False, 0.0, 0.0


def find_mission_guid_by_name(mission_name):
    """Tìm GUID của một Mission dựa vào tên trên robot"""
    if not HEADERS:
        return None
    try:
        url = f"{BASE_URL}/missions"
        response = requests.get(url, headers=HEADERS, timeout=5)
        if response.status_code == 200:
            for m in response.json():
                if m.get("name") == mission_name:
                    return m.get("guid")
    except Exception as e:
        print(f"[ERR] Lỗi khi tìm kiếm mission '{mission_name}': {e}")
    return None

def main():
    is_test_mode = "--test" in sys.argv or "-t" in sys.argv
    
    if is_test_mode:
        run_visual_servoing(is_test=True)
        return

    if not HEADERS:
        print("[CRITICAL] Không thể kết nối hoặc xác thực với MiR API. Vui lòng kiểm tra IP và Wifi!")
        print("💡 GỢI Ý: Bạn có thể chạy lệnh 'python3 dc.py --test' để kiểm tra riêng camera và YOLO không cần robot.")
        return
        
    print("[START] Hệ thống Python giám sát bắt tay đã khởi động. Đang đợi MiR...")
    
    # Reset các thanh ghi về 0 trước khi chạy để đảm bảo an toàn
    set_register(100, 0)
    set_register(101, 0)

    while True:
        # 1. Đọc liên tục thanh ghi 100 để xem MiR đã đến điểm chờ chưa
        reg_100 = get_register(100)
        
        if reg_100 == 1:
            print("\n[EVENT] MiR đã đến Điểm Chờ. Bắt đầu tính toán góc và khoảng cách đỗ...")
            
            # 2. Chạy thuật toán định vị người và ghi tham số vào Reg 102/103
            success, d_move, theta_deg = run_visual_servoing()
            
            # 3. Xóa cờ báo của MiR đi (đưa về 0)
            set_register(100, 0)
            
            # 4. Nếu thành công, tự động tìm và Queue "relativemmove" hoặc "Relative Move Mission" qua REST API
            if success:
                mission_name = "relativemmove"
                mission_guid = find_mission_guid_by_name(mission_name)
                param_x = "MOVE_X"
                param_yaw = "MOVE_YAW"
                
                if not mission_guid:
                    mission_name = "Relative Move Mission"
                    mission_guid = find_mission_guid_by_name(mission_name)
                    param_x = "move_x"
                    param_yaw = "move_yaw"

                if mission_guid:
                    print(f"[INFO] Tìm thấy mission '{mission_name}' trên robot (GUID: {mission_guid}). Đang queue di chuyển...")
                    payload = {
                        "mission_id": mission_guid,
                        "parameters": [
                            {"input_name": param_x, "value": round(d_move, 3)},
                            {"input_name": param_yaw, "value": round(theta_deg, 1)}
                        ]
                    }
                    try:
                        url = f"{BASE_URL}/mission_queue"
                        response = requests.post(url, headers=HEADERS, json=payload, timeout=5)
                        if response.status_code in (200, 201):
                            print(f"[SUCCESS] Đã gửi lệnh Relative Move thành công: X={d_move:.3f}m, Yaw={theta_deg:.1f}deg")
                        else:
                            print(f"[ERR] Không thể queue mission '{mission_name}': HTTP {response.status_code} - {response.text}")
                    except Exception as e:
                        print(f"[ERR] Lỗi kết nối khi gửi lệnh di chuyển tương đối: {e}")
                else:
                    print("[WARN] Không tìm thấy mission 'relativemmove' hoặc 'Relative Move Mission' trên Web UI. Chạy theo cơ chế Register mặc định.")

            
            # 5. Ghi số 1 vào thanh ghi 101 để báo cho Web Mission hoàn thành
            print("[EVENT] Gửi tín hiệu hoàn thành (Set Register 101 = 1) để giải phóng trạng thái chờ.")
            set_register(101, 1)
            
            # Đợi một chút để hệ thống cập nhật, tránh vòng lặp bị lặp lại ngay lập tức
            time.sleep(3)
            set_register(101, 0) # Reset lại về 0 chuẩn bị cho bàn tiếp theo
            
        time.sleep(0.5) # Chu kỳ quét thanh ghi 500ms một lần để tránh làm nghẽn CPU/Mạng


if __name__ == "__main__":
    main()