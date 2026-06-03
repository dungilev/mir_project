#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import time
import cv2
import numpy as np
import threading
from PIL import Image, ImageTk

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[Cảnh báo] Không tìm thấy thư viện pyrealsense2")

class ServoCameraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Camera 3D & Servo (RealSense)")
        self.root.geometry("1000x600")
        
        self.serial_port = None
        self.pipeline = None
        self.camera_running = False
        
        # --- BỘ PHẬN ĐIỀU KHIỂN BÊN TRÁI (SERVO) ---
        frame_control = tk.Frame(root, width=350)
        frame_control.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(frame_control, text="⚙️ ĐIỀU KHIỂN SERVO", font=("Arial", 14, "bold")).pack(pady=10)
        
        frame_port = tk.Frame(frame_control)
        frame_port.pack(pady=5)
        tk.Label(frame_port, text="Cổng USB:").pack(side=tk.LEFT)
        self.port_combobox = ttk.Combobox(frame_port, values=self.get_available_ports(), width=15)
        self.port_combobox.pack(side=tk.LEFT, padx=5)
        if self.port_combobox['values']:
            self.port_combobox.current(0) # Tự động chọn cổng đầu tiên tìm thấy
        self.btn_connect = tk.Button(frame_port, text="Kết nối", command=self.connect_serial, bg="lightblue")
        self.btn_connect.pack(side=tk.LEFT)
        
        self.lbl_status = tk.Label(frame_control, text="Chưa kết nối Arduino", fg="red")
        self.lbl_status.pack(pady=5)
        
        tk.Label(frame_control, text="Góc quay:").pack(pady=(15, 0))
        self.scale_angle = tk.Scale(frame_control, from_=0, to=180, orient=tk.HORIZONTAL, command=self.on_slide, length=300)
        self.scale_angle.set(90)
        self.scale_angle.pack(pady=5)
        
        frame_btns = tk.Frame(frame_control)
        frame_btns.pack(pady=5)
        tk.Button(frame_btns, text="Trái (0°)", command=lambda: self.set_angle(0)).pack(side=tk.LEFT)
        tk.Button(frame_btns, text="Giữa (90°)", command=lambda: self.set_angle(90)).pack(side=tk.LEFT)
        tk.Button(frame_btns, text="Phải (180°)", command=lambda: self.set_angle(180)).pack(side=tk.LEFT)

        self.btn_scan = tk.Button(frame_control, text="Chạy Chế Độ Quét", command=self.auto_scan, bg="orange")
        self.btn_scan.pack(pady=15)
        
        # --- BỘ PHẬN CAMERA BÊN PHẢI ---
        frame_cam = tk.Frame(root, bg="black")
        frame_cam.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        tk.Label(frame_cam, text="📷 CAMERA RealSense 3D", font=("Arial", 14, "bold"), bg="black", fg="white").pack(pady=5)
        
        self.lbl_video = tk.Label(frame_cam, bg="grey", text="Đang chờ camera...", fg="white")
        self.lbl_video.pack(expand=True, fill=tk.BOTH)
        
        self.btn_start_cam = tk.Button(frame_cam, text="Mở Camera", command=self.toggle_camera, bg="green", fg="white")
        self.btn_start_cam.pack(pady=5)

    def get_available_ports(self):
        # Lọc ra các cổng thực sự thường dùng cho Arduino (loại bỏ ttyS ảo)
        ports = [p.device for p in serial.tools.list_ports.comports() if "ttyACM" in p.device or "ttyUSB" in p.device]
        return ports if ports else ["Không tìm thấy Arduino"]

    def connect_serial(self):
        port = self.port_combobox.get()
        if not port: return
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                self.btn_connect.config(text="Kết nối")
                self.lbl_status.config(text="Đã ngắt kết nối", fg="red")
            else:
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                time.sleep(2) 
                self.btn_connect.config(text="Ngắt kết nối")
                self.lbl_status.config(text=f"Đã kết nối: {port}", fg="green")
                self.set_angle(90)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def send_to_arduino(self, angle):
        if self.serial_port and self.serial_port.is_open:
            try:
                # Đảm bảo ép kiểu và chỉ gửi dữ liệu dạng nguyên
                val = int(float(angle))
                self.serial_port.write(f"{val}\n".encode('utf-8'))
            except Exception as e:
                print(f"Lỗi gửi lệnh: {e}")

    def on_slide(self, event):
        # Tránh spam lệnh liên tục xuống Arduino khi kéo thanh trượt
        if not hasattr(self, "_last_send"):
            self._last_send = 0
        now = time.time()
        if (now - self._last_send) > 0.05:  # Tối đa gửi 20 lần/giây
            self.send_to_arduino(self.scale_angle.get())
            self._last_send = now
        
    def set_angle(self, angle):
        self.scale_angle.set(angle)
        # self.scale_angle.set đã tự động gọi on_slide, nên không cần gọi send_to_arduino lần 2 dễ kẹt buffer

    def auto_scan(self):
        if not self.serial_port: return
        def run():
            for a in range(30, 151, 5):
                self.set_angle(a); time.sleep(0.05)
            for a in range(150, 29, -5):
                self.set_angle(a); time.sleep(0.05)
            self.set_angle(90)
        threading.Thread(target=run, daemon=True).start()

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if not REALSENSE_AVAILABLE: return
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            # Bật luồng màu tiêu chuẩn BGR 640x480 để tránh lệch hình
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.camera_running = True
            self.btn_start_cam.config(text="Tắt Camera", bg="red")
            self.update_frame()
        except Exception as e:
            print("Camera Error:", e)
            self.stop_camera()

    def stop_camera(self):
        self.camera_running = False
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.btn_start_cam.config(text="Mở Camera", bg="green")
        self.lbl_video.config(image='', text="Đóng")

    def update_frame(self):
        if not self.camera_running: return
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                # RGB format correction từ BGR của RealSense sang RGB cho Tkinter
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(color_image_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.lbl_video.img_tk = img_tk  
                self.lbl_video.config(image=img_tk, text="")
        except Exception as e:
            pass
            
        if self.camera_running:
            self.root.after(30, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ServoCameraGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()
