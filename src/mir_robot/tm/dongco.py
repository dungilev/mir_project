#!/usr/bin/env python3
import time
import math

class ServoController:
    def __init__(self, pin=18, min_angle=0, max_angle=180, min_pulse=2.5, max_pulse=12.5):
        """
        Khởi tạo điều khiển Servo 35kg (thường sử dụng PWM 50Hz).
        * Lưu ý: Bạn cần chạy lệnh `sudo pip3 install RPi.GPIO` để cài đặt thư viện nếu dùng Raspberry Pi/Jetson.
        
        :param pin: Chân GPIO (BCM) kết nối với dây tín hiệu (màu cam/vàng) của Servo
        :param min_angle: Góc quay nhỏ nhất (ví dụ: 0 độ)
        :param max_angle: Góc quay lớn nhất (ví dụ: 180 hoặc 270 độ tùy loại Servo 35kg)
        :param min_pulse: Duty cycle tương ứng góc nhỏ nhất (thường từ 2% - 2.5%)
        :param max_pulse: Duty cycle tương ứng góc lớn nhất (thường từ 10% - 12.5%)
        """
        self.pin = pin
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            self.GPIO.setup(self.pin, self.GPIO.OUT)
            # Tần số 50Hz (chu kỳ 20ms) - chuẩn cho servo
            self.pwm = self.GPIO.PWM(self.pin, 50)
            self.pwm.start(0)
            print(f"[ServoController] Đã khởi tạo servo tại chân GPIO {self.pin}")
        except ImportError:
            print("[Cảnh báo] Không tìm thấy thư viện RPi.GPIO. Chế độ mô phỏng (Simulation) được bật.")
            self.GPIO = None
            self.pwm = None

    def set_angle(self, angle):
        """
        Quay servo tới một góc cụ thể.
        """
        if angle < self.min_angle:
            angle = self.min_angle
        elif angle > self.max_angle:
            angle = self.max_angle

        # Tính toán duty cycle dựa trên góc quay
        duty_cycle = self.min_pulse + (float(angle - self.min_angle) / (self.max_angle - self.min_angle)) * (self.max_pulse - self.min_pulse)
        
        if self.pwm:
            self.pwm.ChangeDutyCycle(duty_cycle)
            # Chờ một khoảng thời gian nhỏ để servo tới vị trí rồi tắt xung cho đỡ rung
            time.sleep(0.3)
            self.pwm.ChangeDutyCycle(0) 
        else:
            print(f"[Simulation] Servo quay tới {angle} độ (Duty Cycle: {duty_cycle:.2f}%)")

    def scan(self, start_angle=0, end_angle=180, step=10, delay=0.5):
        """
        Chế độ quét (scan) dùng gắn camera xoay qua lại.
        Giúp quét tìm người và tránh vật cản hoặc thân MIR.
        """
        print(f"Bắt đầu quét từ {start_angle} đến {end_angle} độ...")
        # Lượt đi
        for angle in range(start_angle, end_angle + 1, step):
            self.set_angle(angle)
            time.sleep(delay)
            
        # Lượt về
        for angle in range(end_angle, start_angle - 1, -step):
            self.set_angle(angle)
            time.sleep(delay)

    def cleanup(self):
        """
        Giải phóng chân GPIO khi kết thúc chương trình.
        """
        if self.pwm:
            self.pwm.stop()
            self.GPIO.cleanup()
            print("[ServoController] Đã thu hồi chân GPIO.")

if __name__ == "__main__":
    # Ví dụ sử dụng:
    # Điều chỉnh chân pin (18) phù hợp với sơ đồ đấu nối của bạn
    # Servo 35kg thường là loại 180 độ hoặc 270 độ
    servo = ServoController(pin=18, min_angle=0, max_angle=180)
    
    try:
        # Căn chỉnh camera nhìn thẳng (90 độ)
        servo.set_angle(90)
        time.sleep(1)
        
        print("Bắt đầu quét liên tục để nhận diện...")
        while True:
            # Quét camera từ 30 độ đến 150 độ, mỗi bước 15 độ
            servo.scan(start_angle=30, end_angle=150, step=15, delay=0.2)
            
    except KeyboardInterrupt:
        print("Đã dừng thủ công.")
    finally:
        servo.cleanup()
