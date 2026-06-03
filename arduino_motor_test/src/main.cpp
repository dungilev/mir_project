#include <Arduino.h>
#include <Servo.h>

Servo myServo;
const int servoPin = 9; // Kết nối dây tín hiệu Servo vào chân 9 (bạn có thể đổi nếu cắm chân khác)

void setup() {
  // Khởi tạo giao tiếp UART với Baudrate 115200 (phải khớp với Python)
  Serial.begin(115200);
  
  // Gắn servo vào chân
  myServo.attach(servoPin);
  
  // Trạng thái ban đầu: 90 độ (giữa)
  myServo.write(90);
  
  Serial.println("--- Hệ thống Servo đã sẵn sàng ---");
}

void loop() {
  // Kiểm tra xem có dữ liệu gửi từ máy tính (Python GUI) xuống không
  if (Serial.available() > 0) {
    // Đọc một dòng ký tự cho đến khi gặp kí tự '\n'
    String input = Serial.readStringUntil('\n');
    input.trim(); // Loại bỏ khoảng trắng thừa
    
    if (input.length() > 0) {
      int angle = input.toInt(); // Chuyển chuỗi thành số nguyên
      
      // Giới hạn góc quay từ 0 đến 180 độ
      if (angle >= 0 && angle <= 180) {
        myServo.write(angle);
        
        // Phản hồi lại để hiển thị nếu cần (Python ko nhất thiết phải đọc)
        Serial.print("Servo moved to: ");
        Serial.println(angle);
      }
    }
  }
}
