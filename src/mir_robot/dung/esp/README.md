# ESP32 Table Button -> ROSBridge

Firmware này giúp ESP32 gửi tín hiệu nút bấm đến ROS qua `rosbridge_websocket` (`ws://<ip>:9090`) và publish lên topic `/table_call_buttons` (kiểu `std_msgs/String`).

## 1) Phần cứng

- ESP32
- Dùng nút BOOT có sẵn trên board (mặc định)
- (Tuỳ chọn) 1 nút bấm cơ ngoài

Đấu dây:
- Chân 1 nút -> `GND`
- Chân 2 nút -> `GPIO 0` (mặc định là nút BOOT trên ESP32; có thể đổi `BUTTON_PIN` trong code)

Code dùng `INPUT_PULLUP`, không cần điện trở ngoài.

## 2) Dùng PlatformIO (khuyến nghị)

Thư mục này đã có cấu trúc PlatformIO chuẩn:
- `platformio.ini`
- `src/main.cpp`

Trong VS Code:
- Mở đúng thư mục project `src/mir_robot/dung/esp`
- Dùng PlatformIO: `Build` / `Upload` / `Monitor`

Library được khai báo sẵn trong `platformio.ini`:
- `links2004/WebSockets`
- `bblanchon/ArduinoJson`

Lưu ý: project đã dùng duy nhất `src/main.cpp` để tránh lỗi IntelliSense khi mở `.ino`.

## 4) Cấu hình trước khi nạp

Mở `src/main.cpp` và sửa:
- `WIFI_SSID`
- `WIFI_PASSWORD`
- `ROSBRIDGE_HOST` (IP máy chạy ROSBridge, ví dụ `192.168.0.177` hoặc IP máy Docker host)
- `TABLE_NO` (bàn số mấy)
- `BUTTON_PIN` nếu đổi chân

## 5) Payload gửi lên ROS

ESP32 gửi frame rosbridge dạng:
- `op=publish`
- `topic=/table_call_buttons`
- `msg.data` là chuỗi JSON, ví dụ:

`{"ban":1,"event":"button_pressed","source":"esp32","seq":12}`

`order_listener.py` hiện đã parse trường `ban` để điều hướng robot đến đúng bàn.

## 6) Test nhanh trên ROS

Chạy listener và xem log:
- `rosrun tm order_listener.py` (hoặc lệnh launch bạn đang dùng)
- `rostopic echo /table_call_buttons`

Bấm nút trên ESP32, bạn sẽ thấy message và robot nhận lệnh đi bàn tương ứng.

## 7) Lưu ý vận hành

- Có `debounce` (`DEBOUNCE_MS=40`) chống dội phím.
- Có `PRESS_COOLDOWN_MS=2500` chống gửi lặp khi bấm liên tục.
- Tự reconnect Wi-Fi và WebSocket.
- Nếu triển khai nhiều bàn: mỗi ESP32 đặt `TABLE_NO` khác nhau.
