import cv2
from ultralytics import YOLO
import time
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ==============================================================================
# Bật chế độ YOLO Offline chống lag mạng
os.environ['YOLO_OFFLINE'] = 'True' 
# ==============================================================================

model_path = '/home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt'
print("⏳ Đang tải AI Model (YOLO) lên TẬN DỤNG SỨC MẠNH GPU RTX 5060 HOST...")
model = YOLO(model_path)

class AI_RequestHandler(BaseHTTPRequestHandler):
    # Tắt log hệ thống mặc định của http server để cmd đỡ rối
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/check_items':
            query = parse_qs(parsed_path.query)
            expected_coca = int(query.get('coca', ['0'])[0])
            expected_lavie = int(query.get('lavie', ['0'])[0])
            
            if expected_coca <= 0 and expected_lavie <= 0:
                expected_lavie = 1  
                print("\n⚠️ Đơn trống! [Chế độ Demo]: Ép AI nhận ít nhất 1 chai bất kỳ để pass.")
                
            print(f"\n📸 [AI Server] Nhận yêu cầu từ MiR (Docker): Cần {expected_coca} Coca, {expected_lavie} Lavie. Mở Camera 30s...")
            cap = cv2.VideoCapture(0)
            success = False
            message = "Server error"
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                start_time = time.time()
                success_frames = 0
                
                while time.time() - start_time < 30.0:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame = cv2.flip(frame, 1)
                    
                    # AI Chạy 100% bằng kiến trúc sm_120 trên Host, không còn bị kẹt CPU Docker!
                    results = model.track(frame, persist=True, stream=True, conf=0.40, iou=0.45, imgsz=640, verbose=False)
                    count_coca = count_lavie = 0
                    
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                if (x2 - x1) * (y2 - y1) < 1000: continue
                                cls = int(box.cls[0])
                                if cls == 0: count_coca += 1
                                elif cls == 1: count_lavie += 1
                                
                    print(f"\r[AI Tracking] Trên mâm: {count_coca} Coca, {count_lavie} Lavie | Tiêu chuẩn: {expected_coca} Coca, {expected_lavie} Lavie  ", end="")
                    
                    if count_coca >= expected_coca and count_lavie >= expected_lavie:
                        success_frames += 1
                    else:
                        success_frames = 0
                        
                    if success_frames >= 5:
                        print("\n✅ [AI Server] ĐÃ ĐỦ ĐỒ! Trả tín hiệu chạy cho Robot.")
                        success = True
                        message = "Đủ đồ"
                        break
                
                if not success:
                    print("\n❌ [AI Server] Hết 30s. VẪN THIẾU ĐỒ! Báo lỗi về Robot.")
                    message = "Thiếu đồ hoặc hết 30s"
                    
                cap.release()
            else:
                print("❌ [AI Server] Lỗi phần cứng: Lệnh mở Camera thất bại!")
                message = "Không mở được camera máy Host"
                
            self.send_response(200 if success else 400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": success, "message": message}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    port = 5050
    server = HTTPServer(('0.0.0.0', port), AI_RequestHandler)
    print(f"🚀 Bật AI Server trên Host tại Port {port} ... Chờ kết nối từ Docker MiR!")
    server.serve_forever()