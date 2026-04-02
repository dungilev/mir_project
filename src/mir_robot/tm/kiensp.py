import os
import time

# ==============================================================================
# TỬ HUYỆT: SỬA LỖI TREO KHI IMPORT YOLO (TỪ LỊCH SỬ LỖI CỦA LORDKIEN)
# Dòng này phải đặt TRƯỚC khi import YOLO. Nó ép YOLO chạy offline, không lên mạng check update.
# ==============================================================================
os.environ['YOLO_OFFLINE'] = 'True' 

try:
    import cv2
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ Thiếu thư viện: {e}")
    print("👉 Hãy chắc chắn bạn đã activate .venv và cài đặt 'ultralytics' và 'opencv-python'")
    exit(1)

def main():
    # --- CẤU HÌNH ---
    # Đường dẫn tới file model bạn vừa train xong
    model_path = '/home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model tại: {model_path}")
        print("👉 Hãy copy file best.pt vào thư mục mir_project hoặc sửa lại đường dẫn model_path trong code.")
        return

    print("⏳ Đang tải mô hình YOLO tùy chỉnh (best.pt)...")
    model = YOLO(model_path) 

    # Khởi tạo camera (thường là số 0 cho webcam)
    print("⏳ Đang mở Camera...")
    cap = cv2.VideoCapture(0)

    # Kiểm tra camera có mở được không
    if not cap.isOpened():
        print("❌ Không thể mở camera. Vui lòng kiểm tra kết nối.")
        return

    print("====================================================")
    print("🚀 BẮT ĐẦU NHẬN DIỆN COCA VÀ LAVIE")
    print("👉 Nhấn phím 'q' trên cửa sổ video để thoát.")
    print("====================================================")

    # Biến để tính FPS
    prev_frame_time = 0

    while True:
        # 1. Đọc khung hình từ camera
        ret, frame = cap.read()
        if not ret:
            print("❌ Lỗi đọc khung hình từ camera.")
            break

        # 2. CHẠY AI NHẬN DIỆN
        # Bỏ đi tham số classes vì model mới chỉ có classes 0 và 1 của bạn
        results = model.track(frame, conf=0.5, persist=True, verbose=False)

        # 3. ĐẾM SỐ LƯỢNG TỪNG LOẠI
        coca_count = 0
        lavie_count = 0
        
        detected_boxes = results[0].boxes
        if detected_boxes is not None:
            for box in detected_boxes:
                # Lấy ID của class vừa nhận diện được
                class_id = int(box.cls[0]) 
                # 0: coca, 1: lavie (như bạn đã cấu hình lúc train)
                if class_id == 0:
                    coca_count += 1
                elif class_id == 1:
                    lavie_count += 1

        # 4. VẼ LÊN MÀN HÌNH
        annotated_frame = results[0].plot()

        # Tính FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time

        # Vẽ tổng số chai hiển thị góc màn hình
        cv2.rectangle(annotated_frame, (10, 10), (450, 150), (0,0,0), -1)
        
        cv2.putText(annotated_frame, f"Coca: {coca_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Lavie: {lavie_count}", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # 5. Hiển thị khung hình ra cửa sổ
        cv2.imshow("Robot AI - Nhan dien Coca/Lavie", annotated_frame)

        # 6. Thoát khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Đã đóng chương trình.")

if __name__ == "__main__":
    main()