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
    # 1. Đường dẫn đến file model của bạn 
    model_path = '/home/dung/mir_project/src/mir_robot/tm/best/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model tại: {model_path}")
        print("👉 Hãy copy file best.pt vào thư mục mir_project hoặc sửa lại đường dẫn model_path trong code.")
        return

    print("⏳ Đang tải mô hình YOLO tùy chỉnh (best.pt)...")
    model = YOLO(model_path)

    # 2. Khởi tạo camera laptop (ID = 0 là camera mặc định)
    print("⏳ Đang mở Camera...")
    cap = cv2.VideoCapture(0)

    # Kiểm tra camera có mở được không
    if not cap.isOpened():
        print("❌ Không thể mở camera. Vui lòng kiểm tra kết nối.")
        return

    # Cài đặt độ phân giải (Tùy chọn)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("====================================================")
    print("🚀 BẮT ĐẦU NHẬN DIỆN COCA VÀ LAVIE")
    print("👉 Nhấn phím 'q' trên cửa sổ video để thoát.")
    print("====================================================")

    # Biến để tính FPS
    prev_frame_time = 0

    while True:
        success, img = cap.read()
        if not success:
            print("❌ Không thể kết nối đến Camera!")
            break

        # Lật ảnh theo chiều ngang (mirror) để tiện quan sát như đang soi gương
        img = cv2.flip(img, 1)

        # 3. Đưa ảnh vào model để nhận diện
        # Dùng model.track để theo dõi vật thể mượt mà (chống giật lag)
        results = model.track(img, persist=True, stream=True, conf=0.40, iou=0.45, imgsz=640, verbose=False)

        # Khởi tạo biến đếm
        count_lavie = 0
        count_coca = 0
        total_products = 0

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy tọa độ
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Lọc nhiễu: Bỏ qua các khung hình quá nhỏ (diện tích < 1000 pixel)
                    if (x2 - x1) * (y2 - y1) < 1000:
                        continue

                    # Độ tự tin (Confidence)
                    conf = int(box.conf[0] * 100)
                    
                    # Lấy ID của class vừa nhận diện được
                    cls = int(box.cls[0])
                    
                    # Theo file kiensp.py: ID 0 là Coca, 1 là Lavie
                    if cls == 0 or cls == 1:
                        total_products += 1
                        if cls == 0: 
                            count_coca += 1
                            class_name = "COCA"
                            color = (0, 0, 255) # Đỏ cho coca
                        elif cls == 1: 
                            count_lavie += 1
                            class_name = "LAVIE"
                            color = (255, 0, 0) # Xanh cho lavie

                        # Vẽ khung chữ nhật (Màu đổi theo class)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                        # Hiển thị text nhãn hiệu + % tự tin lên màn hình
                        text = f'{class_name} {conf}%'
                        cv2.putText(img, text, (max(0, x1), max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Tính FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time

        # 4. Hiển thị bảng tổng kết lên góc trên bên trái màn hình
        # Tạo nền mờ (Alpha Blending) để bảng tổng kết không che khuất hoàn toàn tầm nhìn camera
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(img, f'Tong cong: {total_products} san pham', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f'Lavie: {count_lavie}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f'Coca: {count_coca}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f'FPS: {int(fps)}', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Hiển thị cửa sổ
        cv2.imshow("Nhan dien San Pham (Lavie & Coca) - Nhan 'Q' de thoat", img)

        # Bấm phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp bộ nhớ
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Đã đóng chương trình.")

if __name__ == "__main__":
    main()
