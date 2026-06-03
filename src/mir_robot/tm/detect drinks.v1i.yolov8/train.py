import os
from ultralytics import YOLO

# Lấy đường dẫn tuyệt đối của thư mục chứa file train.py
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data.yaml')

# Đổi sang model Medium (m) để đạt đỉnh cao về độ chính xác (mAP)
model = YOLO('yolov8m.pt')

# Bắt đầu train
results = model.train(
   data=data_path,
   epochs=300,       # Tăng epochs để model học kỹ hơn
   patience=50,      # Dừng sớm nếu quá 50 epoch độ chính xác không tăng
   imgsz=800,        # TĂNG LÊN 800: Giúp model nhìn rõ các vật thể (chai, cốc) nhỏ ở xa
   batch=16,          # GIẢM XUỐNG 4: Để tránh lỗi Spike Memory ở step TaskAlignedAssigner
   workers=8,
   device=0,
   cache=True,
   amp=True,
   optimizer='auto', 
   lr0=0.02,         # TĂNG: Tham số khởi tạo learning rate ngay từ đầu (mặc định là 0.01)
   val=True,         
   cos_lr=True,      # BẬT: Cosine Annealing giúp tối ưu learning rate mượt hơn
   mosaic=1.0,       # Bật 100% tỷ lệ ghép 4 ảnh
   mixup=0.15,       # BẬT: Trộn đè các ảnh lên nhau với tỷ lệ 15% để model học khó hơn
   close_mosaic=10,  # Tắt các hiệu ứng ảo ở 10 epochs cuối để hội tụ với ảnh thực tế
   
   # Các tham số chống Overfitting & Tăng cường dữ liệu (Augmentation)
   dropout=0.2,      # Thêm dropout 20% giảm overfitting
   weight_decay=0.0005, 
   degrees=10.0,     # Xoay ảnh ngẫu nhiên +-10 độ
   translate=0.1,    # Dịch chuyển ảnh
   scale=0.5,        # Phóng to/thu nhỏ
   hsv_h=0.015,      # Đổi màu sắc (Hue)
   hsv_s=0.7,        # Đổi độ bão hòa (Saturation)
   hsv_v=0.4         # Đổi độ sáng (Value)
)