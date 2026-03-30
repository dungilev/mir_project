import rospy
from sensor_msgs.msg import Joy
import tkinter as tk
import threading
import signal

# Chặn SIGINT lan sang các process khác (joy_node, roscore)
signal.signal(signal.SIGINT, lambda s, f: rospy.signal_shutdown("User pressed Ctrl+C"))

# Biến lưu trữ tin nhắn Joy mới nhất (thread-safe)
latest_msg = None
msg_lock = threading.Lock()

def joy_callback(msg):
    """Callback chạy trên thread của ROS - chỉ lưu dữ liệu, KHÔNG cập nhật GUI."""
    global latest_msg
    with msg_lock:
        latest_msg = msg

def process_joy_msg():
    """Đọc tin nhắn Joy mới nhất và cập nhật GUI (chạy trên main thread)."""
    global latest_msg
    with msg_lock:
        msg = latest_msg
        latest_msg = None

    if msg is not None:
        # Đọc tín hiệu nút R1 (R1 là index số 5) - Deadman Switch
        r1_pressed = msg.buttons[5] == 1

        if r1_pressed:
            # R1 đang giữ -> Joystick HOẠT ĐỘNG
            x = msg.axes[0] * -100
            y = msg.axes[1] * -100
            canvas.coords(stick, 150 + x - 10, 150 + y - 10, 150 + x + 10, 150 + y + 10)
            canvas.itemconfig(stick, fill="#2ecc71", outline="#27ae60")  # Xanh lá = hoạt động
            lbl_r1.config(bg="#27ae60", text="R1: ĐANG GIỮ ✓ (Joystick ON)")
        else:
            # R1 nhả -> Joystick BỊ KHÓA, chấm về tâm
            canvas.coords(stick, 140, 140, 160, 160)  # Reset về giữa
            canvas.itemconfig(stick, fill="#e74c3c", outline="#c0392b")  # Đỏ = khóa
            lbl_r1.config(bg="#c0392b", text="R1: NHẢ ✗ (Joystick KHÓA)")

# 1. Khởi tạo Node ROS ẩn danh (disable_signals=True để ROS không can thiệp SIGINT)
rospy.init_node('joystick_gui', anonymous=True, disable_signals=True)

# 2. Khởi tạo Giao diện Tkinter
root = tk.Tk()
root.title("Trạm điều khiển MiR - Tuan Minh")
root.geometry("350x450")
root.configure(bg="#2c3e50")

# Tiêu đề
title = tk.Label(root, text="JOYSTICK DASHBOARD", fg="white", bg="#2c3e50", font=("Helvetica", 16, "bold"))
title.pack(pady=10)

# Khung vẽ Radar
canvas = tk.Canvas(root, width=300, height=300, bg="#ecf0f1", highlightthickness=0)
canvas.pack(pady=10)

# Vẽ tâm chữ thập và vòng giới hạn
canvas.create_oval(50, 50, 250, 250, outline="#bdc3c7", width=2)
canvas.create_line(150, 50, 150, 250, fill="#bdc3c7", dash=(4, 4))
canvas.create_line(50, 150, 250, 150, fill="#bdc3c7", dash=(4, 4))

# Vẽ chấm đỏ đại diện cho cần gạt
stick = canvas.create_oval(140, 140, 160, 160, fill="#e74c3c", outline="#c0392b")

# Đèn báo nút bấm
lbl_r1 = tk.Label(root, text="Nút R1: NHẢ (Khóa)", bg="red", fg="white", font=("Arial", 12, "bold"), width=25, pady=5)
lbl_r1.pack()

# 3. Lắng nghe topic /joy
rospy.Subscriber("/joy", Joy, joy_callback)

# 4. Vòng lặp song song (Vừa giữ giao diện sống, vừa nghe ROS)
try:
    while not rospy.is_shutdown():
        process_joy_msg()  # Cập nhật GUI an toàn trên main thread
        root.update()
        rospy.sleep(0.02)  # Tốc độ làm mới khung hình (50fps)
except (tk.TclError, KeyboardInterrupt):
    pass  # Bỏ qua lỗi khi nhấn dấu X hoặc Ctrl+C
finally:
    try:
        root.destroy()
    except tk.TclError:
        pass
    rospy.signal_shutdown("GUI closed")
