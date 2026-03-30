#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_srvs.srv import Empty  # Thư viện để gọi dịch vụ xóa costmap

# --- KHAI BÁO BIẾN TOÀN CỤC ---
move_base_client = None

# --- HÀM KHỞI TẠO KẾT NỐI ---
def init_client():
    global move_base_client
    rospy.loginfo("⏳ Đang kết nối tới move_base...")
    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    
    # Chờ server move_base tối đa 10 giây, nếu không thấy thì báo lỗi
    connected = move_base_client.wait_for_server(rospy.Duration(10))
    if not connected:
        rospy.logerr("❌ Không thể kết nối với move_base! Kiểm tra lại xem Navigation đã bật chưa?")
        exit()
        
    rospy.loginfo("✅ Đã kết nối thành công với move_base!")

# --- HÀM XÓA COSTMAP (Cứu hộ khi robot bị kẹt) ---
def clear_costmaps():
    rospy.loginfo("🧹 Robot đang bối rối... Tiến hành xóa Costmap để nhìn lại đường!")
    try:
        # Gọi service clear_costmaps mặc định của move_base
        rospy.wait_for_service('/move_base/clear_costmaps', timeout=2.0)
        reset_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        reset_srv()
        rospy.loginfo("✨ Đã xóa Costmap thành công. Robot sẽ tính lại đường đi.")
    except rospy.ROSException:
        rospy.logwarn("⚠️ Không tìm thấy dịch vụ '/move_base/clear_costmaps'. Bỏ qua bước này.")
    except Exception as e:
        rospy.logwarn(f"⚠️ Lỗi khi xóa costmap: {e}")

# --- HÀM DI CHUYỂN THÔNG MINH ---
def move_to_point(x, y, z_orient, w_orient, point_name):
    global move_base_client

    # 1. Tạo mục tiêu (Goal)
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.z = z_orient
    goal.target_pose.pose.orientation.w = w_orient

    rospy.loginfo(f"🚀 Bắt đầu đi tới: {point_name} [x={x}, y={y}]")
    
    # 2. Gửi lệnh đi
    move_base_client.send_goal(goal)

    # 3. Chờ kết quả (Lần 1: Chờ 60 giây)
    finished_within_time = move_base_client.wait_for_result(rospy.Duration(60))

    # 4. Xử lý nếu quá giờ (Timeout) -> Thử cứu hộ
    if not finished_within_time:
        rospy.logwarn(f"⚠️ CẢNH BÁO: Quá 60s mà chưa tới {point_name}. Đang thử cứu hộ...")
        move_base_client.cancel_goal() # Hủy lệnh cũ
        
        clear_costmaps() # Xóa bản đồ vật cản ảo
        
        rospy.loginfo(f"🔄 Đang thử đi lại tới {point_name} lần 2...")
        move_base_client.send_goal(goal) # Gửi lại lệnh
        
        # Chờ thêm 40 giây nữa
        finished_within_time = move_base_client.wait_for_result(rospy.Duration(40))

    # 5. Kiểm tra kết quả cuối cùng
    if not finished_within_time:
        move_base_client.cancel_goal()
        rospy.logerr(f"❌ THẤT BẠI: Robot bỏ cuộc với điểm {point_name} sau 2 lần thử.")
        return False

    state = move_base_client.get_state()
    if state == GoalStatus.SUCCEEDED:
        rospy.loginfo(f"✅ HOÀN THÀNH: Đã đến {point_name}!")
        return True
    else:
        rospy.logerr(f"❌ THẤT BẠI: Robot dừng với trạng thái lỗi {state}")
        return False

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == '__main__':
    try:
        rospy.init_node('mir_waypoint_controller')
        init_client() # Kết nối 1 lần duy nhất
        
        # --- ĐIỂM 1 ---
        # (Lưu ý: Tọa độ này bạn phải lấy chính xác từ RViz bằng nút 'Publish Point' hoặc xem Odom)
        success = move_to_point(-7.947, -2.469, -0.014, 0.999, "Điểm Nhớ 1")
        
        if success:
            rospy.loginfo("📦 Đang dừng 3s để làm nhiệm vụ...")
            rospy.sleep(3) 
            
            # --- ĐIỂM 2 ---
            move_to_point(8.620, -7.442, -0.999, 0.013, "Điểm Nhớ 2")
        else:
            rospy.logerr("🛑 Dừng quy trình do Điểm 1 thất bại.")
        
        rospy.loginfo("🏁 Kết thúc chương trình.")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Đã hủy điều hướng.")