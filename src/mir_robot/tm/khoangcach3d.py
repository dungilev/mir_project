import json
import os
import socket
import time
from collections import deque

import cv2
import numpy as np
import math
import mediapipe as mp
from ultralytics import YOLO

try:
    import rospy
    from geometry_msgs.msg import PointStamped
except Exception:
    rospy = None
    PointStamped = None

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None
    RS_IMPORT_ERROR = e
else:
    RS_IMPORT_ERROR = None


def resolve_infer_device():
    mode = os.getenv("KHOANGCACH3D_DEVICE", "auto").strip().lower()
    if mode not in {"auto", "cpu", "gpu"}:
        mode = "auto"

    if mode == "cpu":
        return "cpu", False

    try:
        import torch
    except Exception:
        return "cpu", False

    if not torch.cuda.is_available():
        return "cpu", False

    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    target_arch = f"sm_{cc_major}{cc_minor}"
    supported_arch = set(torch.cuda.get_arch_list())
    if target_arch in supported_arch:
        return "0", True

    # Voi GPU doi moi (vi du RTX 50), torch co the chua liet ke SM moi
    # nhung van co kha nang chay duoc. Thu GPU truoc, chi fallback CPU khi infer loi.
    print(
        f"GPU {target_arch} not listed in torch build {sorted(supported_arch)}. "
        "Trying GPU first; will fallback to CPU if inference fails."
    )
    return "0", False


def get_depth_distance_m(depth_frame, box, frame_w, frame_h):
    x1, y1, x2, y2 = map(int, box)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    roi_size = int(max(8, min(x2 - x1, y2 - y1) // 4))

    sample_step = max(1, int(os.getenv("KHOANGCACH3D_DEPTH_STEP", "8")))

    distances = []
    for dx in range(-roi_size, roi_size + 1, sample_step):
        for dy in range(-roi_size, roi_size + 1, sample_step):
            px = center_x + dx
            py = center_y + dy
            
            # Khung hinh mau color_frame (frame) da bi lat ngang bang cv2.flip(frame, 1)
            # Nen ta phai lat lai x de tra cuu dung tren depth_frame goc
            orig_px = frame_w - 1 - px
            
            if 0 <= orig_px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(orig_px, py)
                if 0.3 < d < 6.0:
                    distances.append(d)

    return float(np.median(distances)) if distances else -1.0


def get_person_relative_position_m(depth_frame, box, frame_w, frame_h, depth_intrinsics, distance_m=None):
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # frame hien thi da flip ngang, can doi ve pixel tren depth frame goc
    orig_px = frame_w - 1 - center_x
    if not (0 <= orig_px < frame_w and 0 <= center_y < frame_h):
        return None

    d_m = distance_m if distance_m is not None else get_depth_distance_m(depth_frame, box, frame_w, frame_h)
    if d_m <= 0:
        return None

    if depth_intrinsics is None:
        # fallback xap xi theo FOV ngang ~69 do (D435)
        hfov_rad = math.radians(69.0)
        angle = ((orig_px - frame_w / 2.0) / frame_w) * hfov_rad
        x_cam = d_m * math.tan(angle)
    else:
        x_cam = (orig_px - depth_intrinsics.ppx) / depth_intrinsics.fx * d_m

    # quy uoc: base_link (x tien, y trai)
    forward_m = d_m
    left_m = -x_cam
    return forward_m, left_m


def main():
    if rs is None:
        print(f"pyrealsense2 is not available: {RS_IMPORT_ERROR}")
        print("Install RealSense SDK/python bindings before running this file.")
        return

    model_path = os.getenv("KHOANGCACH3D_MODEL", "yolo11n-pose.pt")
    conf_thres = float(os.getenv("KHOANGCACH3D_CONF", "0.4"))
    smooth_window = int(os.getenv("KHOANGCACH3D_SMOOTH", "5"))
    infer_every = max(1, int(os.getenv("KHOANGCACH3D_INFER_EVERY", "2")))
    hands_every = max(1, int(os.getenv("KHOANGCACH3D_HANDS_EVERY", "3")))
    infer_imgsz = max(256, int(os.getenv("KHOANGCACH3D_IMGSZ", "416")))
    hands_scale = min(1.0, max(0.2, float(os.getenv("KHOANGCACH3D_HANDS_SCALE", "0.5"))))
    lock_confirm_frames = max(1, int(os.getenv("KHOANGCACH3D_LOCK_CONFIRM_FRAMES", "6")))
    unlock_confirm_frames = max(1, int(os.getenv("KHOANGCACH3D_UNLOCK_CONFIRM_FRAMES", "6")))
    show_perf = os.getenv("KHOANGCACH3D_SHOW_PERF", "1").strip().lower() not in {"0", "false", "no", "off"}

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Cannot load model {model_path}: {e}")
        return

    infer_device, use_half = resolve_infer_device()
    gpu_runtime_failed = False
    print(f"Inference device: {'GPU' if infer_device != 'cpu' else 'CPU'}")
    print(
        f"Perf config: infer_every={infer_every}, hands_every={hands_every}, "
        f"imgsz={infer_imgsz}, hands_scale={hands_scale}"
    )

    # Camera dat o duoi robot, cach tam (base_link) khoang 0.475m ve phia sau
    # Can tru offset nay de chuyen tu toa do camera -> toa do base_link
    camera_offset_x = float(os.getenv("KHOANGCACH3D_CAMERA_OFFSET_X", "0.475"))

    ros_enable = os.getenv("KHOANGCACH3D_ENABLE_ROS", "1").strip().lower() not in {"0", "false", "no", "off"}
    ros_pub = None
    udp_sock = None
    udp_target = None
    if ros_enable:
        if rospy is None or PointStamped is None:
            # Fallback: dung UDP socket de gui du lieu vi tri sang container
            udp_host = os.getenv("KHOANGCACH3D_UDP_HOST", "127.0.0.1")
            udp_port = int(os.getenv("KHOANGCACH3D_UDP_PORT", "9877"))
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_target = (udp_host, udp_port)
            print(f"ROS python chua san sang. Dung UDP fallback -> {udp_host}:{udp_port}")
        else:
            try:
                if not rospy.core.is_initialized():
                    rospy.init_node("khoangcach3d_tracker", anonymous=True, disable_signals=True)
                ros_pub = rospy.Publisher("/person_locked_relative", PointStamped, queue_size=10)
            except Exception as e:
                print(f"Khong the init ROS publisher: {e}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Cannot start RealSense pipeline: {e}")
        return

    depth_intrinsics = None
    try:
        profile = pipeline.get_active_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()
    except Exception:
        depth_intrinsics = None

    align = rs.align(rs.stream.color)
    dist_history = deque(maxlen=max(1, smooth_window))
    
    hands_detector = None
    hands_detector_mode = None  # "solutions" | "tasks"
    try:
        mp_hands = getattr(mp, "solutions", None)
        if mp_hands is not None and hasattr(mp_hands, "hands"):
            hands_detector = mp_hands.hands.Hands(
                static_image_mode=False,
                max_num_hands=5,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            hands_detector_mode = "solutions"
        else:
            task_model_hint = os.getenv("KHOANGCACH3D_HANDS_TASK_MODEL", "hand_landmarker.task").strip()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            task_model_candidates = [
                task_model_hint,
                os.path.join(script_dir, task_model_hint),
                os.path.join(script_dir, "..", "..", "..", task_model_hint),
            ]
            task_model_path = None
            for p in task_model_candidates:
                p_norm = os.path.abspath(p)
                if os.path.exists(p_norm):
                    task_model_path = p_norm
                    break
            mp_tasks = getattr(mp, "tasks", None)
            mp_vision = getattr(mp_tasks, "vision", None) if mp_tasks is not None else None
            if mp_tasks is not None and mp_vision is not None and task_model_path is not None:
                base_options = mp_tasks.BaseOptions(model_asset_path=task_model_path)
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=5,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                hands_detector = mp_vision.HandLandmarker.create_from_options(options)
                hands_detector_mode = "tasks"
                print(f"Using MediaPipe Tasks HandLandmarker model: {task_model_path}")
            else:
                print(
                    "Mediapipe does not expose solutions.hands in this environment. "
                    "Gesture lock/unlock is disabled. "
                    f"To enable with Tasks API, set KHOANGCACH3D_HANDS_TASK_MODEL to a valid hand_landmarker.task path. "
                    f"Checked: {', '.join(os.path.abspath(p) for p in task_model_candidates)}"
                )
    except Exception as e:
        print(f"Cannot init MediaPipe Hands ({e}). Gesture lock/unlock is disabled.")
        hands_detector = None
        hands_detector_mode = None

    window_name = "khoangcach3d"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    print("Press 'q' to exit")
    prev_time = 0.0
    frame_idx = 0
    cached_results = []
    detected_hands = []  # (hx, hy, fingers)
    perf_acc = {"infer": 0.0, "hands": 0.0, "depth": 0.0, "frames": 0}
    perf_report_t0 = time.time()

    hand_raise_start = {}  # dict: track_id -> time_started_open5
    open5_confirm_count = {}  # dict: track_id -> consecutive frames of (raise + open5)
    fist_hold_start = None  # time_started_fist for locked target
    fist_confirm_count = 0  # consecutive frames of (raise + fist0) for locked target
    locked_target_id = None
    locked_target_last_seen = None
    locked_relative_pos = None

    try:
        while True:
            frame_idx += 1
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            # Lat ngang anh mau (mirror)
            frame = cv2.flip(frame, 1)
            
            if hands_detector is not None and (frame_idx % hands_every == 0 or not detected_hands):
                t_hands0 = time.perf_counter()
                if hands_scale < 0.999:
                    hand_frame = cv2.resize(
                        frame,
                        (int(frame.shape[1] * hands_scale), int(frame.shape[0] * hands_scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    hand_frame = frame

                rgb_frame = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)
                detected_hands = []

                def _count_fingers_and_open5(get_dist):
                    """Dem so ngon tay mo va kiem tra xoe 5 ngon nghiem ngat."""
                    tip_ids  = [4, 8, 12, 16, 20]
                    pip_ids  = [2, 6, 10, 14, 18]
                    tip_dists = [get_dist(t, 0) for t in tip_ids]
                    pip_dists = [get_dist(p, 0) for p in pip_ids]

                    # Dem ngon co ban (nguong thap)
                    fingers = sum(1 for td, pd in zip(tip_dists, pip_dists) if td > pd)

                    # Kiem tra xoe 5 ngon NGHIEM NGAT:
                    # 1) Tat ca 5 dau ngon tay phai xa co tay hon ro ret (>= 1.3 lan)
                    all_extended = all(
                        td >= 1.3 * pd for td, pd in zip(tip_dists, pip_dists)
                    )
                    # 2) Ngon cai phai xoe ra xa ngon tro (> 45% chieu rong long ban tay)
                    palm_width = max(1e-6, get_dist(5, 17))  # INDEX_MCP -> PINKY_MCP
                    thumb_spread = get_dist(4, 8) > 0.45 * palm_width

                    open5_strict = (fingers == 5) and all_extended and thumb_spread
                    return fingers, open5_strict

                if hands_detector_mode == "solutions":
                    hand_results = hands_detector.process(rgb_frame)
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            def get_dist(idx1, idx2):
                                p1 = hand_landmarks.landmark[idx1]
                                p2 = hand_landmarks.landmark[idx2]
                                return math.hypot(p1.x - p2.x, p1.y - p2.y)

                            fingers, open5_strict = _count_fingers_and_open5(get_dist)

                            wrist = hand_landmarks.landmark[0]
                            hx = int((wrist.x * hand_frame.shape[1]) / hands_scale)
                            hy = int((wrist.y * hand_frame.shape[0]) / hands_scale)
                            detected_hands.append((hx, hy, fingers, open5_strict))
                elif hands_detector_mode == "tasks":
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    hand_results = hands_detector.detect(mp_image)
                    if getattr(hand_results, "hand_landmarks", None):
                        for landmarks in hand_results.hand_landmarks:
                            def get_dist(idx1, idx2):
                                p1 = landmarks[idx1]
                                p2 = landmarks[idx2]
                                return math.hypot(p1.x - p2.x, p1.y - p2.y)

                            fingers, open5_strict = _count_fingers_and_open5(get_dist)

                            wrist = landmarks[0]
                            hx = int((wrist.x * hand_frame.shape[1]) / hands_scale)
                            hy = int((wrist.y * hand_frame.shape[0]) / hands_scale)
                            detected_hands.append((hx, hy, fingers, open5_strict))
                perf_acc["hands"] += (time.perf_counter() - t_hands0) * 1000.0
            
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            nearest_cm = -1.0
            if frame_idx % infer_every == 0 or not cached_results:
                try:
                    t_infer0 = time.perf_counter()
                    cached_results = model.track(
                        frame,
                        conf=conf_thres,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                        device=infer_device,
                        half=use_half,
                        imgsz=infer_imgsz,
                    )
                    perf_acc["infer"] += (time.perf_counter() - t_infer0) * 1000.0
                except Exception as e:
                    cv2.putText(frame, f"Infer error: {str(e)[:60]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if infer_device != "cpu" and not gpu_runtime_failed:
                        print(f"GPU infer failed: {e}")
                        print("Fallback to CPU for next frames.")
                        infer_device = "cpu"
                        use_half = False
                        gpu_runtime_failed = True
                    cached_results = []

            results = cached_results

            # Kiem tra auto-unlock neu muc tieu bi xoa khoi tracker > 3.0s
            if locked_target_id is not None:
                current_ids = set()
                for r in results:
                    if r.boxes is not None and r.boxes.id is not None:
                        current_ids.update(r.boxes.id.cpu().numpy().astype(int).tolist())
                if locked_target_id in current_ids:
                    locked_target_last_seen = curr_time
                elif locked_target_last_seen is not None and curr_time - locked_target_last_seen > 3.0:
                    locked_target_id = None
                    locked_relative_pos = None
                    fist_hold_start = None
                    fist_confirm_count = 0
                    locked_target_last_seen = None
                    hand_raise_start.clear()
                    open5_confirm_count.clear()

            for result in results:
                boxes = result.boxes
                keypoints = getattr(result, "keypoints", None)
                if boxes is None:
                    continue

                for i, box in enumerate(boxes):
                    track_id = int(box.id[0].item()) if box.id is not None else -1
                    
                    cls = int(box.cls[0].cpu().item())
                    conf = box.conf[0].cpu().item()
                    if cls != 0 or conf < conf_thres:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    t_depth0 = time.perf_counter()
                    d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame.shape[1], frame.shape[0])
                    perf_acc["depth"] += (time.perf_counter() - t_depth0) * 1000.0

                    # Dieu chinh khoang cach do camera lap nghieng/cao 1.8m
                    # Su dung dinh ly Pytago: d_m la khoang cach duong cheo (canh huyen)
                    chieu_cao_camera = 1.8
                    chieu_cao_tam_nguoi = 0.6 # Do nguoi DANG NGOI nen tam ban than khoang 0.6m
                    delta_h = chieu_cao_camera - chieu_cao_tam_nguoi  # chenh lech do cao ~ 1.2m
                    if d_m > delta_h:
                        d_ngang_m = math.sqrt(d_m**2 - delta_h**2)
                    else:
                        d_ngang_m = d_m

                    # Kiem tra xem nguoi nay co gio tay khong
                    is_raising = False
                    l_wrist_xy = None
                    r_wrist_xy = None
                    if keypoints is not None and getattr(keypoints, 'data', None) is not None and i < len(keypoints.data):
                        kpts = keypoints.data[i].cpu().numpy()
                        if len(kpts) >= 11:
                            l_shoulder, r_shoulder = kpts[5], kpts[6]
                            l_wrist, r_wrist = kpts[9], kpts[10]
                            
                            def valid_kpt(kp):
                                return kp[2] > 0.4 if len(kp) >= 3 else (float(kp[0]) > 0 and float(kp[1]) > 0)
                            
                            # Cổ tay cao hơn vai (y nhỏ hơn) là giơ tay
                            if valid_kpt(l_shoulder) and valid_kpt(l_wrist) and float(l_wrist[1]) < float(l_shoulder[1]):
                                is_raising = True
                            if valid_kpt(r_shoulder) and valid_kpt(r_wrist) and float(r_wrist[1]) < float(r_shoulder[1]):
                                is_raising = True

                            if valid_kpt(l_wrist):
                                l_wrist_xy = (float(l_wrist[0]), float(l_wrist[1]))
                            if valid_kpt(r_wrist):
                                r_wrist_xy = (float(r_wrist[0]), float(r_wrist[1]))
                    
                    has_open_five = False
                    has_fist = False
                    fingers_for_lock = []
                    fingers_for_unlock = []
                    hand_match_radius_px = max(40.0, 0.25 * max(float(x2 - x1), float(y2 - y1)))
                    open5_flags = []
                    if is_raising:
                        for hx, hy, fingers, open5_strict in detected_hands:
                            in_box = x1 <= hx <= x2 and y1 <= hy <= y2
                            near_left_wrist = l_wrist_xy is not None and math.hypot(hx - l_wrist_xy[0], hy - l_wrist_xy[1]) <= hand_match_radius_px
                            near_right_wrist = r_wrist_xy is not None and math.hypot(hx - r_wrist_xy[0], hy - r_wrist_xy[1]) <= hand_match_radius_px
                            if in_box:
                                fingers_for_lock.append(fingers)
                                fingers_for_unlock.append(fingers)
                                open5_flags.append(open5_strict)
                            elif near_left_wrist or near_right_wrist:
                                fingers_for_unlock.append(fingers)
                    elif hands_detector is not None:
                        # Khi KHONG gio tay: chi thu thap du lieu unlock, KHONG thu thap lock
                        for hx, hy, fingers, open5_strict in detected_hands:
                            in_box = x1 <= hx <= x2 and y1 <= hy <= y2
                            near_left_wrist = l_wrist_xy is not None and math.hypot(hx - l_wrist_xy[0], hy - l_wrist_xy[1]) <= hand_match_radius_px
                            near_right_wrist = r_wrist_xy is not None and math.hypot(hx - r_wrist_xy[0], hy - r_wrist_xy[1]) <= hand_match_radius_px
                            if in_box or near_left_wrist or near_right_wrist:
                                fingers_for_unlock.append(fingers)

                    # Dung open5_strict (nghiem ngat) thay vi chi dem so ngon
                    if open5_flags:
                        has_open_five = any(open5_flags)
                    if fingers_for_unlock:
                        # Cho phep <= 1 vi mediapipe hay nhan nham ngon cai thanh 1 ngon khi nam dam
                        has_fist = any(f <= 1 for f in fingers_for_unlock)
                    elif is_raising and hands_detector is not None:
                        # Gio tay nhung mediapipe KHONG thay ban tay nao -> nam dam qua chat
                        # MediaPipe thuong khong nhan dien duoc nam dam chat -> coi nhu fist
                        has_fist = True
                    
                    # Logic khoa muc tieu: gio tay + xoe dung 5 ngon lien tuc trong 5s
                    if track_id != -1 and locked_target_id is None:
                        lock_gesture_ok = hands_detector is not None and is_raising and has_open_five
                        if lock_gesture_ok:
                            open5_confirm_count[track_id] = open5_confirm_count.get(track_id, 0) + 1
                        else:
                            open5_confirm_count.pop(track_id, None)
                            hand_raise_start.pop(track_id, None)

                        if lock_gesture_ok and open5_confirm_count.get(track_id, 0) >= lock_confirm_frames:
                            if track_id not in hand_raise_start:
                                hand_raise_start[track_id] = curr_time

                            hold_time = curr_time - hand_raise_start[track_id]
                            cv2.putText(frame, f"Giu: {hold_time:.1f}s/5.0s", (int(x1), max(40, int(y1) - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            if hold_time >= 5.0:
                                locked_target_id = track_id
                                fist_hold_start = None
                                fist_confirm_count = 0
                                hand_raise_start.clear()
                                open5_confirm_count.clear()
                                locked_target_last_seen = curr_time
                        elif lock_gesture_ok and open5_confirm_count.get(track_id, 0) > 0:
                            cv2.putText(
                                frame,
                                f"XN 5 ngon: {open5_confirm_count.get(track_id, 0)}/{lock_confirm_frames}",
                                (int(x1), max(40, int(y1) - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 200, 255),
                                2,
                            )

                    # Logic mo khoa: muc tieu da lock gio tay + nam dam dung 0 ngon lien tuc trong 5s
                    if locked_target_id is not None and track_id == locked_target_id:
                        unlock_gesture_ok = hands_detector is not None and is_raising and has_fist
                        if unlock_gesture_ok:
                            fist_confirm_count += 1
                        else:
                            fist_confirm_count = 0
                            fist_hold_start = None

                        if unlock_gesture_ok and fist_confirm_count >= unlock_confirm_frames:
                            if fist_hold_start is None:
                                fist_hold_start = curr_time

                            fist_hold_time = curr_time - fist_hold_start
                            cv2.putText(
                                frame,
                                f"Huy lock: {fist_hold_time:.1f}s/5.0s",
                                (int(x1), max(20, int(y1) - 55)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 140, 255),
                                2,
                            )
                            if fist_hold_time >= 5.0:
                                locked_target_id = None
                                locked_relative_pos = None
                                fist_hold_start = None
                                fist_confirm_count = 0
                                hand_raise_start.clear()
                                open5_confirm_count.clear()
                                locked_target_last_seen = None
                                continue
                        elif unlock_gesture_ok and fist_confirm_count > 0:
                            cv2.putText(
                                frame,
                                f"XN nam dam: {fist_confirm_count}/{unlock_confirm_frames}",
                                (int(x1), max(20, int(y1) - 55)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 180, 255),
                                2,
                            )

                    # Khong ve bounding box va cac text ra man hinh cho nguoi la
                    is_stranger = (locked_target_id is not None and track_id != locked_target_id)
                    if is_stranger:
                        continue

                    if locked_target_id is not None and track_id == locked_target_id:
                        rel = get_person_relative_position_m(
                            depth_frame,
                            (x1, y1, x2, y2),
                            frame.shape[1],
                            frame.shape[0],
                            depth_intrinsics,
                            d_m,
                        )
                        if rel is not None:
                            locked_relative_pos = rel

                    color = (0, 255, 0) if d_ngang_m > 0 else (0, 0, 255) # mac dinh la hien do hoac xanh la (thay duoc)
                    if locked_target_id is not None and track_id == locked_target_id:
                        color = (0, 255, 255) # Mau VANG cho muc tieu da bi khoa (Locked)
                    elif is_raising and has_open_five:
                        color = (255, 0, 255) # Neu dung cu chi thi mau tim

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    status_text = "Gio tay: CO" if is_raising else "Gio tay: KHONG"
                    if locked_target_id is not None and track_id == locked_target_id:
                        status_text += f", Fist: {'CO' if has_fist else 'KHONG'}"
                    cv2.putText(frame, status_text, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if fingers_for_unlock:
                        cv2.putText(
                            frame,
                            f"Ngon tay: {max(fingers_for_unlock)}",
                            (int(x1), int(y2) + 42),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )
                    elif is_raising and hands_detector is not None:
                        cv2.putText(
                            frame,
                            "Ngon tay: N/A (khong thay tay)",
                            (int(x1), int(y2) + 42),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

                    if d_ngang_m > 0:
                        d_cm = d_ngang_m * 100.0
                        if nearest_cm < 0 or d_cm < nearest_cm:
                            nearest_cm = d_cm
                        cv2.putText(frame, f"Cach: {d_cm:.1f} cm", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Cach: N/A", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if locked_target_id is not None and track_id == locked_target_id and locked_relative_pos is not None:
                        forward_m, left_m = locked_relative_pos
                        cv2.putText(
                            frame,
                            f"LOCKED rel x={forward_m:.2f}m y={left_m:.2f}m",
                            (int(x1), int(y2) + 64),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2,
                        )

            if nearest_cm > 0:
                dist_history.append(nearest_cm)

            smooth_cm = float(np.mean(dist_history)) if dist_history else -1.0

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            top_dist = f"KHOANG CACH: {smooth_cm:.1f} cm" if smooth_cm > 0 else "KHOANG CACH: N/A"
            cv2.putText(frame, top_dist, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            lock_text = f"LOCK ID: {locked_target_id}" if locked_target_id is not None else "LOCK ID: NONE"
            cv2.putText(frame, lock_text, (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            perf_acc["frames"] += 1
            if show_perf and perf_acc["frames"] > 0:
                infer_ms = perf_acc["infer"] / perf_acc["frames"]
                hands_ms = perf_acc["hands"] / perf_acc["frames"]
                depth_ms = perf_acc["depth"] / perf_acc["frames"]
                cv2.putText(
                    frame,
                    f"ms infer/hands/depth: {infer_ms:.1f}/{hands_ms:.1f}/{depth_ms:.1f}",
                    (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            if show_perf and time.time() - perf_report_t0 >= 2.0:
                n = max(1, perf_acc["frames"])
                print(
                    f"[perf] fps~{fps:.1f} infer={perf_acc['infer']/n:.1f}ms "
                    f"hands={perf_acc['hands']/n:.1f}ms depth={perf_acc['depth']/n:.1f}ms"
                )
                perf_acc = {"infer": 0.0, "hands": 0.0, "depth": 0.0, "frames": 0}
                perf_report_t0 = time.time()

            if locked_target_id is not None and locked_relative_pos is not None:
                # Tru offset camera de chuyen sang toa do base_link
                corrected_x = float(locked_relative_pos[0]) - camera_offset_x
                corrected_y = float(locked_relative_pos[1])
                if ros_pub is not None:
                    try:
                        msg = PointStamped()
                        msg.header.stamp = rospy.Time.now()
                        msg.header.frame_id = "base_link"
                        msg.point.x = corrected_x
                        msg.point.y = corrected_y
                        msg.point.z = 0.0
                        ros_pub.publish(msg)
                    except Exception:
                        pass
                elif udp_sock is not None and udp_target is not None:
                    try:
                        packet = json.dumps({
                            "x": corrected_x,
                            "y": corrected_y,
                            "t": time.time(),
                        }).encode("utf-8")
                        udp_sock.sendto(packet, udp_target)
                    except Exception:
                        pass

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if hands_detector is not None and hasattr(hands_detector, "close"):
            try:
                hands_detector.close()
            except Exception:
                pass
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
