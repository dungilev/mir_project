import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


try:
    use_onnx = os.getenv("KHOANGCACH_USE_ONNX", "0") == "1"
    if use_onnx:
        model = YOLO("yolo11n-pose.onnx", task="pose")
    else:
        model = YOLO("yolo11n-pose.pt")
except Exception as e:
    print(f"Loi tai mo hinh: {e}")
    model = YOLO("yolo11n-pose.pt")


def resolve_infer_device():
    mode = os.getenv("KHOANGCACH_DEVICE", "auto").strip().lower()
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

    if mode == "gpu":
        print(f"GPU {target_arch} chua duoc torch ho tro {sorted(supported_arch)}. Fallback CPU.")
    return "cpu", False


REAL_TORSO_CM = 55.0
FOCAL_LENGTH = float(os.getenv("KHOANGCACH_FOCAL", "330.0"))
REAL_BBOX_REF_CM = float(os.getenv("KHOANGCACH_BBOX_REF_CM", "55.0"))
# Mac dinh scale theo hieu chuan thuc te gan nhat: 44cm do duoc ~ 60cm thuc.
CM_SCALE = float(os.getenv("KHOANGCACH_CM_SCALE", "1.36"))
DISTANCE_SMOOTHING = 5


def is_hand_raised(keypoints):
    if len(keypoints) < 11:
        return False

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    conf_thresh = 0.5
    hand_raised = False

    if left_shoulder[2] > conf_thresh and left_wrist[2] > conf_thresh:
        if left_wrist[1] < left_shoulder[1] - 20:
            hand_raised = True

    if right_shoulder[2] > conf_thresh and right_wrist[2] > conf_thresh:
        if right_wrist[1] < right_shoulder[1] - 20:
            hand_raised = True

    return hand_raised


def get_torso_pixel_height(keypoints):
    if len(keypoints) < 13:
        return 0.0

    l_shoulder = keypoints[5]
    r_shoulder = keypoints[6]
    l_hip = keypoints[11]
    r_hip = keypoints[12]

    conf_thresh = 0.4
    h_left = abs(l_shoulder[1] - l_hip[1]) if l_shoulder[2] > conf_thresh and l_hip[2] > conf_thresh else 0.0
    h_right = abs(r_shoulder[1] - r_hip[1]) if r_shoulder[2] > conf_thresh and r_hip[2] > conf_thresh else 0.0
    return float(max(h_left, h_right))


def calculate_distance_cm_from_torso(keypoints):
    pix_h = get_torso_pixel_height(keypoints)
    if pix_h <= 10.0:
        return -1.0
    return (REAL_TORSO_CM * FOCAL_LENGTH) / pix_h


def calculate_distance_cm_from_bbox(box):
    x1, y1, x2, y2 = box
    pix_h = max(0.0, float(y2 - y1))
    if pix_h < 25.0:
        return -1.0
    return (REAL_BBOX_REF_CM * FOCAL_LENGTH) / pix_h


def get_depth_distance_m(depth_frame, box, frame_w, frame_h):
    x1, y1, x2, y2 = map(int, box)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    roi_size = max(10, min(x2 - x1, y2 - y1) // 4)

    distances = []
    for dx in range(-roi_size, roi_size + 1, 5):
        for dy in range(-roi_size, roi_size + 1, 5):
            px = center_x + dx
            py = center_y + dy
            if 0 <= px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(px, py)
                if 0.3 < d < 6.0:
                    distances.append(d)

    return float(np.median(distances)) if distances else 0.0


def init_realsense_if_available():
    mode = os.getenv("KHOANGCACH_USE_REALSENSE", "0").strip().lower()
    if rs is None or mode in {"0", "false", "no", "off"}:
        return None, None

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("Nguon camera: RealSense")
        return pipeline, align
    except Exception as e:
        if mode == "1":
            print(f"RealSense loi: {e}")
        return None, None


def init_webcam_with_fallback():
    cam_index = int(os.getenv("KHOANGCACH_CAM_INDEX", "0"))
    index_candidates = [cam_index]
    backend_candidates = [cv2.CAP_ANY, cv2.CAP_V4L2]

    best_cap = None
    best_meta = None
    best_score = -1.0

    for idx in index_candidates:
        for backend in backend_candidates:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            vals = []
            for _ in range(20):
                ret, frame = cap.read()
                if ret and frame is not None:
                    vals.append(float(frame.mean()))

            if not vals:
                cap.release()
                continue

            score = float(np.mean(vals))
            if score > best_score:
                if best_cap is not None:
                    best_cap.release()
                best_cap = cap
                best_score = score
                best_meta = (idx, backend)
            else:
                cap.release()

    if best_cap is None:
        return None

    idx, backend = best_meta
    print(f"Nguon camera: Webcam RGB (index={idx}, backend={backend}, mean={best_score:.1f})")
    return best_cap


def prepare_frame_for_display(frame):
    if frame is None:
        return None

    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if float(frame.mean()) < 20.0:
        frame = cv2.convertScaleAbs(frame, alpha=1.6, beta=35)

    return frame


def main():
    global FOCAL_LENGTH

    pipeline, align = init_realsense_if_available()
    cap = None
    if pipeline is None:
        cap = init_webcam_with_fallback()
        if cap is None:
            print("Khong the mo webcam")
            return

    infer_device, use_half = resolve_infer_device()
    infer_every_n = max(1, int(os.getenv("KHOANGCACH_INFER_EVERY", "2")))
    camera_only = os.getenv("KHOANGCACH_CAMERA_ONLY", "0") == "1"
    calib_cm = float(os.getenv("KHOANGCACH_CALIB_CM", "50.0"))

    print("Nhan 'q' de thoat")
    print(f"Nhan 'c' de calibrate o {calib_cm:.0f} cm")
    print(f"Thiet bi suy luan: {'GPU' if infer_device != 'cpu' else 'CPU'}")
    print(f"FOCAL hien tai: {FOCAL_LENGTH:.1f}")
    print(f"CM_SCALE hien tai: {CM_SCALE:.2f}")

    prev_time = 0
    frame_idx = 0
    latest_distance_cm = -1.0
    latest_torso_px = 0.0
    latest_bbox_h_px = 0.0

    depth_history = deque(maxlen=DISTANCE_SMOOTHING)
    rgb_history = deque(maxlen=DISTANCE_SMOOTHING)
    last_detections = []

    window_name = "Nhan dien"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    try:
        while True:
            frame_idx += 1
            depth_frame = None

            if pipeline is not None:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                frame = prepare_frame_for_display(frame)
                if frame is None:
                    continue

            now = time.time()
            fps = 1.0 / (now - prev_time) if prev_time > 0 else 0.0
            prev_time = now

            display = frame.copy()
            for det in last_detections:
                x1, y1, x2, y2 = det["box"]
                color = det["color"]
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, det["status"], (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display, det["dist"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(display, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            top_dist = f"KHOANG CACH: {latest_distance_cm:.1f} cm" if latest_distance_cm > 0 else "KHOANG CACH: N/A"
            cv2.putText(display, top_dist, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                new_focal = None
                if latest_torso_px > 0:
                    new_focal = (calib_cm * latest_torso_px) / REAL_TORSO_CM
                elif latest_bbox_h_px > 0:
                    new_focal = (calib_cm * latest_bbox_h_px) / REAL_BBOX_REF_CM

                if new_focal and new_focal > 1:
                    FOCAL_LENGTH = float(new_focal)
                    rgb_history.clear()
                    print(f"[CALIB] FOCAL moi = {FOCAL_LENGTH:.1f}")
                else:
                    print("[CALIB] Chua co du lieu nguoi de calibrate")

            if camera_only:
                continue
            if frame_idx % infer_every_n != 0:
                continue

            try:
                results = model(frame, conf=0.4, verbose=False, device=infer_device, half=use_half)
            except Exception as e:
                print(f"Canh bao infer: {e}")
                continue

            new_detections = []
            distance_values_cm = []

            for result in results:
                boxes = result.boxes
                keypoints_list = result.keypoints
                if boxes is None or keypoints_list is None:
                    continue

                for box, keypoints in zip(boxes, keypoints_list):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    cls = int(box.cls[0].cpu().item())

                    if cls != 0 or conf <= 0.4:
                        continue

                    kpts = keypoints.data[0].cpu().numpy()
                    latest_torso_px = get_torso_pixel_height(kpts)
                    latest_bbox_h_px = max(0.0, float(y2 - y1))

                    if depth_frame is not None:
                        d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame.shape[1], frame.shape[0])
                        if d_m > 0:
                            depth_history.append(d_m)

                    if depth_history:
                        display_distance_m = float(np.mean(depth_history))
                    else:
                        d_cm = calculate_distance_cm_from_torso(kpts)
                        if d_cm <= 0:
                            d_cm = calculate_distance_cm_from_bbox((x1, y1, x2, y2))
                        if d_cm > 0:
                            rgb_history.append(d_cm / 100.0)
                        display_distance_m = float(np.mean(rgb_history)) if rgb_history else -1.0

                    raised = is_hand_raised(kpts)
                    color = (0, 255, 0) if raised else (0, 0, 255)
                    status_text = "Gio tay!" if raised else "Binh thuong"
                    dist_text = (
                        f"Cach: {display_distance_m * 100:.1f} cm"
                        if display_distance_m > 0
                        else "Cach: N/A"
                    )

                    if display_distance_m > 0:
                        distance_values_cm.append(display_distance_m * 100.0 * CM_SCALE)

                    new_detections.append(
                        {
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                            "color": color,
                            "status": status_text,
                            "dist": dist_text,
                        }
                    )

            last_detections = new_detections
            latest_distance_cm = float(np.median(distance_values_cm)) if distance_values_cm else -1.0

    finally:
        if cap is not None:
            cap.release()
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
