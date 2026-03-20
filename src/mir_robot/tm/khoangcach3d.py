import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

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

    print(f"GPU {target_arch} not supported by current torch build {sorted(supported_arch)}. Fallback CPU.")
    return "cpu", False


def get_depth_distance_m(depth_frame, box, frame_w, frame_h):
    x1, y1, x2, y2 = map(int, box)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    roi_size = max(8, min(x2 - x1, y2 - y1) // 4)

    distances = []
    for dx in range(-roi_size, roi_size + 1, 5):
        for dy in range(-roi_size, roi_size + 1, 5):
            px = center_x + dx
            py = center_y + dy
            if 0 <= px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(px, py)
                if 0.3 < d < 6.0:
                    distances.append(d)

    return float(np.median(distances)) if distances else -1.0


def main():
    if rs is None:
        print(f"pyrealsense2 is not available: {RS_IMPORT_ERROR}")
        print("Install RealSense SDK/python bindings before running this file.")
        return

    model_path = os.getenv("KHOANGCACH3D_MODEL", "yolo11n-pose.pt")
    conf_thres = float(os.getenv("KHOANGCACH3D_CONF", "0.4"))
    smooth_window = int(os.getenv("KHOANGCACH3D_SMOOTH", "5"))

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Cannot load model {model_path}: {e}")
        return

    infer_device, use_half = resolve_infer_device()
    print(f"Inference device: {'GPU' if infer_device != 'cpu' else 'CPU'}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Cannot start RealSense pipeline: {e}")
        return

    align = rs.align(rs.stream.color)
    dist_history = deque(maxlen=max(1, smooth_window))

    window_name = "khoangcach3d"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    print("Press 'q' to exit")
    prev_time = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            nearest_cm = -1.0
            try:
                results = model(frame, conf=conf_thres, verbose=False, device=infer_device, half=use_half)
            except Exception as e:
                cv2.putText(frame, f"Infer error: {str(e)[:60]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                results = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls = int(box.cls[0].cpu().item())
                    conf = box.conf[0].cpu().item()
                    if cls != 0 or conf < conf_thres:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame.shape[1], frame.shape[0])

                    color = (0, 255, 0) if d_m > 0 else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    if d_m > 0:
                        d_cm = d_m * 100.0
                        if nearest_cm < 0 or d_cm < nearest_cm:
                            nearest_cm = d_cm
                        cv2.putText(frame, f"Cach: {d_cm:.1f} cm", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Cach: N/A", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if nearest_cm > 0:
                dist_history.append(nearest_cm)

            smooth_cm = float(np.mean(dist_history)) if dist_history else -1.0

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            top_dist = f"KHOANG CACH: {smooth_cm:.1f} cm" if smooth_cm > 0 else "KHOANG CACH: N/A"
            cv2.putText(frame, top_dist, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
