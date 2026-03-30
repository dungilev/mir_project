#!/usr/bin/env python3
import json
import os
import re
import signal
import subprocess
import sys
import threading
from collections import deque

import rospy
from std_msgs.msg import String

NAV_QUEUE = deque()
NAV_LOCK = threading.Lock()
NAV_WORKER_STARTED = False
ACTIVE_PROCESS = None
ACTIVE_TARGET = ''
LAST_ORDER_SIG = ''
LAST_ORDER_AT = 0.0


def normalize_target(table_value) -> str:
    if isinstance(table_value, int):
        if 1 <= table_value <= 4:
            return f'ban {table_value}'
        return ''

    if isinstance(table_value, str):
        cleaned = table_value.strip().lower()
        if cleaned == 'bep':
            return 'bep'

        if cleaned.isdigit():
            number = int(cleaned)
            if 1 <= number <= 4:
                return f'ban {number}'

        match = re.search(r'ban\s*([1-4])', cleaned)
        if match:
            return f"ban {match.group(1)}"

        compact = cleaned.replace(' ', '')
        match_compact = re.fullmatch(r'ban([1-4])', compact)
        if match_compact:
            return f"ban {match_compact.group(1)}"

    return ''


def worker_loop() -> None:
    global ACTIVE_PROCESS, ACTIVE_TARGET
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nav_script = os.path.join(script_dir, 'navigationcacdiem.py')
    while not rospy.is_shutdown():
        with NAV_LOCK:
            if not NAV_QUEUE:
                next_target = ''
            else:
                next_target = NAV_QUEUE.popleft()

        if not next_target:
            rospy.sleep(0.2)
            continue

        cmd = [
            'bash',
            '-lc',
            f'source /opt/ros/noetic/setup.bash && '
            f'[ -f /root/catkin_ws/devel/setup.bash ] && source /root/catkin_ws/devel/setup.bash; '
            f'{sys.executable} {nav_script} "{next_target}"'
        ]
        rospy.loginfo('Bắt đầu điều hướng robot tới: %s', next_target)
        rospy.loginfo('Lệnh chạy: %s', ' '.join(cmd))

        process = None
        try:
            process = subprocess.Popen(
                cmd,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            with NAV_LOCK:
                ACTIVE_PROCESS = process
                ACTIVE_TARGET = next_target

            stdout_text, stderr_text = process.communicate(timeout=180)
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout_text,
                stderr=stderr_text,
            )

            if result.returncode == 0:
                rospy.loginfo('Điều hướng %s hoàn tất thành công.', next_target)
            else:
                rospy.logwarn('Điều hướng %s thất bại (code=%s).', next_target, result.returncode)

            output_lines = (result.stdout or '').strip().splitlines()
            error_lines = (result.stderr or '').strip().splitlines()

            for line in output_lines[-8:]:
                rospy.loginfo('[nav stdout] %s', line)
            for line in error_lines[-8:]:
                rospy.logwarn('[nav stderr] %s', line)
        except subprocess.TimeoutExpired:
            rospy.logwarn('Điều hướng %s bị timeout sau 180 giây.', next_target)
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass
                try:
                    process.wait(timeout=3)
                except Exception:
                    pass
        except Exception as exc:
            rospy.logwarn('Lỗi khi chạy điều hướng %s: %s', next_target, exc)
        finally:
            with NAV_LOCK:
                if ACTIVE_PROCESS is process:
                    ACTIVE_PROCESS = None
                    ACTIVE_TARGET = ''


def cancel_active_navigation(reason: str) -> bool:
    with NAV_LOCK:
        process = ACTIVE_PROCESS
        target = ACTIVE_TARGET

    if process is None:
        return False

    try:
        os.killpg(process.pid, signal.SIGKILL)
        rospy.logwarn('Đã hủy điều hướng đang chạy: %s | reason=%s', target, reason)
        return True
    except Exception as exc:
        rospy.logwarn('Không thể hủy điều hướng đang chạy (%s): %s', target, exc)
        return False


def preempt_and_set_queue(targets, reason: str) -> None:
    targets = [t for t in targets if t]
    if not targets:
        return

    with NAV_LOCK:
        current_target = ACTIVE_TARGET
        NAV_QUEUE.clear()
        for target in targets:
            NAV_QUEUE.append(target)
        queue_size = len(NAV_QUEUE)

    if current_target and current_target != targets[0]:
        cancel_active_navigation(reason)

    rospy.loginfo('Ưu tiên lệnh mới: %s (queue=%d) | reason=%s', ' -> '.join(targets), queue_size, reason)


def enqueue_target(target: str, reason: str) -> None:
    with NAV_LOCK:
        NAV_QUEUE.append(target)
        queue_size = len(NAV_QUEUE)
    rospy.loginfo('Đã đưa yêu cầu điều hướng vào hàng đợi: %s (queue=%d) | reason=%s', target, queue_size, reason)


def parse_button_target(raw_data: str) -> str:
    text = (raw_data or '').strip()
    if not text:
        return ''

    try:
        payload = json.loads(text)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        return normalize_target(payload.get('ban'))

    return normalize_target(text)


def on_table_button(msg: String) -> None:
    rospy.loginfo('[OLD SYSTEM] Nút bấm bàn bị bỏ qua để nhường cho mainv1.py!')
    return


def on_order(msg: String) -> None:
    # --- Đang dùng chung với mainv1.py, không được tự động điều robot để tránh xung đột ---
    rospy.loginfo('[OLD SYSTEM] Bỏ qua chuyển hướng vì đã có mainv1.py phụ trách!')
    return


def main() -> None:
    global NAV_WORKER_STARTED
    rospy.init_node('order_listener', anonymous=False)
    rospy.Subscriber('/robot_orders', String, on_order, queue_size=10)
    rospy.Subscriber('/table_call_buttons', String, on_table_button, queue_size=20)

    if not NAV_WORKER_STARTED:
        worker = threading.Thread(target=worker_loop, daemon=True)
        worker.start()
        NAV_WORKER_STARTED = True

    rospy.loginfo('order_listener đã chạy. Đang lắng nghe /robot_orders và /table_call_buttons')
    rospy.spin()


if __name__ == '__main__':
    main()
