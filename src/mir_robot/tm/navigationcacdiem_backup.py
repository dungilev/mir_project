#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation đến các điểm định sẵn qua MiR REST API.
Tạo position tạm → queue mission Move → theo dõi → xóa position tạm.

Cách dùng:
    python3 navigationcacdiem.py bep
    python3 navigationcacdiem.py "ban 1"
    python3 navigationcacdiem.py ban1
    python3 navigationcacdiem.py pos       # xem vị trí hiện tại
    python3 navigationcacdiem.py           # chế độ tương tác
"""

import sys
import math
import time
import hashlib
import base64
import json

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

MIR_IP = "192.168.0.177"

# ─── MiR REST API auth (distributor, SHA-256 hashed password) ───
_pw_hash = hashlib.sha256(b"distributor").hexdigest()
MIR_AUTH = base64.b64encode(f"distributor:{_pw_hash}".encode()).decode()
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Basic {MIR_AUTH}",
}
API = f"http://{MIR_IP}/api/v2.0.0"

# ─────────────────────────────────────────────
#  Danh sách các điểm đích
# ─────────────────────────────────────────────
DIEM = {
    "bep": {
        "x": 1.5493507385253906,
        "y": 15.297784805297852,
        "orientation": 2.0987,  # degrees (from qz/qw)
    },
    "ban 1": {
        "x": 4.804250717163086,
        "y": 15.547493934631348,
        "orientation": 2.209,
    },
    "ban 2": {
        "x": 3.525,
        "y": 11.251,
        "orientation": 2.209,
    },
    "ban 3": {
        "x": 11.357873916625977,
        "y": 15.613031387329102,
        "orientation": -0.143,
    },
    "ban 4": {
        "x": 11.245950698852539,
        "y": 11.012073516845703,
        "orientation": -1.335,
    },
}

ALIAS = {k.replace(" ", ""): k for k in DIEM}


def chuan_hoa(ten: str) -> str:
    ten = ten.strip().lower()
    if ten in DIEM:
        return ten
    return ALIAS.get(ten.replace(" ", ""), ten)


def hien_danh_sach():
    print("Các điểm hợp lệ:")
    for ten, d in DIEM.items():
        print(f"  {ten:8s}  ->  x={d['x']:.3f}, y={d['y']:.3f}")


def api_get(endpoint):
    r = requests.get(f"{API}{endpoint}", headers=HEADERS, timeout=5)
    r.raise_for_status()
    return r.json()


def api_post(endpoint, data):
    r = requests.post(f"{API}{endpoint}", headers=HEADERS, json=data, timeout=5)
    r.raise_for_status()
    return r.json()


def api_put(endpoint, data):
    r = requests.put(f"{API}{endpoint}", headers=HEADERS, json=data, timeout=5)
    r.raise_for_status()
    return r.json()


def api_delete(endpoint):
    r = requests.delete(f"{API}{endpoint}", headers=HEADERS, timeout=5)
    return r.status_code


def get_status():
    return api_get("/status")


def get_position():
    """Lấy vị trí hiện tại từ REST API."""
    s = get_status()
    pos = s.get("position", {})
    return pos.get("x"), pos.get("y"), pos.get("orientation", 0)


def ensure_ready():
    """Đưa robot về READY nếu cần."""
    s = get_status()
    state = s["state_id"]
    print(f"  State: {state} ({s['state_text']})")
    if state != 3:
        print("  Đang chuyển về READY ...")
        api_put("/status", {"state_id": 3})
        time.sleep(1.0)
        s = get_status()
        print(f"  State mới: {s['state_id']} ({s['state_text']})")
    return s["state_id"] == 3


def get_map_id():
    """Lấy map_id hiện tại."""
    s = get_status()
    return s.get("map_id", "")


def create_temp_position(name, x, y, orientation, map_id):
    """Tạo position tạm trên MiR."""
    data = {
        "name": name,
        "pos_x": x,
        "pos_y": y,
        "orientation": orientation,
        "type_id": 0,  # 0 = normal position
        "map_id": map_id,
    }
    result = api_post("/positions", data)
    return result.get("guid", "")


def delete_position(guid):
    """Xóa position tạm."""
    api_delete(f"/positions/{guid}")


def find_move_mission():
    """Tìm mission 'Move' mặc định của MiR."""
    missions = api_get("/missions")
    # Tìm mission tên "Move" (mission mặc định)
    for m in missions:
        if m.get("name") == "Move":
            return m.get("guid", "")
    # Fallback: tìm bất kỳ mission nào có "Move" trong tên
    for m in missions:
        if "move" in m.get("name", "").lower():
            return m.get("guid", "")
    return ""


def queue_mission(mission_guid, position_guid):
    """Queue mission Move đến position."""
    data = {
        "mission_id": mission_guid,
        "parameters": [
            {"input_name": "Position", "value": position_guid}
        ],
        "message": "",
        "priority": 0,
    }
    result = api_post("/mission_queue", data)
    return result.get("id", 0)


def show_position():
    """Hiển thị vị trí robot hiện tại."""
    x, y, ori = get_position()
    print(f"\n  Robot hiện tại: x={x:.4f}, y={y:.4f}, orientation={ori:.2f}°")
    s = get_status()
    print(f"  State: {s['state_id']} ({s['state_text']})")
    print(f"  Mode: {s['mode_id']} ({s['mode_text']})")
    print(f"  Map: {s.get('map_id', '?')[:8]}")
    print()
    if x is not None:
        print("  Khoảng cách đến các điểm:")
        for ten, d in DIEM.items():
            dist = math.sqrt((x - d["x"])**2 + (y - d["y"])**2)
            print(f"    {ten:8s}  ->  {dist:.2f}m")


def go_to(ten_diem: str) -> bool:
    diem = DIEM.get(ten_diem)
    if diem is None:
        print(f"❌ Không tìm thấy điểm '{ten_diem}'.")
        return False

    x, y, ori = get_position()
    print(f"→ Đang di chuyển đến: {ten_diem}")
    print(f"  Mục tiêu: x={diem['x']:.3f}, y={diem['y']:.3f}")
    if x is not None:
        d0 = math.sqrt((x - diem["x"])**2 + (y - diem["y"])**2)
        print(f"  Khoảng cách ban đầu: {d0:.2f}m")

    # Bước 1: Đưa robot về READY
    ensure_ready()

    # Bước 2: Lấy map_id
    map_id = get_map_id()
    print(f"  Map ID: {map_id[:8]}...")

    # Bước 3: Tạo position tạm
    temp_name = f"_nav_{ten_diem}_{int(time.time())}"
    print(f"  Tạo position tạm: {temp_name}")
    pos_guid = create_temp_position(
        temp_name, diem["x"], diem["y"], diem["orientation"], map_id
    )
    if not pos_guid:
        print("❌ Không tạo được position!")
        return False
    print(f"  Position GUID: {pos_guid[:8]}...")

    # Bước 4: Tìm mission Move
    move_guid = find_move_mission()
    if not move_guid:
        print("❌ Không tìm thấy mission 'Move'!")
        delete_position(pos_guid)
        return False
    print(f"  Mission Move GUID: {move_guid[:8]}...")

    # Bước 5: Queue mission
    print("  Đang queue mission Move ...")
    try:
        queue_id = queue_mission(move_guid, pos_guid)
        print(f"  ✅ Đã queue mission! Queue ID: {queue_id}")
    except Exception as e:
        print(f"  ❌ Lỗi queue mission: {e}")
        delete_position(pos_guid)
        return False

    # Bước 6: Chuyển robot sang Executing
    try:
        api_put("/status", {"state_id": 3})
    except Exception:
        pass

    # Bước 7: Theo dõi tiến trình
    print("  Đang theo dõi robot ...")
    deadline = time.time() + 120.0
    last_log = 0

    while time.time() < deadline:
        try:
            s = get_status()
            state = s["state_id"]
            pos = s.get("position", {})
            cx, cy = pos.get("x", 0), pos.get("y", 0)
            dist = math.sqrt((cx - diem["x"])**2 + (cy - diem["y"])**2)

            if time.time() - last_log > 3.0:
                print(f"  [{s['state_text']}] khoảng cách: {dist:.2f}m | vị trí: ({cx:.2f}, {cy:.2f})")
                last_log = time.time()

            # Robot đã đến nơi
            if dist < 0.5:
                print(f"✅ Đã đến '{ten_diem}'! (cách {dist:.2f}m)")
                delete_position(pos_guid)
                return True

            # Mission hoàn thành
            if state == 3:  # READY = mission xong
                # Kiểm tra mission queue item
                try:
                    q = api_get(f"/mission_queue/{queue_id}")
                    q_state = q.get("state", "")
                    if q_state == "Done":
                        print(f"✅ Mission Done - đã đến '{ten_diem}'!")
                        delete_position(pos_guid)
                        return True
                    elif q_state in ("Aborted", "Error"):
                        print(f"⚠️ Mission {q_state}")
                        delete_position(pos_guid)
                        return False
                except Exception:
                    pass

            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n⛔ Đã hủy!")
            delete_position(pos_guid)
            return False
        except Exception as e:
            print(f"  Lỗi: {e}")
            time.sleep(2.0)

    print("⏰ Timeout!")
    delete_position(pos_guid)
    return False


def main():
    if len(sys.argv) >= 2:
        ten_input = " ".join(sys.argv[1:]).strip()
    else:
        hien_danh_sach()
        print("  pos      ->  Xem vị trí robot hiện tại")
        ten_input = input("Nhập tên điểm đích (hoặc 'pos'): ").strip()

    if ten_input.lower() == "pos":
        show_position()
        sys.exit(0)

    ten_chuan = chuan_hoa(ten_input)
    if ten_chuan not in DIEM:
        print(f"❌ Điểm '{ten_input}' không hợp lệ.")
        hien_danh_sach()
        sys.exit(1)

    ok = go_to(ten_chuan)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
