#!/usr/bin/env python3
"""Đưa MiR robot về trạng thái READY (state=3) qua REST API."""
import sys
import base64

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

MIR_IP = "192.168.0.177"
AUTH = base64.b64encode(b"distributor:distributor").decode()
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Basic {AUTH}"
}
API = f"http://{MIR_IP}/api/v2.0.0"

# Kiểm tra state hiện tại
r = requests.get(f"{API}/status", headers=HEADERS, timeout=5)
data = r.json()
print(f"State hiện tại: {data['state_id']} ({data['state_text']})")
print(f"Mode: {data['mode_id']} ({data['mode_text']})")
print(f"Mission: {data.get('mission_text', 'N/A')}")

if data["state_id"] == 3:
    print("✅ Robot ĐÃ ở trạng thái READY!")
    sys.exit(0)

# Đưa về READY
r = requests.put(f"{API}/status", headers=HEADERS, json={"state_id": 3}, timeout=5)
if r.status_code == 200:
    print("✅ Đã chuyển robot về READY!")
else:
    print(f"❌ Lỗi: {r.status_code} - {r.text}")
    sys.exit(1)
