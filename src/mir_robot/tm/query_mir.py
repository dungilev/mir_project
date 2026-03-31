import requests
import base64
MIR_IP = "192.168.12.20"
API_URL = f"http://{MIR_IP}/api/v2.0.0"
headers = {"Authorization": "Basic " + base64.b64encode(b"distributor:62b8471ff0258cbaca0344b1c7dc7e76aaed1c9b679b3cc1f574d752dd1df538").decode("ascii")}
try:
    print(requests.get(f"{API_URL}/actions", headers=headers, timeout=5).json())
except Exception as e:
    print(e)
