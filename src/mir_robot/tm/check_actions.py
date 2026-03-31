import navigationcacdiem as nav
import requests, json
headers = nav.api_login()
r = requests.get(f"{nav.API_URL}/actions", headers=headers)
if r.status_code == 200:
    for a in r.json():
        if "sound" in a.get("action_type","").lower() or "sound" in a.get("name","").lower() or "play" in a.get("action_type","").lower():
            print(json.dumps(a, indent=2))
