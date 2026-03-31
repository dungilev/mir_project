import navigationcacdiem as nav
import requests, json
headers = nav.api_login()
r = requests.get(f"{nav.API_URL}/actions/sound", headers=headers)
print(json.dumps(r.json(), indent=2))
