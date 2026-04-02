import navigationcacdiem as nav
import requests
headers = nav.api_login()
requests.delete(f"{nav.API_URL}/mission_queue", headers=headers, timeout=5)
print("Queue cleared.")
