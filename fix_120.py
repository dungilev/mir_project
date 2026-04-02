with open("/home/tuanminh/mir_project/src/mir_robot/tm/navigationcacdiem.py", "r") as f:
    text = f.read()

bad_snippet = """    try:
            r = requests.get(f"{API_URL}/missions", headers=headers, timeout=5)
            if r.status_code == 200:
                print(f"  REST API OK ({user}, sha256)")
                return headers
        except Exception:"""

good_snippet = """        try:
            r = requests.get(f"{API_URL}/missions", headers=headers, timeout=5)
            if r.status_code == 200:
                print(f"  REST API OK ({user}, sha256)")
                return headers
        except Exception:"""
        
text = text.replace(bad_snippet, good_snippet)
with open("/home/tuanminh/mir_project/src/mir_robot/tm/navigationcacdiem.py", "w") as f:
    f.write(text)
