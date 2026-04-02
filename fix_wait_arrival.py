import re

with open("/home/tuanminh/mir_project/src/mir_robot/tm/navigationcacdiem.py", "r") as f:
    text = f.read()

# We need to find `def wait_arrival` and wrap the part from `deadline = ...` to `return False`

start_marker = "    deadline  = time.time() + timeout"
end_marker = "    return False"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker, start_idx) + len(end_marker) + 1

if start_idx != -1 and end_idx != -1:
    body = text[start_idx:end_idx]
    
    # indent body by 4 spaces
    indented = "\n".join(["    " + line if line.strip() else line for line in body.split("\n")[:-1]]) + "\n"
    
    new_code = "    try:\n" + indented + "    finally:\n        robot.unhook(pose_cb)\n        robot.unhook(status_cb)\n"
    
    text = text[:start_idx] + new_code + text[end_idx:]
    with open("/home/tuanminh/mir_project/src/mir_robot/tm/navigationcacdiem.py", "w") as f:
        f.write(text)
    print("Fixed!")
else:
    print("Not found")

