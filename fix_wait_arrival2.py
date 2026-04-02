with open("/home/tuanminh/mir_project/src/mir_robot/tm/navigationcacdiem.py", "r") as f:
    text = f.read()

start_idx = text.find("def wait_arrival(")
end_idx = text.find("def handle_command(", start_idx)

wait_arrival_code = text[start_idx:end_idx]

import re
# Replace ALL returns inside the function to unhook first!
def replace_return(match):
    indent = match.group(1)
    ret_val = match.group(2)
    # We don't want to unhook if it's returning from a nested callback... but these inner callbacks don't return anything.
    # So we're safe.
    return f"{indent}robot.unhook(pose_cb)\n{indent}robot.unhook(status_cb)\n{indent}return {ret_val}"

wait_arrival_code_fixed = re.sub(r"([ \t]+)return (.*?)(?=\n)", replace_return, wait_arrival_code)

text = text[:start_idx] + wait_arrival_code_fixed + text[end_idx:]

with open("/home/tuanminh/mir_project/src/mir_robot/tm/navigationcacdiem.py", "w") as f:
    f.write(text)

