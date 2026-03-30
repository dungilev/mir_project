import sys

with open('/tmp/nav.remote.py', 'r') as f:
    content = f.read()

import re

# We will replace the DIEM block
old_diem = """DIEM = {
    "bep":   {"x": 5.5,  "y": 17.05, "qz": 0.707, "qw": 0.707, "arrive_dist": 1.0},
    "ban 1": {"x": 6.900,  "y": 19.150, "qz": 0, "qw": 1, "arrive_dist": 0.9},
    "ban 2": {"x": 6.200,  "y": 14.700, "qz": 0, "qw": 1, "arrive_dist": 0.9},
    "ban 3": {"x": 14.550, "y": 17.500, "qz": 0, "qw": 1, "arrive_dist": 0.9},
    "ban 4": {"x": 14.800, "y": 14.250, "qz": 0, "qw": 1, "arrive_dist": 0.9},
}"""

new_diem = """DIEM = {
    "bep":   {"x": 5.5,  "y": 17.05, "qz": 0.707, "qw": 0.707, "arrive_dist": 1.0},
    "ban 1": {"x": 6.900,  "y": 19.150, "qz": 0, "qw": 1, "arrive_dist": 0.9},
    "ban 2": {"x": 6.200,  "y": 14.700, "qz": 0, "qw": 1, "arrive_dist": 0.9},
    "ban 3": {"x": 14.550, "y": 17.700, "qz": 0, "qw": 1, "arrive_dist": 0.9},
    "ban 4": {"x": 14.800, "y": 14.250, "qz": 0, "qw": 1, "arrive_dist": 0.9},
}"""

content = content.replace(old_diem, new_diem)

with open('src/mir_robot/tm/navigationcacdiem.py', 'w') as f:
    f.write(content)
