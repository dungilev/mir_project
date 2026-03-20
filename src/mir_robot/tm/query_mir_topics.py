#!/usr/bin/env python3
import json, time, sys
sys.path.insert(0, '/root/catkin_ws/devel/lib/python3/dist-packages')
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
from mir_driver.rosbridge import RosbridgeSetup

robot = RosbridgeSetup('192.168.0.177', 9090)
time.sleep(1.5)
if not robot.is_connected():
    print('NOT CONNECTED')
    sys.exit(1)
print('CONNECTED')

resp = robot.callService('/rosapi/topics', msg={})
topics = sorted(resp.get('topics', []))
print(f'Total: {len(topics)}')

print('\n=== move_base topics ===')
for t in topics:
    if 'move_base' in t:
        r2 = robot.callService('/rosapi/topic_type', msg={'topic': t})
        print(f'  {t}  [{r2.get("type","")}]')

print('\n=== robot mode/state ===')
for t in topics:
    if 'robot_mode' in t or 'robot_state' in t:
        r2 = robot.callService('/rosapi/topic_type', msg={'topic': t})
        print(f'  {t}  [{r2.get("type","")}]')

# Subscribe to /move_base/status to see latest status
results = []
def on_status(msg):
    results.append(msg)
robot.subscribe('/move_base/status', on_status)
time.sleep(2)
print(f'\n=== /move_base/status ({len(results)} messages) ===')
if results:
    latest = results[-1]
    statuses = latest.get('status_list', [])
    print(f'  Status list length: {len(statuses)}')
    for s in statuses[-3:]:
        print(f'  goal_id: {s.get("goal_id",{}).get("id","")}  status: {s.get("status","")}  text: {s.get("text","")}')

print('\nDONE')
