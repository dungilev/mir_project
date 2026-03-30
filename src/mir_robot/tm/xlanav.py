#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import time
import signal
import os

processes = []

def signal_handler(sig, frame):
    print("\n[Main] Nhan tin hieu dung. Dang the hien cac tien trinh (Ctrl+C)...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
            p.wait()
    print("[Main] Da thoat an toan.")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    # script_dir: .../mir_project/src/mir_robot/tm
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # project_root: .../mir_project
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    start_sh_path = os.path.join(project_root, "start.sh")
    
    # 1. khoangcach3d.py can chay tren host de bat camera (va xu ly anh GPU neu co)
    cmd_cam = [start_sh_path, "run-host", "khoangcach3d.py"]
    
    # 2. navigationnguoi.py nen chay ben TRONG container ROS (de co library tieu chuan move_base_msgs, tf, ...)
    cmd_nav = [start_sh_path, "run", "navigationnguoi.py"]
    
    print("=====================================================")
    print("[Main] Khoi dong khoangcach3d.py (TREN HOST)...")
    print("=====================================================")
    p1 = subprocess.Popen(cmd_cam, cwd=project_root)
    processes.append(p1)
    
    time.sleep(3.0) # Cho file camera khoi dong len
    
    print("=====================================================")
    print("[Main] Khoi dong navigationnguoi.py (TRONG CONTAINER)...")
    print("=====================================================")
    p2 = subprocess.Popen(cmd_nav, cwd=project_root)
    processes.append(p2)
    
    print("\n[Main] Tat ca he thong dang chay. Nhan Ctrl+C de dung toan bo.\n")
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
