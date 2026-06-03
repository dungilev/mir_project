import tkinter as tk
import math
import random

class SearchingFace:
    def __init__(self, root):
        self.root = root
        self.root.title("MiR Searching Face")
        self.root.geometry("800x480")
        self.root.configure(bg='black')
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.canvas = tk.Canvas(root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.width = 800
        self.height = 480
        
        # Dimensions
        self.sclera_r = 130  # Lòng trắng
        self.pupil_r = 55    # Lòng đen
        self.eye_gap = 350   # Khoảng cách giữa 2 mắt (tăng lên)
        
        self.cx = self.width / 2
        self.cy = self.height / 2
        
        # Pupil positions
        self.px = 0.0
        self.py = 0.0
        self.target_px = 0.0
        self.target_py = 0.0
        
        # Blink animation
        self.blink_scale = 1.0
        self.target_blink = 1.0
        
        self.root.bind('<Configure>', self.on_resize)
        self.loop()
        self.behavior_loop()

    def draw(self):
        self.canvas.delete("all")
        
        # Interpolate pupil movement for smooth darting eyes (like a human looking around)
        speed = 0.35
        self.px += (self.target_px - self.px) * speed
        self.py += (self.target_py - self.py) * speed
        
        # Interpolate blink
        self.blink_scale += (self.target_blink - self.blink_scale) * 0.4
        
        lx = self.cx - self.eye_gap / 2
        rx = self.cx + self.eye_gap / 2
        
        # Eye height depends on blink_scale
        h = self.sclera_r * self.blink_scale
        if h < 2: h = 2
        
        # Draw Cyan Sclera
        self.canvas.create_oval(lx - self.sclera_r, self.cy - h, lx + self.sclera_r, self.cy + h, fill='#00f3ff', outline='')
        self.canvas.create_oval(rx - self.sclera_r, self.cy - h, rx + self.sclera_r, self.cy + h, fill='#00f3ff', outline='')
        
        # Draw Black Pupils (Only if eyes are mostly open)
        if self.blink_scale > 0.2:
            # Constrain pupil so it stays inside the white part
            max_d = self.sclera_r - self.pupil_r - 10
            d = math.hypot(self.px, self.py)
            if d > max_d:
                ratio = max_d / d
                cx_p = self.px * ratio
                cy_p = self.py * ratio
            else:
                cx_p = self.px
                cy_p = self.py
                
            # Left pupil
            self.canvas.create_oval(lx + cx_p - self.pupil_r, self.cy + cy_p - self.pupil_r*self.blink_scale, 
                                    lx + cx_p + self.pupil_r, self.cy + cy_p + self.pupil_r*self.blink_scale, 
                                    fill='black')
            # Right pupil
            self.canvas.create_oval(rx + cx_p - self.pupil_r, self.cy + cy_p - self.pupil_r*self.blink_scale, 
                                    rx + cx_p + self.pupil_r, self.cy + cy_p + self.pupil_r*self.blink_scale, 
                                    fill='black')
                                    
            # Little white reflection (đốm sáng trong mắt) to make it look alive
            hl_r = self.pupil_r * 0.25
            hl_dx = cx_p + self.pupil_r * 0.25
            hl_dy = cy_p - self.pupil_r * 0.25
            
            # Highlight size shrinks when blinking
            hl_rh = hl_r * self.blink_scale
            
            self.canvas.create_oval(lx + hl_dx - hl_r, self.cy + hl_dy - hl_rh,
                                    lx + hl_dx + hl_r, self.cy + hl_dy + hl_rh,
                                    fill='white', outline='')
            self.canvas.create_oval(rx + hl_dx - hl_r, self.cy + hl_dy - hl_rh,
                                    rx + hl_dx + hl_r, self.cy + hl_dy + hl_rh,
                                    fill='white', outline='')

    def loop(self):
        self.draw()
        self.root.after(30, self.loop)

    def on_resize(self, event):
        if event.widget == self.root:
            self.width, self.height = event.width, event.height
            self.cx, self.cy = self.width / 2, self.height / 2
            
    def blink(self):
        self.target_blink = 0.0
        def open_eyes():
            self.target_blink = 1.0
        self.root.after(150, open_eyes)

    def behavior_loop(self):
        # 15% chance to just blink
        if random.random() < 0.15:
            self.blink()
            
        # Searching logic (Đảo mắt tìm người)
        max_d = self.sclera_r - self.pupil_r
        
        # Pick a looking angle (frequent horizontal scans, some up/down)
        # Angles: 0 (right), pi (left), pi/2 (down), -pi/2 (up), etc.
        angle = random.choice([0, math.pi, math.pi/4, 3*math.pi/4, -math.pi/4, -3*math.pi/4, math.pi/2, -math.pi/2])
        angle += random.uniform(-0.4, 0.4) # Thêm nhiễu ngẫu nhiên
        
        # How far to look (usually looking far when searching)
        dist = random.uniform(max_d * 0.6, max_d)
        
        # 20% chance to look straight ahead
        if random.random() < 0.2:
            dist = 0
            
        self.target_px = dist * math.cos(angle)
        self.target_py = dist * math.sin(angle)
        
        # Mắt đảo liên tục và nhanh khi đang tìm kiếm
        self.root.after(random.randint(500, 1500), self.behavior_loop)

if __name__ == "__main__":
    root = tk.Tk()
    # root.attributes('-fullscreen', True)
    app = SearchingFace(root)
    root.mainloop()
