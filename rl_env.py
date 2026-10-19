import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import time
import math
from pynput.mouse import Button, Controller as MouseController

class HitTheLoserEnv(gym.Env):
    def __init__(self):
        super(HitTheLoserEnv, self).__init__()
        
        self.mouse = MouseController()
        self.game_window = self.setup_game_window()
        
        self.lower_white = np.array([0, 0, 200], dtype=np.uint8)
        self.upper_white = np.array([180, 30, 255], dtype=np.uint8)
        self.lower_orange = np.array([10, 100, 100], dtype=np.uint8)
        self.upper_orange = np.array([30, 255, 255], dtype=np.uint8)
        
        self.sct = mss.mss()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        self.max_health = 10
        self.current_health = self.max_health
        self.current_step = 0
        
        self.last_t_pos = None
        self.last_time = time.time()
        self.t_velocity = (0.0, 0.0)

    def setup_game_window(self):
            print("\n>>> OYUN EKRANINA GEÇMEK İÇİN 3 SANİYENİZ VAR... <<<")
            print(">>> Lütfen farenizle oyunun İÇ ÇERÇEVESİNİ seçin. (Yamuk çizseniz bile sistem KARE yapacaktır) <<<")
            time.sleep(3)
            
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                scr = np.array(sct.grab(monitor))
                scr_bgr = cv2.cvtColor(scr, cv2.COLOR_RGBA2BGR)
                
                # Serbest çizim aracı (ROI)
                roi = cv2.selectROI("AI Oyun Alani Secimi", scr_bgr, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("AI Oyun Alani Secimi")
                
                # [KUSURSUZLUK DÜZELTMESİ]: Kullanıcı ne kadar dikdörtgen çizerse çizsin, 
                # genişliği baz alarak alanı her zaman KUSURSUZ BİR KARE'ye dönüştür!
                base_size = int(roi[2]) # Çizdiğin genişliği referans alıyoruz
                
                selected_roi = {
                    "left": int(roi[0]), 
                    "top": int(roi[1]), 
                    "width": base_size, 
                    "height": base_size # Yüksekliği zorla genişliğe eşitledik (Örn: 600x600)
                }
                print(f">>> Alan kilitlendi ve otomatik olarak KARE yapıldı: {selected_roi}")
                return selected_roi

    def get_ball_coordinates(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if 20 < area < 3000:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None

    def get_state(self):
        screenshot = self.sct.grab(self.game_window)
        frame = np.array(screenshot)
        
        # [YENİ] Görüntüyü boyutu ne olursa olsun zorla 500x500'e presle
        frame = cv2.resize(frame, (500, 500))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask_white = cv2.inRange(hsv_frame, self.lower_white, self.upper_white)
        mask_target = cv2.inRange(hsv_frame, self.lower_orange, self.upper_orange)
        
        w_pos = self.get_ball_coordinates(mask_white)
        t_pos = self.get_ball_coordinates(mask_target)
        
        if w_pos and t_pos:
            if math.hypot(w_pos[0] - t_pos[0], w_pos[1] - t_pos[1]) < 30: 
                w_pos = None 
                
        current_time = time.time()
        if t_pos:
            if self.last_t_pos:
                dt = current_time - self.last_time
                if 0 < dt < 0.5:
                    vx = (t_pos[0] - self.last_t_pos[0]) / dt
                    vy = (t_pos[1] - self.last_t_pos[1]) / dt
                    self.t_velocity = (self.t_velocity[0]*0.5 + vx*0.5, self.t_velocity[1]*0.5 + vy*0.5)
            self.last_t_pos = t_pos
            self.last_time = current_time
        else:
            self.t_velocity = (0.0, 0.0)
            
        return w_pos, t_pos
        
    def _create_obs(self, w_pos, t_pos):
        if w_pos and t_pos:
            return np.array([
                w_pos[0] / 500.0,
                w_pos[1] / 500.0,
                t_pos[0] / 500.0,
                t_pos[1] / 500.0,
                np.clip(self.t_velocity[0] / 1000.0, -1.0, 1.0), 
                np.clip(self.t_velocity[1] / 1000.0, -1.0, 1.0),
                0.5, 
                0.5  
            ], dtype=np.float32)
        return np.zeros(8, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_health = self.max_health
        self.current_step = 0
        
        w_pos, t_pos = self.get_state()
        wait_counter = 0
        while (not w_pos or not t_pos) and wait_counter < 30: 
            time.sleep(0.1)
            w_pos, t_pos = self.get_state()
            wait_counter += 1
            
        return self._create_obs(w_pos, t_pos), {}

    def smooth_drag(self, start_x, start_y, end_x, end_y):
        self.mouse.position = (start_x, start_y)
        time.sleep(0.05) 
        self.mouse.press(Button.left)
        time.sleep(0.02) 
        
        steps = 5
        dx = (end_x - start_x) / steps
        dy = (end_y - start_y) / steps
        
        for i in range(1, steps + 1):
            self.mouse.position = (int(start_x + dx * i), int(start_y + dy * i))
            time.sleep(0.01)
            
        time.sleep(0.05) 
        self.mouse.release(Button.left)

    def step(self, action):
        self.current_step += 1
        w_pos, t_pos_before = self.get_state()
        
        if not w_pos or not t_pos_before:
            time.sleep(0.2) 
            return np.zeros(8, dtype=np.float32), -1.0, False, False, {}
            
        pull_x = action[0] * 30
        pull_y = action[1] * 30
        
        # [YENİ] Seçtiğin oyun ekranının gerçek büyüklüğüne göre farenin atış gücünü esnet
        scale_x = self.game_window["width"] / 500.0
        scale_y = self.game_window["height"] / 500.0
        
        screen_w_x = self.game_window["left"] + int(w_pos[0] * scale_x)
        screen_w_y = self.game_window["top"] + int(w_pos[1] * scale_y)
        
        target_pull_x = screen_w_x + int(pull_x * scale_x)
        target_pull_y = screen_w_y + int(pull_y * scale_y)
        
        self.smooth_drag(screen_w_x, screen_w_y, target_pull_x, target_pull_y)
        
        time.sleep(0.6)
        
        w_pos_after, t_pos_after = self.get_state()
        
        target_hit = False
        if not t_pos_after:
            target_hit = True
            for _ in range(3):
                time.sleep(0.05)
                w_check, t_check = self.get_state()
                if t_check: 
                    target_hit = False
                    t_pos_after = t_check
                    w_pos_after = w_check
                    break
                    
        reward = 0.0
        terminated = False
        truncated = False
        
        if target_hit:
            self.current_health = self.max_health
            reward = 100.0
            print(f">>> [BAŞARILI] Hedef Vuruldu! (+100 Puan) | Adım: {self.current_step}")
            for _ in range(15):
                time.sleep(0.1)
                w_check, t_check = self.get_state()
                if t_check:
                    t_pos_after = t_check
                    w_pos_after = w_check
                    break
                    
        elif w_pos_after and t_pos_after:
            self.current_health -= 1
            print(f"[ISKA] Kalan Can: {self.current_health}")
            if self.current_health <= 0:
                reward = -50.0
                terminated = True
                print(">>> [GAME OVER] Botun canı bitti!")
                
        if self.current_step >= 50:
            truncated = True
            
        return self._create_obs(w_pos_after, t_pos_after), reward, terminated, truncated, {}