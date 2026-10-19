import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class SimulatedLoserEnv(gym.Env):
    """
    Flash oyununun 500x500 piksellik matematiksel klonu.
    Katı SWF kuralları ve Zeki Ödül Sistemi (Engel Cezası) içerir.
    """
    def __init__(self):
        super(SimulatedLoserEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        self.game_size = 500.0
        self.obstacle_x = 250.0
        self.obstacle_y = 250.0
        self.obstacle_r = 76.85
        self.ball_r = 15.0
        self.target_r = 15.0
        self.max_pull = 30.0
        
        self.max_health = 10
        self.current_health = self.max_health
        self.current_step = 0
        
        # Orijinal SWF Katı Kuralları (Keskin nişancı hassasiyeti için sabitlendi)
        self.friction = 1.02
        self.bounce_loss = 0.75
        self.target_speed = 0.05
        
        self.w_x, self.w_y = 250.0, 400.0
        self.t_angle = 0.0
        self.t_x, self.t_y = 250.0, 100.0
        self.t_vx, self.t_vy = 0.0, 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_health = self.max_health
        self.current_step = 0
        
        self.w_x = np.random.uniform(100, 400)
        self.w_y = np.random.uniform(350, 450)
        self.t_angle = np.random.uniform(0, 2 * math.pi)
        self._update_target_position()
        
        return self._get_obs(), {}

    def _update_target_position(self):
        orbit_radius = 140.0
        self.t_x = self.obstacle_x + math.cos(self.t_angle) * orbit_radius
        self.t_y = self.obstacle_y + math.sin(self.t_angle) * orbit_radius
        self.t_vx = -math.sin(self.t_angle) * orbit_radius * (self.target_speed * 5)
        self.t_vy = math.cos(self.t_angle) * orbit_radius * (self.target_speed * 5)

    def _get_obs(self):
        return np.array([
            self.w_x / self.game_size,
            self.w_y / self.game_size,
            self.t_x / self.game_size,
            self.t_y / self.game_size,
            np.clip(self.t_vx / 50.0, -1.0, 1.0),
            np.clip(self.t_vy / 50.0, -1.0, 1.0),
            self.obstacle_x / self.game_size,
            self.obstacle_y / self.game_size
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        pull_x = action[0] * self.max_pull
        pull_y = action[1] * self.max_pull
        
        distance = math.hypot(pull_x, pull_y)
        if distance > self.max_pull:
            pull_x = (pull_x / distance) * self.max_pull
            pull_y = (pull_y / distance) * self.max_pull
            
        w_vx = -pull_x
        w_vy = -pull_y
        
        dist_before = math.hypot(self.w_x - self.t_x, self.w_y - self.t_y)
        target_hit = False
        hit_obstacle = False  # YENİ: Engele çarpma takibi
        
        while True:
            self.t_angle += self.target_speed 
            self._update_target_position()
            
            self.w_x += w_vx
            self.w_y += w_vy
            
            w_vx /= self.friction
            w_vy /= self.friction
            
            if self.w_x < self.ball_r:
                self.w_x = self.ball_r
                w_vx = -w_vx * self.bounce_loss
            elif self.w_x > self.game_size - self.ball_r:
                self.w_x = self.game_size - self.ball_r
                w_vx = -w_vx * self.bounce_loss
                
            if self.w_y < self.ball_r:
                self.w_y = self.ball_r
                w_vy = -w_vy * self.bounce_loss
            elif self.w_y > self.game_size - self.ball_r:
                self.w_y = self.game_size - self.ball_r
                w_vy = -w_vy * self.bounce_loss
                
            dist_to_center = math.hypot(self.w_x - self.obstacle_x, self.w_y - self.obstacle_y)
            if dist_to_center < (self.obstacle_r + self.ball_r):
                hit_obstacle = True # YENİ: Engele çarptığını kaydet
                nx = (self.w_x - self.obstacle_x) / dist_to_center
                ny = (self.w_y - self.obstacle_y) / dist_to_center
                
                overlap = (self.obstacle_r + self.ball_r) - dist_to_center
                self.w_x += nx * overlap
                self.w_y += ny * overlap
                
                dot_product = w_vx * nx + w_vy * ny
                w_vx = (w_vx - 2 * dot_product * nx) * self.bounce_loss
                w_vy = (w_vy - 2 * dot_product * ny) * self.bounce_loss

            dist_to_target = math.hypot(self.w_x - self.t_x, self.w_y - self.t_y)
            if dist_to_target < (self.ball_r + self.target_r):
                target_hit = True
                break
                
            if abs(w_vx) < 0.1 and abs(w_vy) < 0.1:
                break
                
        reward = 0.0
        terminated = False
        truncated = False
        
        # YENİ: ZEKİ ÖDÜL MEKANİZMASI
        if target_hit:
            self.current_health = self.max_health
            reward = 100.0
            self.t_angle += math.pi / 2
            self._update_target_position()
        else:
            self.current_health -= 1
            dist_after = math.hypot(self.w_x - self.t_x, self.w_y - self.t_y)
            
            if hit_obstacle:
                reward = -15.0  # Engele boşuna vurursa ağır ceza
            elif dist_after < dist_before:
                reward = 10.0   # Yaklaşırsa daha yüksek ödül
            else:
                reward = -5.0   # Uzaklaşırsa standart ceza
                
            if self.current_health <= 0:
                reward = -50.0
                terminated = True
                
        if self.current_step >= 50:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}