from stable_baselines3 import PPO
from rl_env import HitTheLoserEnv
import time

env = HitTheLoserEnv()

print(">>> Yapay zeka beyni yükleniyor...")
model = PPO.load("terminator_final_brain")

obs, info = env.reset()
print(">>> Şov başlıyor! Lütfen fareyi serbest bırakın.")

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(">>> Bölüm bitti veya can tükendi. Sistem sıfırlanıyor...")
        time.sleep(2)  
        obs, info = env.reset()