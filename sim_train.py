from stable_baselines3 import PPO
from sim_env import SimulatedLoserEnv
from stable_baselines3.common.callbacks import CheckpointCallback

env = SimulatedLoserEnv()

checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='./models/', name_prefix='terminator_god_mode')

# --- ALTIN VURUŞ (GOD MODE) AYARLARI ---
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.00005,     # [KIRITIK DEĞİŞİKLİK] Öğrenme hızını çok düşürdük. Artık büyük adımlar atmayacak, milimetrik "ince ayar" yapacak.
    n_steps=4096,              # [ARTIRILDI] Karar vermeden önce eskisinin 2 katı fazla geleceği düşünecek.
    batch_size=512,            # [ARTIRILDI] Geçmiş tecrübelerini çok daha büyük paketler halinde analiz edecek.
    ent_coef=0.01,             # [DÜŞÜRÜLDÜ] Artık "çılgın yeni şeyler deneme" modunu kapattık. Bildiği doğruları keskinleştirmesini istiyoruz.
    gamma=0.995,               # [ARTIRILDI] Sadece ilk sekmeyi değil, duvardan duvara 3. sekmeyi bile önceden planlayacak.
    policy_kwargs=dict(net_arch=[512, 512]), # [DEVASA BEYİN] 256'lık ağları 512'ye çıkardık. Açısal hesaplama kapasitesi ikiye katlandı!
    device="auto"
)

print(">>> [SİMÜLASYON BAŞLIYOR] Model 'Tanrı Modunda' milimetrik ince ayar ile eğitiliyor...")

# Eğitim süresini devasa bir seviyeye çıkardık (5 Milyon adım)
model.learn(total_timesteps=5000000, callback=checkpoint_callback)

model.save("terminator_final_brain")
print(">>> [EĞİTİM BİTTİ] Kusursuz beyin başarıyla kaydedildi!")