# 🎯 Hit The Loser - Autonomous Deep Reinforcement Learning Agent

## Proje Özeti
Bu proje, dışarıdan hiçbir API desteği olmayan Flash tabanlı "Hit The Loser" oyununu ekran görüntüsü işleyerek ve farenin hareketlerini simüle ederek otonom şekilde oynayabilen bir yapay zeka ajanı geliştirmeyi amaçlamaktadır. Projede **Proximal Policy Optimization (PPO)** algoritması kullanılmış ve eğitim sürelerini optimize etmek için **Sim-to-Real (Simülasyondan Gerçeğe)** mimarisi inşa edilmiştir.

## 🛠️ Kullanılan Teknolojiler
*   **Makine Öğrenmesi & RL:** `Stable-Baselines3`, `Gymnasium`
*   **Bilgisayarlı Görme:** `OpenCV`, `mss`
*   **Sistem Kontrolü & Modelleme:** `pynput`, `NumPy`, `Math`
*   **Tersine Mühendislik:** `JPEXS Free Flash Decompiler`

## 🚀 Karşılaşılan Zorluklar ve Mühendislik Çözümleri

### 1. Sistem Modelleme ve Tersine Mühendislik (Reverse Engineering)
**Sorun:** Yapay zeka modelinin sınırları bilmeden gereksiz büyük atışlar yapması ve oyunun fizik motoruna uyum sağlayamaması.
**Çözüm:** Oyunun `.swf` dosyası deşifre edildi. Atış gücünün maksimum 30 birim olduğu, hedefin doğrusal değil merkez etrafında dairesel/açısal döndüğü ve sekme kayıp katsayısının (-0.75) olduğu tespit edilerek algoritmanın gözlem uzayına (Observation Space) entegre edildi.

### 2. Dijital İkiz (Digital Twin) ve Sim-to-Real Mimarisi
**Sorun:** Gerçek zamanlı bilgisayarlı görme (Computer Vision) ve fare kontrolü ile model eğitmenin, her atış için saniyeler harcaması nedeniyle uygulanabilir olmaması.
**Çözüm:** Ekran okuma ortamı tamamen devreden çıkarılarak, tersine mühendislikle elde edilen fizik formülleriyle oyunun %100 matematiksel bir simülasyonu (`sim_env.py`) kuruldu. Günler sürecek deneme-yanılma (trial-and-error) süreci, işlemci gücüyle saniyeler içinde çözülüp eğitilen ağırlıklar gerçek dünyaya aktarıldı.

### 3. Geometrik Deformasyon ve Görüntü Standardizasyonu
**Sorun:** Farklı ekran boyutları veya manuel alan seçimleri (ROI) nedeniyle sekme açılarının bozulması ve yapay zekanın isabet oranının düşmesi.
**Çözüm:** Modelin giriş katmanına bir "Görüntü Lensi" yazıldı. Kullanıcı ekrandan yamuk veya farklı boyutta bir alan seçse bile, sistem bunu otomatik olarak referans bir kareye dönüştürerek sekme açılarındaki fiziksel sapmaları (geometrik deformasyon) sıfırladı. 

## 🧠 Model Başarısı
"God Mode" olarak adlandırılan son ince ayar (fine-tuning) aşamasında, modelin sinir ağları genişletilmiş (`net_arch=[512, 512]`) ve öğrenme katsayısı düşürülerek milimetrik kararlar alması sağlanmıştır. Model, ortadaki fiziksel engeli aşmak için doğrudan atış yapmak yerine bilardo mantığıyla duvar bantlarını kullanarak sekme hesaplamaları yapabilmektedir.
