import pandas as pd
import numpy as np

# ── VERİYİ OKUYOR (Kaggle'dan indirdiğin gerçek CSV)
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ── ADIM 1: VERİYE İLK BAKIŞ
# Kaç satır, kaç sütun var?
print("Satır sayısı :", df.shape[0])   # shape[0] = satır sayısı
print("Sütun sayısı :", df.shape[1])   # shape[1] = sütun sayısı

# İlk 5 satırı göster — veri nasıl görünüyor?
print("\nİlk 5 müşteri:")
print(df.head())

# Sütun isimlerini listele — hangi bilgiler var?
print("\nSütunlar:")
print(df.columns.tolist())

#customerID → müşteri numarası
# gender → cinsiyet
# SeniorCitizen → yaşlı mı?
# Partner, Dependents → eş/bağımlı var mı?
# tenure → kaç aydır müşteri
# PhoneService, MultipleLines → telefon hizmetleri
# InternetService, OnlineSecurity, OnlineBackup → internet hizmetleri
# DeviceProtection, TechSupport → destek hizmetleri
# StreamingTV, StreamingMovies → yayın hizmetleri
# Contract → sözleşme tipi
# PaperlessBilling, PaymentMethod → ödeme bilgileri
# MonthlyCharges, TotalCharges → ücretler
# Churn → ayrıldı mı? ← TAHMİN ETMEYE ÇALIŞTIĞIMIZ 

# ── ADIM 2: EKSİK VERİ VAR MI?
# Her sütunda kaç tane boş/eksik değer var?
print("\nEksik değer sayıları:")
print(df.isnull().sum())
# isnull() = "bu hücre boş mu?" diye sorar, True/False döner
# .sum()   = True olanları toplar, yani kaç tane boş var sayar

# ── ADIM 3: GİZLİ EKSİK VERİ KONTROLÜ
# TotalCharges sayı olmalı ama bazı satırlar boş karakter içeriyor
# Sayıya çevirmeye çalışıyoruz, olmayanlar NaN (boş) olur
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# to_numeric = sayıya çevir demek
# errors='coerce' = sayıya çevrilemeyen hücreyi NaN (boş) yap

# Şimdi tekrar kontrol et
print("\nTotalCharges'taki gizli eksik değerler:")
print(df['TotalCharges'].isnull().sum())
# Kaç tane NaN çıktı?

# Bu boş satırları medyan ile doldur
medyan = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(medyan)
# median() = ortanca değer — ortalamadan daha sağlıklı
# fillna  = boş olanları doldur
# inplace = d
#İlk bakısta eksik veri yoktu ama TotalCharges sütununda 11 satır boş karakter.
#Bunu sayıya çevirip medyanla doldurduk.

# ── ADIM 4: CHURN DAĞILIMI
print("\n--- CHURN DAĞILIMI ---")
churn_dagilim = df['Churn'].value_counts()

print(f"Kalan müşteri  (No) : {churn_dagilim['No']:,} kişi")
print(f"Ayrılan müşteri(Yes): {churn_dagilim['Yes']:,} kişi")

oran = churn_dagilim['Yes'] / len(df) * 100
print(f"Ayrılma oranı       : %{oran:.1f}")

# Önce oranı gör, sonra karar ver
if oran < 35:
    print("⚠️  Dengesiz veri! SMOTE ile dengeleyeceğiz.")
else:
    print("✅ Veri dengeli, SMOTE gerekmeyebilir.")
#SMOTE = Synthetic Minority Over-sampling Technique 
#Azınlık sınıfı için yapay örnek üretme yöntemi

# ── ADIM 5: KİMLER DAHA ÇOK AYRILIYOR?

# SORU 1: Sözleşme tipine göre
print("\n--- SÖZLEŞME TİPİNE GÖRE AYRILMA ORANI ---")
sozlesme = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).round(1)
# groupby = Contract sütununa göre grupla
# lambda x = her grup için küçük bir fonksiyon çalıştır
# (x == 'Yes').mean() = Yes olanların oranını al
# * 100 = yüzdeye çevir
print(sozlesme)

# SORU 2: İnternet hizmetine göre
print("\n--- İNTERNET HİZMETİNE GÖRE AYRILMA ORANI ---")
internet = df.groupby('InternetService')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).round(1)
print(internet)

# SORU 3: Yeni müşteri mi eski müşteri mi daha riskli?
print("\n--- YENİ vs ESKİ MÜŞTERİ ---")
yeni = (df[df['tenure'] <= 6]['Churn'] == 'Yes').mean() * 100
# df[df['tenure'] <= 6] = sadece ilk 6 aydaki müşterileri filtrele
eski = (df[df['tenure'] > 6]['Churn'] == 'Yes').mean() * 100
print(f"İlk 6 aydaki müşteri ayrılma oranı : %{yeni:.1f}")
print(f"6 aydan eski müşteri ayrılma oranı  : %{eski:.1f}")

# SORU 4: Yaşlı mı genç mi?
print("\n--- YAŞLI vs GENÇ MÜŞTERİ ---")
yasli = (df[df['SeniorCitizen'] == 1]['Churn'] == 'Yes').mean() * 100
genc  = (df[df['SeniorCitizen'] == 0]['Churn'] == 'Yes').mean() * 100
print(f"Yaşlı müşteri ayrılma oranı : %{yasli:.1f}")
print(f"Genç müşteri ayrılma oranı  : %{genc:.1f}")

print("\n✅ EDA TAMAMLANDI!")
