import pandas as pd #veriyi tablo olarak okumak, düzenlemek için
import os # Klasör oluşturmak için

# Veriyi oku
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# customerID modelde işimize yaramıyor, sil
df = df.drop(columns=['customerID'])
# drop = sütunu sil
# columns = hangi sütunu silecegimizi belirt
print("Veri okundu!")
print(f"Boyut: {df.shape[0]} satir, {df.shape[1]} sutun")

# ── GİZLİ EKSİK VERİYİ DÜZELT
# TotalCharges sütununu sayıya çevir
# Sayıya çevrilemeyen boşluk karakterleri NaN olur
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# NaN olan 11 satırı medyanla doldur
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
# EDA'da bulduğumuz gizli eksik 11 satır burada düzeltiliyor

print(f"Eksik değer kaldı mı: {df.isnull().sum().sum()}")
# 0 çıkmalı

#CSV OKUMA
##python -c "import pandas as pd; df = pd.read_csv
# ('data/WA_Fn-UseC_-Telco-Customer-Churn.csv'); print(df.nunique())"

# ── İKİLİ DÖNÜŞÜM
# Yes/No → 1/0, Male/Female → 1/0
ikili_sutunlar = ['gender', 'Partner', 'Dependents',
                  'PhoneService', 'PaperlessBilling','Churn']

for sutun in ikili_sutunlar:
    # her sütun için Yes=1, No=0, Male=1, Female=0
    df[sutun] = df[sutun].map({'Yes': 1, 'No': 0,
                                'Male': 1, 'Female': 0})
# map = "bu değeri şununla eşleştir" demek

print("İkili dönüşüm tamamlandı!")
print(df[ikili_sutunlar].head(3))
# ilk 3 satırı göster, 0/1 olmuş mu kontrol et

##gender, Partner, Dependents, PhoneService, PaperlessBilling 
#2 seçenek (Yes/No) İkili dönüşüm → 0/1
##MultipleLines, InternetService, Contract, PaymentMethod vb
#3-4 seçenek --> One-Hot Encoding

# ── ONE-HOT ENCODING
# Her seçenek ayrı bir sütun olur
donusturulecek_sutunlar = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
]

df = pd.get_dummies(df, columns= donusturulecek_sutunlar)
# get_dummies = one-hot encoding uygula
# columns = hangi sütunlara uygulayacağımız

print(f"One-Hot Encoding sonrası sütun sayısı: {df.shape[1]}")
# 20 sütundan çok daha fazlasına çıkacak
# çünkü her seçenek ayrı sütun oldu
print("Yeni sütunların ilk 5'i:")
print(df.columns.tolist()[:5])

# ── ÖZELLİK MÜHENDİSLİĞİ
df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
# tenure 6 veya altındaysa 1, değilse 0

df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
# aylık ortalama ücret — tenure+1 sıfıra bölme hatasını önler
#Model sadece TotalCharges'a baksa ikisini aynı görür. 
#Her müşterinin toplam ücretini, müşteri olduğu ay(tenure) sayısına böl.
#  Yeni bir sütun oluştur, adı ChargesPerMonth olsun.

df['AddOnServices'] = (
    df['OnlineSecurity_Yes'].astype(int) +
    df['OnlineBackup_Yes'].astype(int) +
    df['DeviceProtection_Yes'].astype(int) +
    df['TechSupport_Yes'].astype(int) +
    df['StreamingTV_Yes'].astype(int) +
    df['StreamingMovies_Yes'].astype(int)
)
# kaç tane ek hizmet kullandığı — 0'dan 6'ya kadar

print(f"Özellik mühendisliği sonrası sütun sayısı: {df.shape[1]}")
print(df[['IsNewCustomer', 'ChargesPerMonth', 'AddOnServices']].head(3))

# ── TEMİZ VERİYİ KAYDET
os.makedirs('data', exist_ok=True)
df.to_csv('data/clean_data.csv', index=False)

print(f"\n✅ Temiz veri kaydedildi: data/clean_data.csv")
print(f"Final boyut: {df.shape[0]} satır, {df.shape[1]} sütun")