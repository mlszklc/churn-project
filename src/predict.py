import pandas as pd     #Veriyi dataframe haline getirip manipüle etmek için
import joblib           #.pkl dosyalarını açmak için 
import json             #columns.json dosyasını liste formatında okumak için
import os               # Bilgisayarımdaki klasör yollarını akıllıca yönetmek için

class PredictionService:
    def __init__(self):
        # Dosya yolları belirlendi
        base_path = os.path.dirname(os.path.abspath(__file__))      #os.path.abspath(__file__) şuan çalıştığım dosya hangisi diye bakar
        model_dir = os.path.join(base_path, '..', 'models')         # '..' bir üst klasöre çık demek. src içindeyken bir üst klasöre çıkıp models klasörüne gidiyoruz.

        print(f"--- Modelleri burada arıyorum: {os.path.abspath(model_dir)}")
        print(f"--- Klasördeki dosyalar: {os.listdir(model_dir)}")


        # Kaydedilmiş beyinleri (model ve scaler) belleğe yüklüyoruz.
        self.model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

        # Model eğitilirken kullanılan sütun sıralamasını listeye alıyoruz.
        with open(os.path.join(model_dir, 'columns.json'), 'r') as f:
            self.model_columns = json.load(f)
    
    def prepare_data(self, input_data):
        # Gelen sözlüğü (dict) Pandas'ın anlayacağı tek satırlık bir tabloya çevirir.
        df = pd.DataFrame([input_data])

        # Feature Engineering: Melisa'nın yaptığı yeni özellik türetme işlemi.
        # Eğer müşteri 6 aydan az süredir bizdeyse 'yeni' kabul ediyoruz.
        if 'tenure' in df.columns:
            df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
        
        # One-Hot Encoding: Metinleri (Cinsiyet, Sözleşme vb.) 0 ve 1'lere çevirir.
        df_encoded = pd.get_dummies(df)

        # KRİTİK NOKTA: reindex.
        # Elimizde sadece 5-10 sütun var ama model 44 tane bekliyor.
        # Bu satır, eksik olan tüm sütunları oluşturur ve içlerine 0 yazar.
        # Böylece modelin beklediği kolon sırası asla bozulmaz.
        df_final = df_encoded.reindex(columns=self.model_columns, fill_value=0)
        
        return df_final
    
    def predict(self, raw_data):
        # 1. Ham veriyi yukarıdaki metoda gönderip 44 sütunlu hale getir.
        prepared_df = self.prepare_data(raw_data)
        
        # 2. Ölçeklendirme: Sayıları (MonthlyCharges gibi) scaler ile 0-1 arasına çek.
        # Scaler eğitilirken hangi oranları kullandıysa, bu yeni veriye de aynısını yapar.
        scaled_data = self.scaler.transform(prepared_df)
        
        # 3. Tahmin: 'predict' bize 0 veya 1 döner.
        prediction = self.model.predict(scaled_data)[0]
        
        # 4. Olasılık: 'predict_proba' bize [kalma_olasılığı, gitme_olasılığı] döner.
        # Biz gitme (Churn) tarafını, yani 1. indeksi alıyoruz.
        probability = self.model.predict_proba(scaled_data)[0][1]
        
        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability) * 100) # İki basamaktan fazlasını yüzdelik olarak almak istemedim.
        }
