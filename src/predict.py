import pandas as pd
import joblib
import json
import os

class PredictionService:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_path, '..', 'models')

        self.model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

        with open(os.path.join(model_dir, 'columns.json'), 'r') as f:
            self.model_columns = json.load(f)

    def prepare_data(self, input_data):
        df = pd.DataFrame([input_data])

        # Churn sütunu varsa sil — model bunu görmemeli
        # Churn sütunu varsa sil — bu hedef değişken, girdi değil
        for churn_col in ['Churn', 'Churn_Yes', 'Churn_No']:
            if churn_col in df.columns:
                df = df.drop(columns=[churn_col])

        # ── İKİLİ DÖNÜŞÜM — Yes/No → 1/0, Male/Female → 1/0
        ikili_sutunlar = ['gender', 'Partner', 'Dependents',
                          'PhoneService', 'PaperlessBilling']
        for sutun in ikili_sutunlar:
            if sutun in df.columns:
                df[sutun] = df[sutun].map({'Yes': 1, 'No': 0,
                                           'Male': 1, 'Female': 0})

        # ── TotalCharges sayıya çevir
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

        # ── ÖZELLİK MÜHENDİSLİĞİ
        # Yeni müşteri mi?
        if 'tenure' in df.columns:
            df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)

        # Aylık ortalama ücret
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

        # ── ONE-HOT ENCODING
        ohe_sutunlar = [
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
        ]
        mevcut_ohe = [s for s in ohe_sutunlar if s in df.columns]
        df = pd.get_dummies(df, columns=mevcut_ohe)

        # ── AddOnServices — one-hot encoding sonrası hesaplanmalı
        ek_hizmetler = [
            'OnlineSecurity_Yes', 'OnlineBackup_Yes',
            'DeviceProtection_Yes', 'TechSupport_Yes',
            'StreamingTV_Yes', 'StreamingMovies_Yes'
        ]
        df['AddOnServices'] = sum(
            df[s].astype(int) if s in df.columns else 0
            for s in ek_hizmetler
        )

        # ── Modelin beklediği sütun sıralamasına getir
        df = df.fillna(0)
        df_final = df.reindex(columns=self.model_columns, fill_value=0)
        return df_final

    def predict(self, raw_data):
        prepared_df = self.prepare_data(raw_data)
        scaled_data = self.scaler.transform(prepared_df)
        prediction = self.model.predict(scaled_data)[0]
        probability = self.model.predict_proba(scaled_data)[0][1]

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability) * 100)
        }