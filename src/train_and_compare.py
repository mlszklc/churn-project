import joblib
import json

import pandas as pd #veri okur
import matplotlib.pyplot as plt #garfik çizmek için

from sklearn.model_selection import train_test_split #veriyi eğitim/test olarak bölmek için
from sklearn.preprocessing import StandardScaler #veriyi ölçeklendirmek için
from sklearn.linear_model import LogisticRegression #ilk sınıflandırma algoritması
from sklearn.tree import DecisionTreeClassifier #karar ağacı sınıflandırma algoritması
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #birden fazla karar ağacından oluşan sınıflandırma algoritmaları, boosting mantığıyla çalışan sınıflandırma algoritmaları
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score #model performansını değerlendirmek için kullanılan metrikler

from xgboost import XGBClassifier #xgboost sınıflandırma algoritması
from lightgbm import LGBMClassifier #lightgbm sınıflandırma algoritması

df = pd.read_csv("data/processed/clean_data.csv") #veriyi yükler

print("İlk 5 satır:")
print(df.head())

print("\nVeri boyutu:", df.shape)
print("\nSütunlar:")
print(df.columns.tolist()) #bütün sütunları liste olarak yazdırır

print("\nEksik değer kontrolü:")
print(df.isnull().sum().sum())

target_column = "Churn" #hedef değişkeni tanımlanır, bu örnekte müşteri kaybı (churn) tahmin edilmeye çalışılacak
X =df.drop(columns=[target_column]) #hedef değişkeni hariç tüm sütunlar özellikler (X) olarak alınır
y = df[target_column] #hedef değişkeni (y) olarak alınır

#x ve y kontol edelim
print("\nX boyutu: ", X.shape)
print("y boyutu: ", y.shape)
print("\nHedef değişken dağılımı:")
print(y.value_counts()) #hedef değişkenin dağılımını yazdırır, bu sayede sınıfların dengesiz olup olmadığını görebiliriz

#train/test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, #test setinin oranı %20
    random_state=42,#rastgele bölme işleminin tekrarlanabilir olması için sabit bir random state değeri kullanılır
    stratify=y
) #veriyi %80 eğitim ve %20 test olarak böler

#bölünmüş veri kontrolü
print("\nEğitim veri boyutu: ", X_train.shape)
print("Test veri boyutu: ", X_test.shape)

scaler = StandardScaler() #veriyi ölçeklendirmek için StandardScaler kullanılır, bu yöntem her özelliği ortalaması 0 ve standart sapması 1 olacak şekilde dönüştürür
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

#model performansını değerlendirmek için bir fonksiyon
def evaluate_model(model, X_train_data, X_test_data, y_train, y_test, model_name):
    model.fit(X_train_data, y_train) #modeli eğitim verisiyle eğitir
    y_pred = model.predict(X_test_data) #test verisi için tahmin yapar

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_data)[:, 1]
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }
    
trained_models = {} 

results = []

for model_name, model in models.items():
    print(f"\n{model_name} eğitiliyor...")
    
    if model_name == "Logistic Regression":
        result = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, model_name) #lojistik regresyon için ölçeklendirilmiş veriler kullanılır
    else:
        result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name) #diğer modeller için orijinal veriler kullanılır
    trained_models[model_name] = model #eğitilen modeli trained_models sözlüğüne kaydeder
    results.append(result) #her modelin sonuçları results listesine eklenir
    
results_df = pd.DataFrame(results) #model karşılaştırma sonuçlarını bir DataFrame'e dönüştürür, bu sayede sonuçları daha düzenli bir şekilde görebiliriz
print("\nModel Karşılaştırma Sonuçları:")
print(results_df)

results_df.to_csv("results/model_comparison_results.csv", index=False)
print("\nSonuçlar results/model_comparison_results.csv dosyasına kaydedildi.")

#Her bir metriği görselleştirmek için bar grafikleri oluşturulur, bu sayede modellerin performansını görsel olarak karşılaştırabiliriz
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
for metric in metrics:
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Model"], results_df[metric])
    plt.title(f"{metric} Karşılaştırması")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"results/{metric.lower().replace('-', '_').replace(' ', '_')}_comparison.png")
    plt.show()
    
best_model_row = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]
print("\nF1 Score'a göre en iyi model:") #Precision ve recall dengesini gösterir.
print(best_model_row)

best_model_name = best_model_row["Model"]
print("\nSeçilen en iyi model:", best_model_name)

best_model = trained_models[best_model_name] #en iyi modelin adını kullanarak trained_models sözlüğünden modeli alır

joblib.dump(best_model, "models/best_model.pkl") #en iyi modeli models klasörüne kaydeder
joblib.dump(scaler, "models/scaler.pkl") #ölçekleyiciyi de kaydeder, böylece modelin tahmin yaparken aynı ölçeklendirmeyi kullanabiliriz

with open("models/columns.json", "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, ensure_ascii=False, indent=4)

print("\nModel, scaler ve sütun bilgileri başarıyla kaydedildi.")
print("Kaydedilen dosyalar:")
print("- models/best_model.pkl")
print("- models/scaler.pkl")
print("- models/columns.json")
        
best_model_roc = results_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
print("\nROC-AUC'ye göre en iyi model:") #Modelin genel ayırma başarısını gösterir.
print(best_model_roc)    