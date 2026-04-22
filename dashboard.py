import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Churn Tahmin", layout="wide")

SUMMARY_URL = "http://127.0.0.1:8000/predict/batch/summary"
SINGLE_URL  = "http://127.0.0.1:8000/predict/single"

st.title("Musteri Churn Tahmin Sistemi")
st.divider()

sekme1, sekme2 = st.tabs(["Tek Musteri Tahmini", "Toplu CSV Tahmini"])

# ── SEKME 1: Tek musteri formu ──────────────────────────────────────────────
with sekme1:
    st.subheader("Musteri Bilgilerini Girin")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Musteri Suresi (ay)", min_value=0, max_value=72, value=12)
        monthly = st.number_input("Aylik Ucret ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayir")

    with col2:
        contract = st.selectbox("Sozlesme Turu", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Turu", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Odeme Yontemi", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

    if st.button("Tahmin Yap", type="primary"):
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "SeniorCitizen": senior,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment
        }

        try:
            response = requests.post(SINGLE_URL, json=payload, timeout=30)
        except requests.exceptions.ConnectionError:
            st.error("API'ye baglanılamadi. uvicorn app:app --reload komutunu calistir.")
            st.stop()

        if response.status_code != 200:
            st.error(f"API hatasi: {response.status_code} - {response.text}")
            st.stop()

        result = response.json()
        prob = result["churn_probability"]
        prediction = result["churn_prediction"]
        risk = result["risk_level"]

        st.divider()
        st.subheader("Tahmin Sonucu")

        c1, c2, c3 = st.columns(3)

        with c1:
            if prediction == "Yes":
                st.error("Tahmin: AYRILACAK")
            else:
                st.success("Tahmin: KALACAK")

        with c2:
            st.metric("Ayrilma Olasiligi", f"%{prob:.1f}")

        with c3:
            if risk == "Yuksek Risk":
                st.error(f"Risk: {risk}")
            elif risk == "Orta Risk":
                st.warning(f"Risk: {risk}")
            else:
                st.success(f"Risk: {risk}")

        st.progress(int(prob))

# ── SEKME 2: CSV yukleme ────────────────────────────────────────────────────
with sekme2:
    st.subheader("CSV Dosyasi Yukle")

    uploaded_file = st.file_uploader("CSV dosyasi secin", type=["csv"])

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"Yuklenen veri: {len(df_preview)} musteri, {len(df_preview.columns)} sutun")
        st.dataframe(df_preview.head(5))
        st.divider()

        if st.button("Tahminleri Baslat", type="primary"):
            uploaded_file.seek(0)
            file_bytes = uploaded_file.getvalue()

            with st.spinner("Tahmin yapiliyor..."):
                try:
                    response = requests.post(
                        SUMMARY_URL,
                        files={"file": (uploaded_file.name, file_bytes, "text/csv")},
                        timeout=120
                    )
                except requests.exceptions.ConnectionError:
                    st.error("API'ye baglanılamadi. uvicorn app:app --reload komutunu calistir.")
                    st.stop()

            if response.status_code != 200:
                st.error(f"API hatasi: {response.status_code} - {response.text}")
                st.stop()

            data = response.json()

            st.subheader("Genel Ozet")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Toplam Musteri",        data["total_customers"])
            m2.metric("Ayrilacak",             data["churn_count"])
            m3.metric("Kalacak",               data["stay_count"])
            m4.metric("Churn Orani",           f"%{data['churn_rate']}")
            m5.metric("Ort. Olasilik",         f"%{data['avg_churn_probability']}")

            st.divider()

            st.subheader("Risk Dagilimi")
            r1, r2, r3 = st.columns(3)
            r1.error(f"Yuksek Risk: {data['high_risk']} musteri")
            r2.warning(f"Orta Risk: {data['medium_risk']} musteri")
            r3.success(f"Dusuk Risk: {data['low_risk']} musteri")

            st.divider()

            st.subheader("Musteri Bazinda Sonuclar")
            preds = pd.DataFrame(data["predictions"])
            result_df = pd.concat([df_preview.reset_index(drop=True), preds], axis=1)
            st.dataframe(result_df, use_container_width=True)

            st.divider()

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="CSV olarak indir",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
