import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Churn Prediction", layout="wide")

SUMMARY_URL = "http://127.0.0.1:8000/predict/batch/summary"
SINGLE_URL  = "http://127.0.0.1:8000/predict/single"

st.title("Customer Churn Prediction Dashboard")
st.divider()

tab1, tab2 = st.tabs(["Enter Data", "Upload Data (.csv)"])

# ── SEKME 1: Tek musteri formu ──────────────────────────────────────────────
with tab1:
    st.subheader("Enter Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Basic Information**")
        tenure = st.number_input("Tenure", min_value=0, max_value=72, value=12)
        monthly = st.number_input("Monthly Charges($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])

    with col2:
        st.markdown("**Contract & Service**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

    with col3:
        st.markdown("**Additional Services**")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    if st.button("Predict", type="primary"):
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "SeniorCitizen": senior,
            "Partner": partner,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment
        }

        try:
            response = requests.post(SINGLE_URL, json=payload, timeout=30)
        except requests.exceptions.ConnectionError:
            st.error("Couldn't connect API. Please run 'uvicorn app:app --reload' in your terminal.")
            st.stop()

        if response.status_code != 200:
            st.error(f"API error code: {response.status_code} - {response.text}")
            st.stop()

        result = response.json()
        prob = result["churn_probability"]
        prediction = result["churn_prediction"]
        risk = result["risk_level"]

        st.divider()
        st.subheader("Prediction Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            if prediction == "Yes":
                st.error("Prediction: WILL CHURN")
            else:
                st.success("Prediction: WILL STAY")

        with c2:
            st.metric("Churn Probability", f"%{prob:.1f}")

        with c3:
            if risk == "High Risk":
                st.error(f"Risk: {risk}")
            elif risk == "Mid Risk":
                st.warning(f"Risk: {risk}")
            else:
                st.success(f"Risk: {risk}")

        st.progress(int(prob))

# ── SEKME 2: CSV yukleme ────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload CSV file")

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"Data include: {len(df_preview)} customer, {len(df_preview.columns)} column")
        st.dataframe(df_preview.head(5))
        st.divider()

        if st.button("Start Prediction", type="primary"):
            uploaded_file.seek(0)
            file_bytes = uploaded_file.getvalue()

            with st.spinner("..."):
                try:
                    response = requests.post(
                        SUMMARY_URL,
                        files={"file": (uploaded_file.name, file_bytes, "text/csv")},
                        timeout=120
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Couldn't connect API. Please run 'uvicorn app:app --reload' in your terminal.")
                    st.stop()

            if response.status_code != 200:
                st.error(f"API error code: {response.status_code} - {response.text}")
                st.stop()

            data = response.json()

            st.subheader("Sum")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Customers",        data["total_customers"])
            m2.metric("Will Churn",             data["churn_count"])
            m3.metric("Will Stay",               data["stay_count"])
            m4.metric("Churn Rate",           f"%{data['churn_rate']}")
            m5.metric("Average Churn Probability",         f"%{data['avg_churn_probability']}")

            st.divider()

            st.subheader("Risk Distribution")
            r1, r2, r3 = st.columns(3)
            r1.error(f"High Risk: {data['high_risk']} customer")
            r2.warning(f"Mid Risk: {data['medium_risk']} customer")
            r3.success(f"Low Risk: {data['low_risk']} customer")

            st.divider()

            st.subheader("Results by Customer")
            preds = pd.DataFrame(data["predictions"])
            result_df = pd.concat([df_preview.reset_index(drop=True), preds], axis=1)
            st.dataframe(result_df, use_container_width=True)

            st.divider()

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results(.csv)",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )