import streamlit as st
import pandas as pd
import joblib


# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

CHURN_THRESHOLD = 0.70

def interpret_risk(prob):
    if prob >= 0.70:
        return "RISIKO TINGGI"
    elif prob >= 0.40:
        return "RISIKO MENENGAH"
    else:
        return "RISIKO RENDAH"

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.joblib")
    except Exception:
        st.error("❌ Model tidak dapat dimuat. Silakan hubungi administrator sistem.")
        st.stop()

model = load_model()

# ===============================
# NEW UI LAYOUT
# ===============================
st.title("📉 CHURN PREDICTION")
st.markdown(
    "Aplikasi prediksi churn pelanggan untuk perusahaan telekomunikasi menggunakan model Machine Learning."
)

# Sidebar
st.sidebar.header("🔧 Input Pelanggan")

presets = {
    "Average": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
    },
    "Loyal": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 120.0,
    }
}

preset_choice = st.sidebar.selectbox("Preset Cepat", ["Custom"] + list(presets.keys()))


with st.sidebar.form("churn_form"):
    if preset_choice != "Custom":
        p = presets[preset_choice]
        gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(p["gender"]))
        senior = st.selectbox("Senior Citizen", [0, 1], index=[0, 1].index(p["SeniorCitizen"]))
        partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(p["Partner"]))
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(p["Dependents"]))
        tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=p["tenure"])

        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(p["PhoneService"]))
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=0)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(p["InternetService"]))
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], index=0 if p["OnlineSecurity"] == "Yes" else 1)
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=0 if p["OnlineBackup"] == "Yes" else 1)
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], index=0 if p["DeviceProtection"] == "Yes" else 1)
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=0 if p["TechSupport"] == "Yes" else 1)
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=0 if p["StreamingTV"] == "Yes" else 1)
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=0 if p["StreamingMovies"] == "Yes" else 1)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(p["Contract"]))
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(p["PaperlessBilling"]))
        payment_method = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], index=0)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=p["MonthlyCharges"])
    else:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)

        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0
        )

    submit = st.form_submit_button("🔍 Prediksi Churn")

# helper

def build_input_df(**kwargs):
    tenure = int(kwargs["tenure"])
    monthly_charges = float(kwargs["MonthlyCharges"])
    total_charges = tenure * monthly_charges
    risk_score = int(kwargs["Contract"] == "Month-to-month") + int(kwargs["TechSupport"] == "No") + int(kwargs["OnlineSecurity"] == "No")

    return pd.DataFrame([{
        "gender": kwargs["gender"],
        "SeniorCitizen": kwargs["SeniorCitizen"],
        "Partner": kwargs["Partner"],
        "Dependents": kwargs["Dependents"],
        "tenure": tenure,
        "PhoneService": kwargs["PhoneService"],
        "MultipleLines": kwargs["MultipleLines"],
        "InternetService": kwargs["InternetService"],
        "OnlineSecurity": kwargs["OnlineSecurity"],
        "OnlineBackup": kwargs["OnlineBackup"],
        "DeviceProtection": kwargs["DeviceProtection"],
        "TechSupport": kwargs["TechSupport"],
        "StreamingTV": kwargs["StreamingTV"],
        "StreamingMovies": kwargs["StreamingMovies"],
        "Contract": kwargs["Contract"],
        "PaperlessBilling": kwargs["PaperlessBilling"],
        "PaymentMethod": kwargs["PaymentMethod"],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "RiskScore": risk_score,
    }])

# main
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Hasil & Ringkasan")

    if submit:
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
        }

        input_df = build_input_df(**input_data)

        try:
            proba = model.predict_proba(input_df)[0][1]
        except Exception as e:
            st.exception(e)
            st.stop()

        pred = int(proba >= CHURN_THRESHOLD)
        risk_level = interpret_risk(proba)

        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Probabilitas Churn", f"{proba:.2%}")
        rcol2.metric("Tingkat Risiko", risk_level)
        rcol3.metric("Prediksi", "CHURN" if pred == 1 else "TIDAK CHURN")

        st.progress(min(proba, 1.0))

        if pred == 1:
            st.error("⚠️ Pelanggan **BERPOTENSI CHURN**")
        else:
            st.success("✅ Pelanggan **TIDAK CHURN**")

        st.caption("📌 Catatan: Pelanggan dikategorikan churn jika probabilitas ≥ 70%.")

    st.markdown("---")
    st.subheader("🔍 Feature Descriptions")
    with st.expander("Lihat deskripsi fitur"):
        st.write({
            "Gender": "Jenis kelamin pelanggan: Male / Female",
            "SeniorCitizen": "Apakah pelanggan adalah warga senior (1 = Ya, 0 = Tidak)",
            "Partner": "Apakah pelanggan memiliki pasangan",
            "Dependents": "Apakah pelanggan memiliki tanggungan",
            "Tenure": "Lama pelanggan menggunakan layanan (bulan)",
            "PhoneService": "Apakah pelanggan menggunakan layanan telepon",
            "MultipleLines": "Apakah pelanggan memiliki beberapa jalur telepon",
            "InternetService": "Jenis layanan internet (DSL, Fiber optic, No)",
            "OnlineSecurity": "Apakah pelanggan memiliki keamanan online",
            "OnlineBackup": "Apakah pelanggan menggunakan backup online",
            "DeviceProtection": "Apakah perangkat pelanggan dilindungi",
            "TechSupport": "Layanan dukungan teknis",
            "StreamingTV": "Apakah pelanggan menggunakan layanan Streaming TV",
            "StreamingMovies": "Apakah pelanggan menggunakan layanan Streaming Movies",
            "Contract": "Jenis kontrak pelanggan (Month-to-month, One year, Two year)",
            "PaperlessBilling": "Apakah pelanggan menggunakan tagihan tanpa kertas",
            "PaymentMethod": "Metode pembayaran",
            "MonthlyCharges": "Biaya bulanan yang dibayarkan pelanggan",
        })

with col2:
    st.header("🛠️ Input Pelanggan")
    st.info("Gunakan formulir di sidebar untuk memasukkan detail pelanggan dan memprediksi churn.")