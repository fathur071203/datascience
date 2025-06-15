import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model dan scaler
@st.cache_resource
def load_model():
    model = joblib.load("gradient_boosting_model.pkl")
    scaler = joblib.load("scaler.pkl") if "scaler.pkl" in os.listdir() else None
    return model, scaler

model, scaler = load_model()

# Daftar fitur yang digunakan model
feature_names = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course', 'Daytime_evening_attendance',
    'Previous_qualification', 'Previous_qualification_grade', 'Nacionality', 'Mothers_occupation',
    'Fathers_occupation', 'Admission_grade', 'Displaced', 'Educational_special_needs', 'Gender',
    'Scholarship_holder', 'Age_at_enrollment', 'International', 'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations', 'Pass_Rate_1st_Sem', 'Financial_Risk',
    'Parents_Highest_Qualification'
]

status_mapping = {2: "Graduate", 1: "Enrolled", 0: "Dropout"}

st.title("Prediksi Dropout Mahasiswa - Jaya Jaya Institut")

st.markdown("""
Prototype ini menggunakan model machine learning Gradient Boosting untuk memprediksi status mahasiswa (Graduate, Enrolled, Dropout) berdasarkan data akademik dan demografi.
""")

st.header("Input Data Mahasiswa")
input_data = {}
for col in feature_names:
    if col in ['Previous_qualification_grade', 'Admission_grade', 'Age_at_enrollment',
               'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_evaluations',
               'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
               'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
               'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
               'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
               'Pass_Rate_1st_Sem', 'Financial_Risk', 'Parents_Highest_Qualification']:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    else:
        input_data[col] = st.number_input(f"{col}", value=0)

if st.button("Prediksi Status Mahasiswa"):
    # DataFrame dari input user
    df_input = pd.DataFrame([input_data])
    # Normalisasi jika scaler tersedia
    if scaler is not None:
        df_input[df_input.columns] = scaler.transform(df_input[df_input.columns])
    # Prediksi
    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    st.success(f"Status Prediksi: **{status_mapping[pred]}**")
    st.write("Probabilitas:")
    st.write({status_mapping[i]: f"{p:.2%}" for i, p in enumerate(proba)})

st.markdown("---")
st.header("Cara Menjalankan Prototype Ini")
st.markdown("""
1. **Deploy di Streamlit Community Cloud**  
   - Upload file `gradient_boosting_model.pkl` (dan `scaler.pkl` jika ada) serta file kode ini ke repository GitHub.
   - Buka [https://streamlit.io/cloud](https://streamlit.io/cloud) dan hubungkan ke repo Anda.
   - Klik "Deploy".

2. **Akses Prototype**  
   Setelah deploy, Anda akan mendapatkan link seperti:  
   `https://your-username-your-repo-name.streamlit.app/`

3. **Contoh Link Akses**  
   [https://your-username-your-repo-name.streamlit.app/](https://your-username-your-repo-name.streamlit.app/)
""")