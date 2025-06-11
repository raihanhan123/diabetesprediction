import streamlit as st
import numpy as np
import joblib

# Memuat model yang sudah dilatih
model = joblib.load('diabetes_model.pkl')

st.title("Prediksi Diabetes")

# Form input untuk data pengguna
with st.form("Form_diabetes"):
    st.header("Masukkan data pasien:")
    
    # Input field untuk setiap fitur
    pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Kadar Glukosa', min_value=0, max_value=200)
    blood_pressure = st.number_input('Tekanan Darah (mm Hg)', min_value=0, max_value=150)
    skin_thickness = st.number_input('Ketebalan Kulit (mm)', min_value=0, max_value=100)
    insulin = st.number_input('Kadar Insulin (mu U/ml)', min_value=0, max_value=1000)
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, max_value=70.0, format="%.1f")
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, format="%.3f")
    age = st.number_input('Usia (tahun)', min_value=1, max_value=120)
    
    # Tombol submit form
    submit = st.form_submit_button("Proses")

# Ketika tombol submit ditekan
if submit:
    # Format input ke dalam bentuk array numpy
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Lakukan prediksi menggunakan model
    prediction = model.predict(features)[0]
    
    # Tampilkan hasil prediksi
    st.header("Hasil Prediksi")
    if prediction == 1:
        st.error("Hasil: Positif Diabetes")
    else:
        st.success("Hasil: Tidak Diabetes")
