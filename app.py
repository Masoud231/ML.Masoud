import streamlit as st
import numpy as np
import joblib
import os

# مسیر پوشه‌ی فعلی (همون جایی که app.py اجرا میشه)
current_dir = os.path.dirname(os.path.abspath(__file__))

# مسیرهای احتمالی فایل‌ها
model_path_dsp = os.path.join(current_dir, "eye_state_model.pkl")
scaler_path_dsp = os.path.join(current_dir, "scaler.pkl")

model_path_gta = os.path.join(os.path.dirname(current_dir), "eye_state_model.pkl")
scaler_path_gta = os.path.join(os.path.dirname(current_dir), "scaler.pkl")

# انتخاب مسیر درست
if os.path.exists(model_path_dsp) and os.path.exists(scaler_path_dsp):
    model = joblib.load(model_path_dsp)
    scaler = joblib.load(scaler_path_dsp)
elif os.path.exists(model_path_gta) and os.path.exists(scaler_path_gta):
    model = joblib.load(model_path_gta)
    scaler = joblib.load(scaler_path_gta)
else:
    st.error("❌ فایل‌های مدل پیدا نشدند! لطفاً مطمئن شوید eye_state_model.pkl و scaler.pkl وجود دارند.")
    st.stop()

st.title("EEG Eye State Prediction 🧠👁️")

st.write("این وب‌اپ به شما اجازه می‌دهد 14 ویژگی EEG وارد کنید و پیش‌بینی کند چشم باز است یا بسته.")

# ساخت فرم برای وارد کردن ویژگی‌ها
features = []
for i in range(14):
    val = st.number_input(f"Feature {i+1}", value=0.0, format="%.2f")
    features.append(val)

if st.button("Predict"):
    sample = np.array([features])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]

    if prediction == 0:
        st.error("Eye State: CLOSED 👁️‍🗨️")
    else:
        st.success("Eye State: OPEN 👁️")
