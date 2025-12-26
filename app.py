import streamlit as st
import numpy as np
import joblib
import os
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Eye State Detection", layout="wide")

st.title("👁️ Eye State Detection System (Web Version)")
st.write("پیش‌بینی وضعیت چشم با EEG و تشخیص چشم با دوربین (نسخه تحت وب)")

tabs = st.tabs(["🔵 EEG Prediction", "🟢 Camera Eye Detection"])

# ============================================================
# TAB 1 — EEG MODEL
# ============================================================
with tabs[0]:
    st.header("پیش‌بینی وضعیت چشم با EEG")

    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "eye_state_model.pkl")
    scaler_path = os.path.join(base, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("❌ فایل‌های مدل EEG پیدا نشدند.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    st.subheader("ورود ویژگی‌ها (14 مقدار EEG)")

    if st.button("🔄 تولید نمونه EEG تصادفی"):
        random_sample = np.random.normal(0, 1, 14)
        for i in range(14):
            st.session_state[f"f{i}"] = float(random_sample[i])

    features = []
    for i in range(14):
        val = st.number_input(
            f"Feature {i+1}",
            value=st.session_state.get(f"f{i}", 0.0),
            format="%.4f"
        )
        features.append(val)

    if st.button("🔍 پیش‌بینی EEG"):
        sample = np.array([features])
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)[0]

        if prediction == 0:
            st.success("Eyes Open")
        else:
            st.error("Eyes Closed")


# ============================================================
# TAB 2 — CAMERA DETECTION (WEBRTC)
# ============================================================
class EyeDetector(VideoTransformerBase):
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)

        if len(eyes) == 0:
            status = "Eyes Closed"
            color = (0, 0, 255)
        else:
            status = "Eyes Open"
            color = (0, 255, 0)

        cv2.putText(img, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return img

with tabs[1]:
    st.header("Camera Eye Detection (WebRTC)")
    webrtc_streamer(
        key="eye-detection",
        video_transformer_factory=EyeDetector,
        media_stream_constraints={"video": True, "audio": False},
    )
