import streamlit as st
import numpy as np
import joblib
import os
import cv2
import mediapipe as mp

st.set_page_config(page_title="Eye State Detection", layout="wide")

st.title("👁️ Eye State Detection System")
st.write("پیش‌بینی وضعیت چشم با دو روش: EEG و دوربین")

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
        st.error("❌ فایل‌های مدل EEG پیدا نشدند. لطفاً eye_state_model.pkl و scaler.pkl را در ریپو قرار دهید.")
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
            st.error("👁️‍🗨️ نتیجه: چشم بسته")
        else:
            st.success("👁️ نتیجه: چشم باز")


# ============================================================
# TAB 2 — CAMERA DETECTION
# ============================================================
with tabs[1]:
    st.header("تشخیص باز/بسته بودن چشم با دوربین")

    run = st.checkbox("فعال کردن دوربین")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def eye_aspect_ratio(landmarks, eye_indices):
        pts = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C)

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            if ear < 0.25:
                status = "چشم بسته"
                color = (0, 0, 255)
            else:
                status = "چشم باز"
                color = (0, 255, 0)

            cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
