from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# بارگذاری مدل و اسکیلر
model = joblib.load("eye_state_model.pkl")
scaler = joblib.load("scaler.pkl")

# تعریف اپلیکیشن FastAPI
app = FastAPI(title="EEG Eye State API")

# تعریف ورودی
class EEGInput(BaseModel):
    features: list[float]  # لیست 14 ویژگی

@app.post("/predict")
def predict_eye_state(data: EEGInput):
    # تبدیل ورودی به آرایه
    sample = np.array([data.features])

    # نرمال‌سازی
    sample_scaled = scaler.transform(sample)

    # پیش‌بینی
    prediction = model.predict(sample_scaled)[0]

    return {"prediction": int(prediction), "eye_state": "OPEN" if prediction == 1 else "CLOSED"}
