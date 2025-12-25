import numpy as np
import joblib

# بارگذاری مدل و اسکیلر
model = joblib.load("eye_state_model.pkl")
scaler = joblib.load("scaler.pkl")

# نمونه ورودی EEG (۱۴ ویژگی)
sample = np.array([[0.12, -0.33, 0.55, 0.88, -0.12, 0.44, 0.91, -0.22, 0.11, 0.77, -0.55, 0.33, 0.66, -0.44]])

# نرمال‌سازی ورودی
sample_scaled = scaler.transform(sample)

# پیش‌بینی
prediction = model.predict(sample_scaled)[0]

if prediction == 0:
    print("Eye State: CLOSED")
else:
    print("Eye State: OPEN")
