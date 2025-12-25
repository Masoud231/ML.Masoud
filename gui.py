import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo

# بارگذاری مدل و اسکیلر
model = joblib.load("eye_state_model.pkl")
scaler = joblib.load("scaler.pkl")

# بارگذاری دیتاست برای نمونه‌های واقعی
eeg_eye_state = fetch_ucirepo(id=264)
X = eeg_eye_state.data.features.values
y = eeg_eye_state.data.targets.values.ravel()

def predict_eye_state():
    try:
        # گرفتن ورودی‌ها
        values = [float(entry.get()) for entry in entries]

        if len(values) != 14:
            messagebox.showerror("Error", "Please enter exactly 14 values.")
            return

        # تبدیل به آرایه
        sample = np.array([values])

        # نرمال‌سازی
        sample_scaled = scaler.transform(sample)

        # پیش‌بینی
        prediction = model.predict(sample_scaled)[0]

        if prediction == 0:
            result_label.config(text="Eye State: CLOSED", fg="blue")
        else:
            result_label.config(text="Eye State: OPEN", fg="green")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

def fill_random_values():
    # تولید 14 عدد رندوم بین -1 و 1
    random_values = np.round(np.random.uniform(-1, 1, 14), 2)
    for i in range(14):
        entries[i].delete(0, tk.END)
        entries[i].insert(0, str(random_values[i]))

def fill_sample_from_dataset():
    # انتخاب یک نمونه تصادفی از دیتاست
    idx = np.random.randint(0, len(X))
    sample = X[idx]
    true_label = y[idx]

    for i in range(14):
        entries[i].delete(0, tk.END)
        entries[i].insert(0, str(round(sample[i], 2)))

    # نمایش برچسب واقعی برای تست
    if true_label == 0:
        result_label.config(text="True Label: CLOSED", fg="gray")
    else:
        result_label.config(text="True Label: OPEN", fg="gray")

# ساخت پنجره اصلی
root = tk.Tk()
root.title("EEG Eye State Prediction")
root.geometry("450x700")

title_label = tk.Label(root, text="EEG Eye State Classifier", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

# ساخت ورودی‌ها
entries = []
for i in range(14):
    frame = tk.Frame(root)
    frame.pack(pady=3)

    label = tk.Label(frame, text=f"Feature {i+1}:", font=("Arial", 12))
    label.pack(side=tk.LEFT)

    entry = tk.Entry(frame, width=20)
    entry.pack(side=tk.RIGHT)
    entries.append(entry)

# دکمه پر کردن رندوم
random_button = tk.Button(root, text="Fill Random", font=("Arial", 14), command=fill_random_values)
random_button.pack(pady=10)

# دکمه پر کردن از دیتاست
sample_button = tk.Button(root, text="Fill Dataset Sample", font=("Arial", 14), command=fill_sample_from_dataset)
sample_button.pack(pady=10)

# دکمه پیش‌بینی
predict_button = tk.Button(root, text="Predict", font=("Arial", 14), command=predict_eye_state)
predict_button.pack(pady=20)

# نمایش نتیجه
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=10)

root.mainloop()
