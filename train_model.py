from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# مرحله 1: گرفتن دیتاست EEG Eye State
eeg_eye_state = fetch_ucirepo(id=264)

# مرحله 2: جدا کردن ویژگی‌ها و برچسب‌ها
X = eeg_eye_state.data.features.values
y = eeg_eye_state.data.targets.values.ravel()

# مرحله 3: بررسی توزیع برچسب‌ها
unique, counts = np.unique(y, return_counts=True)
print("Label distribution:", dict(zip(unique, counts)))

# مرحله 4: نرمال‌سازی داده‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# مرحله 5: تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# مدل سریع و بهینه Random Forest
rf_fast = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# آموزش مدل
rf_fast.fit(X_train, y_train)

# پیش‌بینی
y_pred_fast = rf_fast.predict(X_test)

# ارزیابی
print("=== FAST Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_fast))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_fast))
print("Classification Report:\n", classification_report(y_test, y_pred_fast))

# ذخیره مدل و اسکیلر
joblib.dump(rf_fast, "eye_state_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved!")
