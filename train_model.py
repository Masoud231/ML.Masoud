from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# مرحله 1: گرفتن دیتاست EEG Eye State
eeg_eye_state = fetch_ucirepo(id=264)

# مرحله 2: جدا کردن ویژگی‌ها و برچسب‌ها
X = eeg_eye_state.data.features.values
y = eeg_eye_state.data.targets.values.ravel()

# مرحله 3: نرمال‌سازی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# مرحله 4: تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# مدل فوق‌العاده سبک ExtraTrees
model = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# آموزش
model.fit(X_train, y_train)

# ارزیابی
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ذخیرهٔ فشرده
base = os.path.dirname(os.path.abspath(__file__))
joblib.dump(model, os.path.join(base, "eye_state_model.pkl"), compress=3)
joblib.dump(scaler, os.path.join(base, "scaler.pkl"), compress=3)

print("Model and scaler saved in:", base)
