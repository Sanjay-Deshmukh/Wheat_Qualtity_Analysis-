import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import dump

SAVE_DIR = r"D:\DSP_CP2\DSP_CP2\Model"

# Load extracted features & labels
X_train = np.load(os.path.join(SAVE_DIR, "grade_train_features.npy"))
y_train_raw = np.load(os.path.join(SAVE_DIR, "grade_train_labels.npy"))

# Convert B1,B2→B and C1,C2,C3→C etc.
y_train_raw = np.array([label[0].upper() for label in y_train_raw])

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
np.save(os.path.join(SAVE_DIR, "grade_label_encoder.npy"), le.classes_)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train SVM
model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
dump(model, os.path.join(SAVE_DIR, "grade_model.joblib"))
dump(scaler, os.path.join(SAVE_DIR, "deep_feature_scaler.joblib"))

print("\n Grade Model Retrained and Saved Successfully.")
print("Classes Used:", le.classes_)
