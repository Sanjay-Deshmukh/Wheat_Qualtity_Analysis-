import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm

# Paths
DATASET_BASE = r"D:\DSP_CP2\DSP_CP2\Grade"
TRAIN_DIR = os.path.join(DATASET_BASE, "Train")
TEST_DIR = os.path.join(DATASET_BASE, "Test")
SAVE_DIR = r"D:\VIT\DSP_CP2\Model"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load ResNet50 avg_pool layer output: 2048 features
base_model = ResNet50(weights="imagenet", include_top=True)
feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)

def extract_features(folder):
    X, y = [], []
    classes = sorted(os.listdir(folder))

    for cls in classes:
        class_dir = os.path.join(folder, cls)
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {cls}"):
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feat = feature_model.predict(img, verbose=0)
            X.append(feat.flatten())
            y.append(cls)
    return np.array(X), np.array(y)

# Extract
X_train, y_train = extract_features(TRAIN_DIR)
X_test, y_test = extract_features(TEST_DIR)

# Save features
np.save(os.path.join(SAVE_DIR, "grade_train_features.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "grade_train_labels.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "grade_test_features.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "grade_test_labels.npy"), y_test)

print("Feature Extraction Completed & Saved in:", SAVE_DIR)
