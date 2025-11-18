

# 🌾 Wheat Grain Quality Classification Dashboard

**AI-Powered Deep Learning + Machine Learning System for Automated Wheat Grading**

This project is a complete **Flask-based web application** that analyzes wheat grain images, extracts deep features using **ResNet50**, classifies grain quality using a trained ML model, and generates a professionally formatted **PDF report** with detailed analysis.

It is designed for agricultural automation, quality control, and AI-based food inspection systems.

---

# 📂 Dataset Details

## 🌾 **Wheat Grading & Type Dataset**

This project uses a custom wheat grain image dataset created and uploaded by me.

👉 **Dataset Link:**
[https://www.kaggle.com/datasets/sanjaydeshmukh1212/wheat-grading-and-type-dataset](https://www.kaggle.com/datasets/sanjaydeshmukh1212/wheat-grading-and-type-dataset)

---

## 📘 **Dataset Description**

The **Wheat Grading & Type Dataset** contains high-resolution images of wheat grains categorized into **multiple quality grades and grain types**.
It is specifically designed for:

* Wheat quality assessment
* Grain grading automation
* Deep-learning classification
* Feature extraction research
* Agricultural AI applications

The dataset captures a wide range of real-world grain variations including:

✔ Size & texture differences
✔ Discoloration
✔ Shriveled or damaged grains
✔ Clean vs. impure grain samples
✔ Multi-variety wheat samples

---

## 🏷 **Classes / Grade Labels**

The dataset includes the following quality grades:

* **A** – Excellent quality
* **B1 / B2 / B** – Good quality
* **C1 / C2 / C3 / C** – Moderate quality
* **D** – Low quality
* **F** – Rejected / poor quality

These same labels are used by the ML classifier in this project.

---

## 📁 **Dataset Structure**

```
Wheat-Grading-Dataset/
│── A/
│── B1/
│── B2/
│── B/
│── C1/
│── C2/
│── C3/
│── C/
│── D/
│── F/
```

```
Wheat-Variety-Dataset/
│── BG_Gujarati/
│── Black_organic_wheat/
│── Jaora_lokwan/
│── Khapli/
│── Lokwan/
│── MP/
│── MP_lokwan/
│── SMP_lokwan/
│── Super_Rajwadi_lokwan/
│── VIP_Sihore/
```


## 🖼 **Image Specifications**

* Format: `.jpg` / `.png`
* Resolution: High-quality raw images
* Background: Clean and uniform
* Suitable for CNN-based models (ResNet, EfficientNet, ViT, etc.)

---

# 🚀 Features of This Application

## 🔍 **AI-Based Wheat Classification**

* Extracts **2048-D deep features** using ResNet50
* Uses pre-trained scaler + ML classifier
* Predicts wheat quality grade instantly
* Provides descriptive quality interpretation

---

## 🖼️ **Multi-Image Upload**

* Upload one or multiple images simultaneously
* Real-time progress
* Thumbnails shown after prediction
* Saved in `/static/uploads/` with timestamp

---

## 📊 **Automatic PDF Report Generation**

Generated PDF includes:

✔ Cover page with branding
✔ Executive summary
✔ Grade distribution table
✔ Detailed results per image
✔ Image thumbnails
✔ Methodology & performance statistics
✔ Clean, modern styling with colored sections

All processing is done using **ReportLab**.

---

## 💻 **Web Dashboard (Flask + AJAX)**

* Simple, modern UI
* Smooth experience with no page reloads
* Error handling for invalid files / missing models
* Supports `.jpg`, `.jpeg`, `.png`

---

# 📦 Project Structure

```
Wheat Grading/
│── app.py                          # Flask backend server
│── Model/
│   ├── grade_model.joblib
│   ├── deep_feature_scaler.joblib
│   ├── grade_label_encoder.npy
│── templates/
│   ├── index.html
│── static/
│   ├── uploads/                    # Processed images + reports
│── uploads/                        # Temporary input files
│── requirements.txt                # Dependencies
│── README.md                       # Project documentation
```

---

# 🛠 Installation & Setup

## 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Wheat-Grading.git
cd Wheat-Grading
```

## 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run the Flask server

```bash
python app.py
```

## 4️⃣ Open in browser

```
http://localhost:5001
```

---

# 🎯 Usage Guide

### 1. Upload Images

Upload one or multiple wheat grain images.

### 2. Classification

Each uploaded image is:

* Resized & preprocessed
* Passed through ResNet50
* Converted into deep features (2048-D)
* Scaled and classified

### 3. Generate PDF Report

Click “**Download Report**” to export all results.

---

# 📘 Machine Learning Pipeline

### ✔ 1. Feature Extraction (ResNet50)

* Pretrained on ImageNet
* `include_top=False`, global average pooling
* Produces a **2048-D feature vector**

### ✔ 2. Feature Scaling

* Standardized using pre-trained `StandardScaler`

### ✔ 3. ML Classifier

* Trained on wheat dataset
* Predicts one of the grade labels (A–F)

---

# 📝 PDF Contents

The generated PDF includes:

### ⭐ Cover Page

Stylized heading, date, processing stats

### ⭐ Executive Summary

Image count, time taken, overall outcome

### ⭐ Grade Distribution

Table of all grades + percentages

### ⭐ Detailed Results

Image-wise:

* Predicted grade
* Description
* Processing time
* Timestamp
* Thumbnail

### ⭐ Methodology

Explains the entire pipeline

### ⭐ Performance Statistics

Avg runtime, grade diversity, etc.

---

# 🛡 Allowed File Types

* `.jpg`, `.jpeg`, `.png`
* Max upload size: **16 MB**

---

# ⚠ Important: Large File Handling

Do **NOT** commit model files or uploaded images to GitHub.

Add to `.gitignore`:

```
uploads/
static/uploads/
Model/
*.h5
*.npy
*.joblib
*.pt
*.pdf
__pycache__/
```

---

# 🏆 Credits

* Dataset created by **Sanjay Deshmukh**
* Model training, backend, and PDF design by the author
* Deep learning backbone: **ResNet50 (Keras)**
* Web framework: **Flask**


