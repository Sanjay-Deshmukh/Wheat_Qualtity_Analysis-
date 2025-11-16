
# 🌾 Wheat Grain Quality Classification Dashboard

**AI-Powered Deep Learning + Machine Learning System for Automated Wheat Grading**

This project is a complete **Flask-based web application** that processes wheat grain images, extracts deep features using **ResNet50**, classifies grain quality using a trained machine learning model, and generates a beautifully formatted, downloadable **PDF report** with detailed analysis.

---

## 🚀 Features

### 🔍 **AI-Based Classification**

* Extracts **2048-D deep features** using ResNet50 (ImageNet pretrained)
* Predicts wheat grain grades (A, B1, B2, C, D, F)
* Provides quality descriptions for each predicted grade

### 🖼️ **Multi-Image Upload**

* Upload multiple images at once
* Automatic runtime measurement per image
* Processed image thumbnails included in the PDF

### 📊 **Automatic PDF Report**

Includes:

* Cover page with branding
* Executive summary
* Grade distribution table
* Detailed results per image
* Embedded images
* Methodology & system explanation
* Performance statistics

### 💻 **Web-Based Dashboard**

* Clean, modern HTML UI
* AJAX-based processing
* Error handling for missing models or invalid files
* Supports `.jpg`, `.jpeg`, `.png`

### 📁 **Organized File Handling**

* Saves processed images in `static/uploads/`
* Automatically timestamps processed files
* Supports up to 16 MB uploads

---

## 📦 Project Structure

```
Wheat Grading/
│── app.py                          # Flask backend server
│── Model/
│   ├── grade_model.joblib          # ML classifier
│   ├── deep_feature_scaler.joblib  # Feature scaler
│   ├── grade_label_encoder.npy     # Class label encoder
│── templates/
│   ├── index.html                  # Web UI
│── static/
│   ├── uploads/                    # Processed images + reports
│── uploads/                        # Temporary input files
│── requirements.txt                # Dependencies
│── README.md                       # Project documentation
```

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Wheat-Grading.git
cd Wheat-Grading
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask server

```bash
python app.py
```

### 4️⃣ Open in browser

```
http://localhost:5001
```

---

## 🎯 Usage

### Upload Images

Upload one or more wheat grain images via browser.

### Classification

Each image is:

* Preprocessed
* Passed into ResNet50 for feature extraction
* Scaled using pre-trained scaler
* Classified by ML model

### Generate PDF Report

Click **“Download Report”** to export all results into a professional PDF.

---

## 📘 Machine Learning Pipeline

1. **ResNet50 (Feature Extractor)**

   * `include_top = False`
   * Extracts deep image embeddings (2048-d)

2. **Feature Scaling**

   * Pre-trained `StandardScaler`

3. **ML Classifier**

   * Trained using wheat quality dataset
   * Predicts grade labels (A, B1, B2, C1, … F)

---

## 📝 PDF Report Includes

✔ Cover Page
✔ Brand-colored header
✔ Executive Summary
✔ Grade Distribution
✔ Detailed Per-Image Analysis
✔ Embedded thumbnails
✔ Methodology
✔ Performance Statistics

Everything is styled with **ReportLab Tables, Styles, and Custom Colors**.

---

## 🛡 Allowed File Types

* `.jpg`, `.jpeg`, `.png`
* **Max file size:** 16 MB

---

## ⚠ Note on Repository Size

Large ML models or uploaded images should **not** be committed to GitHub.

Add this to your `.gitignore`:

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

✅ A README with badges and images
✅ A README with installation screenshots
Just tell me!
