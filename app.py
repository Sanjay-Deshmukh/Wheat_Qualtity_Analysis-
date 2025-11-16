import os
import numpy as np
import time
import datetime
import shutil
import textwrap
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from joblib import load
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.platypus.frames import Frame
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'wheat-grading-secret-key-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

# ------------------- CONFIG -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

# ------------------- MODEL CHECK -------------------
required_files = [
    "grade_model.joblib",
    "deep_feature_scaler.joblib",
    "grade_label_encoder.npy"
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing_files:
    print(f"Warning: Missing model files: {missing_files}")

# ------------------- LOAD MODELS -------------------
try:
    grade_model = load(os.path.join(MODEL_DIR, "grade_model.joblib"))
    grade_scaler = load(os.path.join(MODEL_DIR, "deep_feature_scaler.joblib"))
    grade_classes = np.load(os.path.join(MODEL_DIR, "grade_label_encoder.npy"), allow_pickle=True)
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False
    grade_model = None
    grade_scaler = None
    grade_classes = None

# ------------------- FEATURE EXTRACTOR -------------------
if models_loaded:
    base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
    feature_model = Model(inputs=base_model.input, outputs=base_model.output)
else:
    feature_model = None

def extract_features(image_path):
    """Extracts 2048D deep features using ResNet50."""
    if feature_model is None:
        raise Exception("Model not loaded")
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_model.predict(img, verbose=0)
    return features.reshape(1, -1)

# ------------------- GRADE DESCRIPTIONS -------------------
grade_descriptions = {
    "A": "Excellent quality — clean, uniform, and healthy grains.",
    "B1": "Good quality — minor defects or discoloration.",
    "B2": "Good quality — minor defects or discoloration.",
    "B": "Good quality — minor defects or discoloration.",
    "C1": "Moderate quality — noticeable irregularities.",
    "C2": "Moderate quality — noticeable irregularities.",
    "C3": "Moderate quality — noticeable irregularities.",
    "C": "Moderate quality — noticeable irregularities.",
    "D": "Low quality — visible damage or poor size.",
    "F": "Rejected — poor grain quality or contamination."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please check model files.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    results = []
    total_start = time.time()
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            start = time.time()
            try:
                deep_feat = extract_features(filepath)
                deep_scaled = grade_scaler.transform(deep_feat)
                pred_index = grade_model.predict(deep_scaled)[0]
                pred_grade = str(grade_classes[pred_index])
                runtime = time.time() - start
                
                # Move file to static/uploads for serving
                static_path = os.path.join('static', 'uploads', filename)
                try:
                    shutil.move(filepath, static_path)
                except:
                    pass
                
                results.append({
                    'filename': file.filename,
                    'predicted_grade': pred_grade,
                    'description': grade_descriptions.get(pred_grade, "Quality assessment complete."),
                    'runtime': round(runtime, 2),
                    'image_url': f"/uploads/{filename}",
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
    
    total_time = time.time() - total_start
    return jsonify({
        'results': results,
        'total_time': round(total_time, 2),
        'count': len(results)
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join('static', 'uploads', filename))

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    data = request.json
    results = data.get('results', [])
    
    if not results:
        return jsonify({'error': 'No results to generate report'}), 400
    
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        return jsonify({'error': 'No valid results to generate report'}), 400
    
    # Generate report filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"wheat_report_{timestamp}.pdf"
    report_path = os.path.join('static', 'uploads', report_filename)
    
    try:
        create_detailed_report(report_path, valid_results, data)
        return jsonify({
            'success': True,
            'report_url': f'/uploads/{report_filename}',
            'filename': report_filename
        })
    except Exception as e:
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

def create_detailed_report(filepath, results, data):
    """Create a detailed PDF report with cover page."""
    # Get report date and time
    report_date = datetime.datetime.now().strftime("%B %d, %Y")
    report_time = datetime.datetime.now().strftime("%I:%M %p")
    
    # Use SimpleDocTemplate with custom page handlers
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=100, bottomMargin=50)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles with enhanced colors
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=42,
        textColor=colors.HexColor('#1e3a5f'),
        spaceAfter=0,
        spaceBefore=0,
        leading=50,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=20,
        textColor=colors.HexColor('#4a90e2'),
        spaceAfter=20,
        leading=24,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=22,
        textColor=colors.HexColor('#1e3a5f'),
        spaceAfter=15,
        spaceBefore=25,
        leading=26,
        fontName='Helvetica-Bold',
        borderPadding=5,
        borderColor=colors.HexColor('#1e3a5f'),
        borderWidth=0
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        leading=16,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )

    # ========== COVER PAGE ==========
    # Decorative header box
    cover_header_data = [['WHEAT GRAIN QUALITY CLASSIFICATION REPORT']]
    cover_header = Table(cover_header_data, colWidths=[6*inch])
    cover_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 24),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(Spacer(1, 2*inch))
    elements.append(cover_header)
    elements.append(Spacer(1, 0.5*inch))
    
    # Main title with icon
    elements.append(Paragraph("🌾", ParagraphStyle('Icon', parent=styles['Normal'],
                                                   fontSize=60, alignment=TA_CENTER,
                                                   spaceAfter=10)))
    elements.append(Paragraph("WHEAT GRAIN QUALITY", title_style))
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph("CLASSIFICATION", title_style))
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph("REPORT", title_style))
    elements.append(Spacer(1, 0.6*inch))
    elements.append(Paragraph("AI-Powered Deep Learning System", subtitle_style))
    elements.append(Spacer(1, 1.2*inch))
    
    # Report details table with enhanced styling
    details = [
        ["Report Date:", report_date],
        ["Report Time:", report_time],
        ["Total Images:", str(len(results))],
        ["Processing Time:", f"{data.get('total_time', 0):.2f} seconds"]
    ]
    
    details_table = Table(details, colWidths=[2.2*inch, 3.8*inch])
    details_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4f8')),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 13),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d0d0')),
    ]))
    elements.append(details_table)
    elements.append(Spacer(1, 2*inch))
    
    # Footer with decorative line
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], 
                                 fontSize=11, textColor=colors.HexColor('#7f8c8d'),
                                 alignment=TA_CENTER, fontName='Helvetica-Oblique')
    elements.append(Paragraph("─" * 50, footer_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Generated by Wheat Quality Classification System", footer_style))
    elements.append(PageBreak())
    
    # ========== MAIN REPORT CONTENT ==========
    # Add decorative divider
    divider = Table([['']], colWidths=[6*inch], rowHeights=[0.05*inch])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e3a5f')),
        ('LINEBELOW', (0, 0), (-1, -1), 2, colors.HexColor('#4a90e2')),
    ]))
    elements.append(divider)
    elements.append(Spacer(1, 0.3*inch))
    
    # Executive Summary with box
    summary_box_data = [['Executive Summary']]
    summary_header = Table(summary_box_data, colWidths=[6*inch])
    summary_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(summary_header)
    elements.append(Spacer(1, 0.2*inch))
    
    summary_text = (
        f"This report presents the quality classification results for <b>{len(results)}</b> wheat grain images "
        f"processed using an AI-powered deep learning system. The analysis was completed in "
        f"<b>{data.get('total_time', 0):.2f} seconds</b>, with an average processing time of "
        f"<b>{data.get('total_time', 0) / len(results):.2f} seconds</b> per image."
    )
    elements.append(Paragraph(summary_text, body_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Grade Distribution with box
    grade_header_data = [['Grade Distribution']]
    grade_header = Table(grade_header_data, colWidths=[6*inch])
    grade_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(grade_header)
    elements.append(Spacer(1, 0.2*inch))
    grade_counts = {}
    for result in results:
        grade = result['predicted_grade']
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
    
    # Create grade distribution table
    grade_data = [['Grade', 'Count', 'Percentage', 'Quality Description']]
    for grade, count in sorted(grade_counts.items()):
        percentage = (count / len(results)) * 100
        description = grade_descriptions.get(grade, "Quality assessment complete.")
        grade_data.append([grade, str(count), f"{percentage:.1f}%", description])
    
    grade_table = Table(grade_data, colWidths=[0.9*inch, 0.9*inch, 1.1*inch, 3.1*inch])
    grade_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 13),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
        ('TOPPADDING', (0, 0), (-1, 0), 15),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(grade_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Detailed Results with box
    results_header_data = [['Detailed Results']]
    results_header = Table(results_header_data, colWidths=[6*inch])
    results_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(results_header)
    elements.append(Spacer(1, 0.2*inch))
    
    for idx, result in enumerate(results, 1):
        # Result header
        result_header = f"Image {idx}: {result['filename']}"
        elements.append(Paragraph(result_header, 
                                  ParagraphStyle('ResultHeader', parent=styles['Heading3'],
                                               fontSize=14, textColor=colors.HexColor('#1e3a5f'),
                                               spaceAfter=6, leading=18)))
        
        # Result details
        result_details = [
            ["Predicted Grade:", result['predicted_grade']],
            ["Quality Description:", result['description']],
            ["Processing Time:", f"{result['runtime']} seconds"],
            ["Timestamp:", result.get('timestamp', 'N/A')]
        ]
        
        result_table = Table(result_details, colWidths=[2.2*inch, 3.8*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0e0f0')),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ]))
        elements.append(result_table)
        
        # Try to add image
        try:
            img_path = os.path.join('static', 'uploads', result['image_url'].split('/')[-1])
            if os.path.exists(img_path):
                img = Image(img_path, width=2*inch, height=2*inch)
                elements.append(Spacer(1, 0.1*inch))
                elements.append(img)
        except:
            pass
        
        elements.append(Spacer(1, 0.2*inch))
        
        if idx < len(results):
            elements.append(Spacer(1, 0.1*inch))
    
    # Methodology
    elements.append(PageBreak())
    methodology_header_data = [['Methodology']]
    methodology_header = Table(methodology_header_data, colWidths=[6*inch])
    methodology_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(methodology_header)
    elements.append(Spacer(1, 0.2*inch))
    methodology_text = (
        "This wheat grain quality classification system utilizes advanced deep learning techniques "
        "to automatically assess grain quality. The system employs a ResNet50 convolutional neural "
        "network, pre-trained on ImageNet, to extract 2048-dimensional feature embeddings from each "
        "wheat grain image. These features are then normalized and fed into a machine learning "
        "classifier trained on labeled wheat grain samples to predict quality grades ranging from "
        "A (excellent) to F (rejected)."
    )
    elements.append(Paragraph(methodology_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Statistics
    stats_header_data = [['Performance Statistics']]
    stats_header = Table(stats_header_data, colWidths=[6*inch])
    stats_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(stats_header)
    elements.append(Spacer(1, 0.2*inch))
    avg_runtime = sum(r['runtime'] for r in results) / len(results)
    stats_data = [
        ["Metric", "Value"],
        ["Total Images Processed", str(len(results))],
        ["Total Processing Time", f"{data.get('total_time', 0):.2f} seconds"],
        ["Average Time per Image", f"{avg_runtime:.2f} seconds"],
        ["Unique Grades Detected", str(len(grade_counts))],
        ["Highest Grade", max(grade_counts.keys(), key=lambda x: grade_counts[x]) if grade_counts else "N/A"],
    ]
    
    stats_table = Table(stats_data, colWidths=[3.2*inch, 2.8*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 13),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
        ('TOPPADDING', (0, 0), (-1, 0), 15),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(stats_table)
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = f"Report generated on {report_date} at {report_time}"
    elements.append(Paragraph(footer_text, 
                              ParagraphStyle('Footer', parent=styles['Normal'],
                                           fontSize=9, textColor=colors.HexColor('#999999'),
                                           leading=11,
                                           alignment=TA_CENTER)))
    
    # Build PDF with custom header/footer
    def onFirstPage(canvas, doc):
        canvas.saveState()
        # Draw header line
        canvas.setStrokeColor(colors.HexColor('#1e3a5f'))
        canvas.setLineWidth(3)
        canvas.line(72, A4[1] - 50, A4[0] - 72, A4[1] - 50)
        canvas.restoreState()
    
    def onLaterPages(canvas, doc):
        canvas.saveState()
        # Draw header line on all pages
        canvas.setStrokeColor(colors.HexColor('#1e3a5f'))
        canvas.setLineWidth(3)
        canvas.line(72, A4[1] - 80, A4[0] - 72, A4[1] - 80)
        # Page number
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor('#999999'))
        page_num = canvas.getPageNumber()
        canvas.drawRightString(A4[0] - 72, 30, f"Page {page_num}")
        canvas.restoreState()
    
    doc.build(elements, onFirstPage=onFirstPage, onLaterPages=onLaterPages)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Wheat Grain Quality Classification Dashboard")
    print("="*60)
    print("\nStarting server on http://localhost:5001")
    print("Press CTRL+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
