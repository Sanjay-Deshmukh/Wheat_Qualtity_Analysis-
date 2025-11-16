// Global state
let selectedFiles = [];
let currentResults = [];
let processing = false;
let currentData = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const processBtn = document.getElementById('processBtn');
const gradePreview = document.getElementById('gradePreview');
const predictionCard = document.getElementById('predictionCard');
const predictionImage = document.getElementById('predictionImage');
const gradeBadge = document.getElementById('gradeBadge');
const gradeDescription = document.getElementById('gradeDescription');
const progressFill = document.getElementById('progressFill');
const imageCount = document.getElementById('imageCount');
const runtimeValue = document.getElementById('runtimeValue');
const totalTimeValue = document.getElementById('totalTimeValue');
const previewLink = document.getElementById('previewLink');
const reportCard = document.getElementById('reportCard');
const downloadReportBtn = document.getElementById('downloadReportBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        if (!processing) {
            fileInput.click();
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFiles(Array.from(e.target.files));
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!processing) {
            uploadArea.classList.add('dragover');
        }
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (!processing) {
            const files = Array.from(e.dataTransfer.files).filter(file => 
                file.type.startsWith('image/')
            );
            handleFiles(files);
        }
    });

    // Process button
    processBtn.addEventListener('click', processImages);

    // Preview link
    previewLink.addEventListener('click', (e) => {
        e.preventDefault();
        if (currentResults.length > 0) {
            scrollToPrediction();
        }
    });

    // Download report button
    downloadReportBtn.addEventListener('click', generateReport);
}

// Handle files
function handleFiles(files) {
    files.forEach(file => {
        if (file.size > 16 * 1024 * 1024) {
            alert(`File ${file.name} is too large. Maximum size is 16MB.`);
            return;
        }

        if (!selectedFiles.find(f => f.name === file.name && f.size === file.size)) {
            selectedFiles.push(file);
        }
    });

    updateFileList();
    updateProcessButton();
}

// Update file list display
function updateFileList() {
    if (selectedFiles.length === 0) {
        fileList.innerHTML = '';
        return;
    }

    fileList.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-item">
            <span class="file-name">${file.name}</span>
            <button class="file-remove" onclick="removeFile(${index})" title="Remove">×</button>
        </div>
    `).join('');
}

// Remove file
function removeFile(index) {
    if (processing) return;
    selectedFiles.splice(index, 1);
    updateFileList();
    updateProcessButton();
}

// Update process button state
function updateProcessButton() {
    processBtn.disabled = selectedFiles.length === 0 || processing;
}

// Process images
async function processImages() {
    if (selectedFiles.length === 0 || processing) return;

    processing = true;
    updateProcessButton();
    uploadArea.classList.add('loading');

    // Reset UI
    progressFill.style.width = '0%';
    progressFill.textContent = '';
    currentResults = [];

    // Create FormData
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('file', file);
    });

    // Animate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 5;
        if (progress <= 90) {
            progressFill.style.width = progress + '%';
            progressFill.textContent = `${progress}%`;
        }
    }, 200);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to process images');
        }

        const data = await response.json();
        currentResults = data.results;

        // Complete progress
        progressFill.style.width = '100%';
        progressFill.textContent = '100%';

        // Update UI with results
        setTimeout(() => {
            displayResults(data);
            processing = false;
            updateProcessButton();
            uploadArea.classList.remove('loading');
        }, 500);

    } catch (error) {
        clearInterval(progressInterval);
        processing = false;
        updateProcessButton();
        uploadArea.classList.remove('loading');
        alert('Error: ' + error.message);
        console.error('Error:', error);
    }
}

// Display results
function displayResults(data) {
    currentData = data;
    
    // Show prediction card with first result
    if (data.results.length > 0 && !data.results[0].error) {
        const firstResult = data.results[0];
        displayPredictionResult(firstResult);
    }

    // Update grade preview
    updateGradePreview(data.results);

    // Update data summary
    updateDataSummary(data);
    
    // Show report card
    reportCard.style.display = 'block';
}

// Display prediction result
function displayPredictionResult(result) {
    predictionCard.style.display = 'block';
    predictionImage.src = result.image_url;
    predictionImage.alt = result.filename;
    
    // Update grade badge
    gradeBadge.textContent = `Grade ${result.predicted_grade}`;
    gradeBadge.className = `grade-badge grade-${result.predicted_grade}`;
    
    // Update description
    gradeDescription.textContent = result.description;
    
    // Scroll to prediction
    setTimeout(() => {
        scrollToPrediction();
    }, 100);
}

// Update grade preview
function updateGradePreview(results) {
    const validResults = results.filter(r => !r.error);
    
    if (validResults.length === 0) {
        gradePreview.innerHTML = '<p class="empty-message">No images processed yet</p>';
        return;
    }

    gradePreview.innerHTML = validResults.map((result, index) => `
        <div class="grade-preview-item" onclick="showPrediction(${index})" title="${result.filename}">
            <img src="${result.image_url}" alt="${result.filename}">
        </div>
    `).join('');
}

// Show prediction for specific image
function showPrediction(index) {
    if (currentResults[index] && !currentResults[index].error) {
        displayPredictionResult(currentResults[index]);
    }
}

// Update data summary
function updateDataSummary(data) {
    const validResults = data.results.filter(r => !r.error);
    
    // Update image count
    imageCount.textContent = validResults.length;
    
    // Calculate average runtime
    if (validResults.length > 0) {
        const avgRuntime = validResults.reduce((sum, r) => sum + r.runtime, 0) / validResults.length;
        runtimeValue.textContent = avgRuntime.toFixed(2);
    } else {
        runtimeValue.textContent = '0.00';
    }
    
    // Update total time
    totalTimeValue.textContent = data.total_time.toFixed(2);
    
    // Update progress bar (reset after a moment)
    setTimeout(() => {
        progressFill.style.width = '0%';
        progressFill.textContent = '';
    }, 2000);
}

// Scroll to prediction
function scrollToPrediction() {
    predictionCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Generate and download report
async function generateReport() {
    if (!currentResults || currentResults.length === 0) {
        alert('No results available to generate report');
        return;
    }

    downloadReportBtn.disabled = true;
    downloadReportBtn.innerHTML = '<span class="btn-icon">⏳</span><span>Generating Report...</span>';

    try {
        const response = await fetch('/api/generate-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                results: currentResults,
                total_time: currentData ? currentData.total_time : 0,
                count: currentResults.length
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate report');
        }

        const data = await response.json();
        
        // Download the PDF
        const link = document.createElement('a');
        link.href = data.report_url;
        link.download = data.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        // Show success message
        downloadReportBtn.innerHTML = '<span class="btn-icon">✓</span><span>Report Downloaded!</span>';
        setTimeout(() => {
            downloadReportBtn.innerHTML = '<span class="btn-icon">📄</span><span>Download PDF Report</span>';
            downloadReportBtn.disabled = false;
        }, 2000);

    } catch (error) {
        alert('Error generating report: ' + error.message);
        console.error('Error:', error);
        downloadReportBtn.disabled = false;
        downloadReportBtn.innerHTML = '<span class="btn-icon">📄</span><span>Download PDF Report</span>';
    }
}

// Handle image load errors
document.addEventListener('error', (e) => {
    if (e.target.tagName === 'IMG') {
        e.target.src = 'data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%27200%27 height=%27200%27%3E%3Crect fill=%27%23f0f0f0%27 width=%27200%27 height=%27200%27/%3E%3Ctext fill=%27%23999%27 font-family=%27Arial%27 font-size=%2714%27 x=%2750%25%27 y=%2750%25%27 text-anchor=%27middle%27 dominant-baseline=%27middle%27%3EImage not found%3C/text%3E%3C/svg%3E';
    }
}, true);

