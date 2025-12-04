/**
 * Main application logic for Universal Optics Detection and Degradation Tracker
 */

// Global state
let currentResults = null;
let currentSignal = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeFileUploads();
    initializeForms();
    initializeModal();
    checkHealth();
});

// Health check
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        if (!data.model_loaded) {
            console.warn('Model not loaded yet');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Initialize file upload areas
function initializeFileUploads() {
    // Standard mode
    const standardArea = document.getElementById('standard-upload-area');
    const standardFile = document.getElementById('standard-file');
    const standardFileName = document.getElementById('standard-file-name');
    
    standardArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        standardArea.style.borderColor = '#2563eb';
    });
    
    standardArea.addEventListener('dragleave', () => {
        standardArea.style.borderColor = '#e2e8f0';
    });
    
    standardArea.addEventListener('drop', (e) => {
        e.preventDefault();
        standardArea.style.borderColor = '#e2e8f0';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            standardFile.files = files;
            handleFileSelect(standardFile, standardFileName, 'standard');
        }
    });
    
    standardFile.addEventListener('change', (e) => {
        handleFileSelect(e.target, standardFileName, 'standard');
    });
    
    // Universal mode
    const universalArea = document.getElementById('universal-upload-area');
    const universalFile = document.getElementById('universal-file');
    const universalFileName = document.getElementById('universal-file-name');
    
    universalArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        universalArea.style.borderColor = '#2563eb';
    });
    
    universalArea.addEventListener('dragleave', () => {
        universalArea.style.borderColor = '#e2e8f0';
    });
    
    universalArea.addEventListener('drop', (e) => {
        e.preventDefault();
        universalArea.style.borderColor = '#e2e8f0';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            universalFile.files = files;
            handleFileSelect(universalFile, universalFileName, 'universal');
        }
    });
    
    universalFile.addEventListener('change', (e) => {
        handleFileSelect(e.target, universalFileName, 'universal');
    });
}

// Handle file selection
function handleFileSelect(input, fileNameElement, mode) {
    const file = input.files[0];
    if (file) {
        fileNameElement.textContent = file.name;
        previewFile(file, mode);
    }
}

// Preview file data
async function previewFile(file, mode) {
    const previewId = mode === 'standard' ? 'standard-preview' : 'universal-preview';
    const previewContent = mode === 'standard' ? 'standard-preview-content' : 'universal-preview-content';
    const previewDiv = document.getElementById(previewId);
    const previewText = document.getElementById(previewContent);
    
    try {
        const text = await file.text();
        const lines = text.split('\n').slice(0, 10);
        previewText.textContent = lines.join('\n') + (text.split('\n').length > 10 ? '\n...' : '');
        previewDiv.style.display = 'block';
    } catch (error) {
        previewText.textContent = 'Unable to preview file';
        previewDiv.style.display = 'block';
    }
}

// Initialize forms
function initializeForms() {
    // Standard form
    document.getElementById('standard-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await handlePrediction('standard');
    });
    
    // Universal form
    document.getElementById('universal-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await handlePrediction('universal');
    });
    
    // Download results button
    document.getElementById('download-results-btn').addEventListener('click', downloadResults);
    
    // Generate dashboard button
    document.getElementById('generate-dashboard-btn').addEventListener('click', generateDashboard);
}

// Handle prediction
async function handlePrediction(mode) {
    const form = mode === 'standard' ? document.getElementById('standard-form') : document.getElementById('universal-form');
    const fileInput = mode === 'standard' ? document.getElementById('standard-file') : document.getElementById('universal-file');
    
    if (!fileInput.files[0]) {
        showError('Please select a file');
        return;
    }
    
    // Show loading
    showLoading();
    
    // Create form data
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    if (mode === 'standard') {
        const samplingRate = document.getElementById('standard-sampling-rate').value;
        const isMultichannel = document.getElementById('standard-multichannel').checked;
        formData.append('sampling_rate', samplingRate);
        formData.append('is_multichannel', isMultichannel);
    } else {
        const samplingRate = document.getElementById('universal-sampling-rate').value;
        const isMultichannel = document.getElementById('universal-multichannel').checked;
        if (samplingRate) {
            formData.append('original_sampling_rate', samplingRate);
        }
        formData.append('is_multichannel', isMultichannel);
    }
    
    // Update loading status
    updateLoadingStatus('Uploading file...', 20);
    
    try {
        const endpoint = mode === 'standard' ? '/api/predict/standard' : '/api/predict/universal';
        
        updateLoadingStatus('Processing signal...', 40);
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        updateLoadingStatus('Extracting features...', 60);
        
        const data = await response.json();
        
        updateLoadingStatus('Running model...', 80);
        
        if (data.success) {
            updateLoadingStatus('Finalizing results...', 100);
            
            // Store results
            currentResults = data;
            
            // Store signal if available
            if (data.signal_info) {
                currentSignal = data.signal_info;
            }
            
            // Hide loading and show results
            setTimeout(() => {
                hideLoading();
                displayResults(data);
                scrollToResults();
            }, 500);
        } else {
            hideLoading();
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        hideLoading();
        showError('Network error: ' + error.message);
    }
}

// Show loading
function showLoading() {
    document.getElementById('loading-section').style.display = 'block';
    updateLoadingStatus('Initializing...', 0);
    scrollToLoading();
}

// Hide loading
function hideLoading() {
    document.getElementById('loading-section').style.display = 'none';
}

// Update loading status
function updateLoadingStatus(message, progress) {
    document.getElementById('loading-status').textContent = message;
    document.getElementById('progress-bar').style.width = progress + '%';
    
    // Update loading message based on progress
    let detailMessage = '';
    if (progress < 30) {
        detailMessage = 'Preparing model...';
    } else if (progress < 60) {
        detailMessage = 'Analyzing signal features...';
    } else if (progress < 90) {
        detailMessage = 'Running neural network...';
    } else {
        detailMessage = 'Almost done...';
    }
    document.getElementById('loading-message').textContent = detailMessage;
}

// Display results
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    const resultsGrid = document.getElementById('results-grid');
    
    resultsSection.style.display = 'block';
    resultsGrid.innerHTML = '';
    
    const prediction = data.prediction;
    
    // Define metrics with explanations
    const metrics = [
        {
            key: 'event_type',
            title: 'Event Type',
            value: prediction.event_type,
            explanation: 'Classification of detected event. Possible events include: car, walk, running, longboard, fence, manipulation, construction, openclose, regular, background, dig, knock, water, shake, and walk_phi.'
        },
        {
            key: 'event_confidence',
            title: 'Event Confidence',
            value: (prediction.event_confidence * 100).toFixed(2) + '%',
            explanation: 'Model\'s confidence in event classification. Range: 0-100%. Higher values indicate more reliable predictions.'
        },
        {
            key: 'risk_score',
            title: 'Risk Score',
            value: (prediction.risk_score * 100).toFixed(2) + '%',
            explanation: 'Predicted risk level. Range: 0-100% (0-1). Higher values indicate greater risk. This is a regression output from the model.',
            riskClass: prediction.risk_score > 0.7 ? 'high' : prediction.risk_score > 0.4 ? 'medium' : 'low'
        },
        {
            key: 'damage_type',
            title: 'Damage Type',
            value: prediction.damage_type,
            explanation: 'Type of fiber damage detected. Possible types: clean (no damage), reflective (reflective event), non-reflective (non-reflective event), saturated (signal saturation).'
        },
        {
            key: 'damage_confidence',
            title: 'Damage Confidence',
            value: (prediction.damage_confidence * 100).toFixed(2) + '%',
            explanation: 'Model\'s confidence in damage classification. Range: 0-100%. Higher values indicate more reliable damage detection.'
        },
        {
            key: 'sensor_type',
            title: 'Sensor Type',
            value: prediction.sensor_type,
            explanation: 'Identified sensor type. Possible types: DAS (Distributed Acoustic Sensing), Phi-OTDR (Phase-sensitive OTDR), or OTDR (Optical Time Domain Reflectometry).'
        },
        {
            key: 'sensor_confidence',
            title: 'Sensor Confidence',
            value: (prediction.sensor_confidence * 100).toFixed(2) + '%',
            explanation: 'Model\'s confidence in sensor identification. Range: 0-100%. Higher values indicate more reliable sensor type detection.'
        }
    ];
    
    // Create metric cards
    metrics.forEach(metric => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <div class="metric-header">
                <div>
                    <div class="metric-title">${metric.title}</div>
                    <div class="metric-value ${metric.riskClass ? 'risk-' + metric.riskClass : ''}">${metric.value}</div>
                </div>
                <span class="expand-icon">▼</span>
            </div>
            <div class="metric-explanation">
                ${metric.explanation}
            </div>
        `;
        
        // Add click handler for expansion
        card.addEventListener('click', () => {
            card.classList.toggle('expanded');
        });
        
        resultsGrid.appendChild(card);
    });
    
    // Show preprocessing info if available (universal mode)
    if (data.preprocessing_info) {
        const infoCard = document.createElement('div');
        infoCard.className = 'metric-card';
        infoCard.innerHTML = `
            <div class="metric-header">
                <div>
                    <div class="metric-title">Preprocessing Information</div>
                    <div class="metric-value" style="font-size: 1rem;">Universal Mode Applied</div>
                </div>
                <span class="expand-icon">▼</span>
            </div>
            <div class="metric-explanation">
                <strong>Resample Ratio:</strong> ${data.preprocessing_info.resample_ratio.toFixed(4)}<br>
                <strong>Length Ratio:</strong> ${data.preprocessing_info.length_ratio.toFixed(4)}<br>
                ${data.preprocessing_info.warnings && data.preprocessing_info.warnings.length > 0 ? 
                    '<strong>Warnings:</strong> ' + data.preprocessing_info.warnings.join(', ') : 
                    'No warnings'}
            </div>
        `;
        
        infoCard.addEventListener('click', () => {
            infoCard.classList.toggle('expanded');
        });
        
        resultsGrid.appendChild(infoCard);
    }
    
    scrollToResults();
}

// Scroll to results
function scrollToResults() {
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Scroll to loading
function scrollToLoading() {
    document.getElementById('loading-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Download results
function downloadResults() {
    if (!currentResults) {
        showError('No results to download');
        return;
    }
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'prediction_results.json';
    link.click();
    URL.revokeObjectURL(url);
}

// Generate dashboard
function generateDashboard() {
    if (!currentResults) {
        showError('No results available for dashboard');
        return;
    }
    
    const dashboardSection = document.getElementById('dashboard-section');
    dashboardSection.style.display = 'block';
    
    // Initialize charts
    initializeCharts(currentResults);
    
    // Load training stats
    loadTrainingStats();
    
    // Scroll to dashboard
    dashboardSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Initialize modal
function initializeModal() {
    const modal = document.getElementById('error-modal');
    const closeBtn = document.querySelector('.close-modal');
    
    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });
    
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// Show error
function showError(message) {
    const modal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    modal.style.display = 'block';
}

// Load training stats
async function loadTrainingStats() {
    try {
        const response = await fetch('/api/training-stats');
        const data = await response.json();
        
        const statsContainer = document.getElementById('training-stats');
        statsContainer.innerHTML = '';
        
        data.datasets.forEach(dataset => {
            const statCard = document.createElement('div');
            statCard.className = 'stat-card';
            statCard.innerHTML = `
                <h4>${dataset.name}</h4>
                <p><strong>Task:</strong> ${dataset.task}</p>
                <p><strong>Samples:</strong> ${dataset.samples.toLocaleString()}</p>
                <p><strong>Classes:</strong> ${dataset.classes}</p>
                <p><strong>Accuracy:</strong> ${dataset.accuracy}%</p>
            `;
            statsContainer.appendChild(statCard);
        });
        
        // Add overall stats
        const overallCard = document.createElement('div');
        overallCard.className = 'stat-card';
        overallCard.innerHTML = `
            <h4>Overall Performance</h4>
            <p><strong>Total Samples:</strong> ${data.total_samples.toLocaleString()}</p>
            <p><strong>Risk Regression MSE:</strong> ${data.risk_mse}</p>
        `;
        statsContainer.appendChild(overallCard);
    } catch (error) {
        console.error('Failed to load training stats:', error);
    }
}





