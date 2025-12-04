/**
 * Chart initialization and visualization for dashboard
 */

let riskGaugeChart = null;
let confidenceChart = null;
let eventChart = null;
let signalChart = null;

// Initialize all charts
function initializeCharts(data) {
    const prediction = data.prediction;
    
    // Initialize risk gauge
    initializeRiskGauge(prediction.risk_score);
    
    // Initialize confidence chart
    initializeConfidenceChart(prediction);
    
    // Initialize event chart
    initializeEventChart(prediction);
    
    // Initialize signal chart (if signal data available)
    if (data.signal_info) {
        initializeSignalChart(data);
    }
}

// Initialize risk gauge
function initializeRiskGauge(riskScore) {
    const ctx = document.getElementById('risk-gauge');
    if (!ctx) return;
    
    // Destroy existing chart if any
    if (riskGaugeChart) {
        riskGaugeChart.destroy();
    }
    
    const riskPercent = riskScore * 100;
    const color = riskScore > 0.7 ? '#ef4444' : riskScore > 0.4 ? '#f59e0b' : '#10b981';
    
    riskGaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [riskPercent, 100 - riskPercent],
                backgroundColor: [color, '#e2e8f0'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '75%',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            }
        },
        plugins: [{
            id: 'centerText',
            beforeDraw: (chart) => {
                const ctx = chart.ctx;
                const centerX = chart.chartArea.left + (chart.chartArea.right - chart.chartArea.left) / 2;
                const centerY = chart.chartArea.top + (chart.chartArea.bottom - chart.chartArea.top) / 2;
                
                ctx.save();
                ctx.font = 'bold 2rem sans-serif';
                ctx.fillStyle = color;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(riskPercent.toFixed(1) + '%', centerX, centerY - 10);
                
                ctx.font = '1rem sans-serif';
                ctx.fillStyle = '#64748b';
                ctx.fillText('Risk Score', centerX, centerY + 20);
                ctx.restore();
            }
        }]
    });
}

// Initialize confidence chart
function initializeConfidenceChart(prediction) {
    const ctx = document.getElementById('confidence-chart');
    if (!ctx) return;
    
    if (confidenceChart) {
        confidenceChart.destroy();
    }
    
    const confidences = [
        {
            label: 'Event',
            value: prediction.event_confidence * 100
        },
        {
            label: 'Damage',
            value: prediction.damage_confidence * 100
        },
        {
            label: 'Sensor',
            value: prediction.sensor_confidence * 100
        }
    ];
    
    confidenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: confidences.map(c => c.label),
            datasets: [{
                label: 'Confidence (%)',
                data: confidences.map(c => c.value),
                backgroundColor: [
                    '#2563eb',
                    '#10b981',
                    '#f59e0b'
                ],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Initialize event chart
function initializeEventChart(prediction) {
    const ctx = document.getElementById('event-chart');
    if (!ctx) return;
    
    if (eventChart) {
        eventChart.destroy();
    }
    
    // Create a simple pie chart for event type
    // In a real scenario, you might want to show all possible events with their probabilities
    eventChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: [prediction.event_type, 'Other Events'],
            datasets: [{
                data: [prediction.event_confidence * 100, (1 - prediction.event_confidence) * 100],
                backgroundColor: [
                    '#2563eb',
                    '#e2e8f0'
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return label + ': ' + value.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Initialize signal chart
function initializeSignalChart(data) {
    const ctx = document.getElementById('signal-chart');
    if (!ctx) return;
    
    if (signalChart) {
        signalChart.destroy();
    }
    
    // For now, create a placeholder chart
    // In a real implementation, you would fetch the actual signal data
    // For demonstration, create a sample waveform
    const sampleData = generateSampleWaveform(data.signal_info.length || 1000);
    
    signalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: Math.min(sampleData.length, 1000) }, (_, i) => i),
            datasets: [{
                label: 'Signal Amplitude',
                data: sampleData.slice(0, 1000), // Limit to 1000 points for performance
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Sample Index'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Amplitude'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });
}

// Generate sample waveform (placeholder)
function generateSampleWaveform(length) {
    const data = [];
    for (let i = 0; i < length; i++) {
        // Generate a simple sine wave with some noise
        const value = Math.sin(i * 0.1) + Math.random() * 0.3;
        data.push(value);
    }
    return data;
}

// Cleanup charts
function cleanupCharts() {
    if (riskGaugeChart) {
        riskGaugeChart.destroy();
        riskGaugeChart = null;
    }
    if (confidenceChart) {
        confidenceChart.destroy();
        confidenceChart = null;
    }
    if (eventChart) {
        eventChart.destroy();
        eventChart = null;
    }
    if (signalChart) {
        signalChart.destroy();
        signalChart = null;
    }
}





