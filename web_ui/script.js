// ===== STATE MANAGEMENT =====
let currentMode = null;
let isDetecting = false;
let detectionCount = 0;
let videoStream = null;
let animationFrameId = null;

// ===== DOM ELEMENTS =====
const detectionInterface = document.getElementById('detectionInterface');
const videoElement = document.getElementById('videoElement');
const videoCanvas = document.getElementById('videoCanvas');
const uploadZone = document.getElementById('uploadZone');
const videoFileInput = document.getElementById('videoFileInput');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const screenshotBtn = document.getElementById('screenshotBtn');
const detectionCountEl = document.getElementById('detectionCount');
const logEntries = document.getElementById('logEntries');

// ===== MODE SELECTION =====
function selectMode(mode) {
    currentMode = mode;

    // Hide mode selection
    document.querySelector('.detection-modes').style.display = 'none';

    // Show detection interface
    detectionInterface.style.display = 'block';

    // Configure interface based on mode
    if (mode === 'webcam') {
        uploadZone.style.display = 'none';
        initWebcam();
    } else if (mode === 'video') {
        uploadZone.style.display = 'flex';
        videoCanvas.style.display = 'none';
    } else if (mode === 'stream') {
        addLog('RTSP streaming coming soon!', 'info');
    }

    addLog(`${mode.toUpperCase()} mode selected`, 'info');

    // Smooth scroll to interface
    setTimeout(() => {
        detectionInterface.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

function resetMode() {
    // Stop any active detection
    if (isDetecting) {
        stopDetection();
    }

    // Reset state
    currentMode = null;

    // Show mode selection
    document.querySelector('.detection-modes').style.display = 'block';

    // Hide detection interface
    detectionInterface.style.display = 'none';

    // Reset video
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    videoElement.srcObject = null;
    videoElement.src = '';

    // Reset UI
    resetPredictionDisplay();

    addLog('Mode reset', 'info');
}

// ===== WEBCAM INITIALIZATION =====
async function initWebcam() {
    try {
        addLog('Requesting webcam access...', 'info');

        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        });

        videoStream = stream;
        videoElement.srcObject = stream;
        videoElement.classList.add('active');
        videoCanvas.style.display = 'none';

        addLog('Webcam initialized successfully', 'success');

        // Enable start button
        startBtn.disabled = false;

    } catch (error) {
        console.error('Webcam error:', error);
        addLog('Failed to access webcam: ' + error.message, 'danger');

        // Show demo message
        showDemoMode();
    }
}

// ===== VIDEO FILE UPLOAD =====
videoFileInput.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        const videoURL = URL.createObjectURL(file);
        videoElement.src = videoURL;
        videoElement.classList.add('active');
        uploadZone.style.display = 'none';
        videoCanvas.style.display = 'none';

        addLog(`Video file loaded: ${file.name}`, 'success');

        // Enable start button
        startBtn.disabled = false;
    }
});

// ===== DETECTION CONTROL =====
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
screenshotBtn.addEventListener('click', takeScreenshot);

function startDetection() {
    if (!videoElement.srcObject && !videoElement.src) {
        addLog('Please select a video source first', 'warning');
        return;
    }

    isDetecting = true;
    startBtn.style.display = 'none';
    stopBtn.style.display = 'flex';

    addLog('Detection started', 'success');

    // Start video playback if it's a file
    if (videoElement.src) {
        videoElement.play();
    }

    // Start simulated detection (in real implementation, this would call the Python backend)
    runDetectionLoop();
}

function stopDetection() {
    isDetecting = false;
    startBtn.style.display = 'flex';
    stopBtn.style.display = 'none';

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (videoElement.src) {
        videoElement.pause();
    }

    addLog('Detection stopped', 'info');
}

// ===== DETECTION LOOP (SIMULATED) =====
function runDetectionLoop() {
    if (!isDetecting) return;

    // Simulate AI detection (in real app, this sends frames to Python backend)
    simulateDetection();

    // Continue loop
    animationFrameId = requestAnimationFrame(() => {
        setTimeout(() => runDetectionLoop(), 1000); // Check every second
    });
}

function simulateDetection() {
    // Simulate random detection results
    const rand = Math.random();

    let prediction, confidence, icon;

    if (rand < 0.7) {
        // Normal behavior (70% of the time)
        prediction = 'normal';
        confidence = {
            normal: 85 + Math.random() * 12,
            suspicious: Math.random() * 10,
            theft: Math.random() * 5
        };
        icon = '✅';
    } else if (rand < 0.9) {
        // Suspicious activity (20% of the time)
        prediction = 'suspicious';
        confidence = {
            normal: 30 + Math.random() * 20,
            suspicious: 50 + Math.random() * 30,
            theft: Math.random() * 20
        };
        icon = '⚠️';
    } else {
        // Theft detected (10% of the time)
        prediction = 'theft';
        confidence = {
            normal: Math.random() * 15,
            suspicious: 20 + Math.random() * 20,
            theft: 70 + Math.random() * 25
        };
        icon = '🚨';
        detectionCount++;
        updateDetectionCount();
    }

    updatePredictionDisplay(prediction, confidence, icon);
}

// ===== UI UPDATES =====
function updatePredictionDisplay(prediction, confidence, icon) {
    const predictionStatus = document.getElementById('predictionStatus');

    // Update status card
    predictionStatus.className = `prediction-status ${prediction}`;
    predictionStatus.innerHTML = `
        <div class="status-icon">${icon}</div>
        <h4>${getPredictionLabel(prediction)}</h4>
        <p>Confidence: ${confidence[prediction].toFixed(1)}%</p>
    `;

    // Update confidence bars
    updateConfidenceBars(confidence);

    // Add to log
    if (prediction === 'theft') {
        addLog(`THEFT DETECTED with ${confidence.theft.toFixed(1)}% confidence!`, 'danger');
    } else if (prediction === 'suspicious') {
        addLog(`Suspicious activity detected (${confidence.suspicious.toFixed(1)}%)`, 'warning');
    }
}

function updateConfidenceBars(confidence) {
    // Normal
    document.getElementById('normalBar').style.width = `${confidence.normal}%`;
    document.getElementById('normalConf').textContent = `${confidence.normal.toFixed(1)}%`;

    // Suspicious
    document.getElementById('suspiciousBar').style.width = `${confidence.suspicious}%`;
    document.getElementById('suspiciousConf').textContent = `${confidence.suspicious.toFixed(1)}%`;

    // Theft
    document.getElementById('theftBar').style.width = `${confidence.theft}%`;
    document.getElementById('theftConf').textContent = `${confidence.theft.toFixed(1)}%`;
}

function resetPredictionDisplay() {
    const predictionStatus = document.getElementById('predictionStatus');
    predictionStatus.className = 'prediction-status';
    predictionStatus.innerHTML = `
        <div class="status-icon">⏳</div>
        <h4>Waiting for detection...</h4>
        <p>Start analysis to see results</p>
    `;

    updateConfidenceBars({ normal: 0, suspicious: 0, theft: 0 });
}

function getPredictionLabel(prediction) {
    const labels = {
        'normal': 'Normal Behavior',
        'suspicious': 'Suspicious Activity',
        'theft': '🚨 THEFT DETECTED'
    };
    return labels[prediction] || 'Unknown';
}

function updateDetectionCount() {
    detectionCountEl.textContent = detectionCount;

    // Animate
    detectionCountEl.style.transform = 'scale(1.3)';
    detectionCountEl.style.color = 'var(--danger)';
    setTimeout(() => {
        detectionCountEl.style.transform = 'scale(1)';
        detectionCountEl.style.color = 'var(--text-primary)';
    }, 300);
}

// ===== LOGGING =====
function addLog(message, type = 'info') {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour12: false });

    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.innerHTML = `
        <span class="log-time">${timeStr}</span>
        <span class="log-message">${message}</span>
    `;

    logEntries.insertBefore(logEntry, logEntries.firstChild);

    // Keep only last 20 logs
    while (logEntries.children.length > 20) {
        logEntries.removeChild(logEntries.lastChild);
    }
}

// ===== SCREENSHOT =====
function takeScreenshot() {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth || 1280;
    canvas.height = videoElement.videoHeight || 720;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);

    // Download
    canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `detection_${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);

        addLog('Screenshot saved', 'success');
    });
}

// ===== DEMO MODE (when webcam not available) =====
function showDemoMode() {
    const canvas = videoCanvas;
    const ctx = canvas.getContext('2d');

    canvas.width = 1280;
    canvas.height = 720;
    canvas.classList.add('active');
    canvas.style.display = 'block';
    videoElement.style.display = 'none';

    // Draw demo message
    function drawDemo() {
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#6366f1';
        ctx.font = 'bold 48px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('DEMO MODE', canvas.width / 2, canvas.height / 2 - 40);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '24px Inter';
        ctx.fillText('Webcam not available', canvas.width / 2, canvas.height / 2 + 20);
        ctx.fillText('Upload a video file to test detection', canvas.width / 2, canvas.height / 2 + 60);

        // Animated gradient
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        const time = Date.now() / 1000;
        gradient.addColorStop(0, `hsl(${time * 50 % 360}, 70%, 50%)`);
        gradient.addColorStop(1, `hsl(${(time * 50 + 180) % 360}, 70%, 50%)`);

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 4;
        ctx.strokeRect(20, 20, canvas.width - 40, canvas.height - 40);

        requestAnimationFrame(drawDemo);
    }

    drawDemo();
    startBtn.disabled = false;
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    addLog('System initialized', 'success');

    // Animate metric bars on load
    setTimeout(() => {
        document.querySelectorAll('.metric-fill').forEach(fill => {
            const width = fill.style.width;
            fill.style.width = '0%';
            setTimeout(() => {
                fill.style.width = width;
            }, 100);
        });
    }, 500);
});

// ===== REAL BACKEND INTEGRATION (Template) =====
/*
// In a production environment, you would call your Python backend like this:

async function callDetectionAPI(frameData) {
    try {
        const response = await fetch('http://localhost:5000/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                frame: frameData,
                timestamp: Date.now()
            })
        });
        
        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error('API Error:', error);
        addLog('Backend connection failed', 'danger');
        return null;
    }
}

function realDetectionLoop() {
    if (!isDetecting) return;
    
    // Capture frame from video
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    
    // Convert to base64
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Send to backend
    callDetectionAPI(frameData).then(result => {
        if (result) {
            updatePredictionDisplay(result.prediction, result.confidence, result.icon);
        }
    });
    
    // Continue loop
    setTimeout(() => realDetectionLoop(), 1000);
}
*/
