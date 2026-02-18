// DeepGuard Video Analysis Script

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loadingState = document.getElementById('loading-state');
const resultState = document.getElementById('result-state');

// Drag & Drop Events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    preventDefaults(e);
    dropZone.classList.add('dragover');
}

function unhighlight(e) {
    preventDefaults(e);
    dropZone.classList.remove('dragover');
}

dropZone.addEventListener('dragover', highlight, false);
dropZone.addEventListener('dragleave', unhighlight, false);
dropZone.addEventListener('drop', handleDrop, false);

// Note: Click is handled natively by <label for="file-input">

fileInput.addEventListener('change', handleFiles, false);


function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files: files } });
}

function handleFiles(e) {
    const files = e.target.files;
    console.log("File change detected:", files);
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function uploadFile(file) {
    console.log("Starting upload for:", file.name);

    // UI Update
    dropZone.style.display = 'none';
    loadingState.style.display = 'block';
    resultState.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:8000/analyze/video', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            showResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert("Analysis failed. Is the server running? Check console for details.");
            resetUpload();
        });
}

function showResults(data) {
    loadingState.style.display = 'none';
    resultState.style.display = 'block';

    // Update Text
    const statusEl = document.getElementById('result-status');
    const scoreEl = document.getElementById('result-score');
    const listEl = document.getElementById('anomaly-list');
    const container = document.getElementById('result-state');

    statusEl.innerText = data.status;
    scoreEl.innerText = data.risk_score;

    // Colors based on risk
    container.classList.remove('result-crit', 'result-susp');
    statusEl.style.color = 'var(--text-heading)';

    if (data.risk_score > 75) {
        container.classList.add('result-crit');
        statusEl.style.color = '#ef4444';
    } else if (data.risk_score > 40) {
        container.classList.add('result-susp');
        statusEl.style.color = '#f59e0b';
    } else {
        statusEl.style.color = 'var(--primary)';
    }

    // Anomalies
    listEl.innerHTML = '';
    if (data.anomalies && data.anomalies.length > 0) {
        data.anomalies.forEach(anomaly => {
            const li = document.createElement('li');
            li.style.padding = '8px 0';
            li.style.borderBottom = '1px solid rgba(148, 163, 184, 0.1)';
            li.innerText = `⚠️ ${anomaly}`;
            listEl.appendChild(li);
        });
    } else {
        listEl.innerHTML = '<li style="padding: 8px 0;">✅ No significant anomalies detected.</li>';
    }

    // Details
    document.getElementById('video-risk-detail').innerText = `Video Risk: ${data.details.video_risk}%`;
    document.getElementById('audio-risk-detail').innerText = `Audio Risk: ${data.details.audio_risk}%`;
}

function resetUpload() {
    dropZone.style.display = 'block';
    loadingState.style.display = 'none';
    resultState.style.display = 'none';
    fileInput.value = ''; // clear input
}
