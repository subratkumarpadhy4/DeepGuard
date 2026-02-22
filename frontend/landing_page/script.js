/**
 * DeepGuard Landing Page Interactivity
 * Handles Sidebar Navigation, Theme Toggling, and View Switching
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const menuBtn = document.getElementById('menuBtn');
    const sidebar = document.getElementById('sidebar');
    const closeSidebar = document.getElementById('closeSidebar');
    const sidebarBackdrop = document.getElementById('sidebarBackdrop');
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;

    // View Switching
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    const views = document.querySelectorAll('.view-container');

    // Drag & Drop
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');

    // --- Sidebar Logic ---
    function openSidebar() {
        sidebar.classList.add('active');
        sidebarBackdrop.style.display = 'flex';
    }

    function closeSidebarMenu() {
        sidebar.classList.remove('active');
        sidebarBackdrop.style.display = 'none';
    }

    if (menuBtn) {
        menuBtn.addEventListener('click', openSidebar);
    }

    if (closeSidebar) {
        closeSidebar.addEventListener('click', closeSidebarMenu);
    }

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener('click', closeSidebarMenu);
    }

    // --- Theme Toggling ---
    const savedTheme = localStorage.getItem('deepguard-theme') || 'light';
    if (savedTheme === 'dark') {
        body.classList.add('dark-theme');
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            body.classList.toggle('dark-theme');
            const newTheme = body.classList.contains('dark-theme') ? 'dark' : 'light';
            localStorage.setItem('deepguard-theme', newTheme);
        });
    }

    // --- View Switching ---
    sidebarLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            // Remove active class from all links
            sidebarLinks.forEach(l => l.classList.remove('active'));
            // Add active class to clicked link
            link.classList.add('active');

            const targetViewId = link.getAttribute('data-view');

            // Hide all views
            views.forEach(view => {
                view.classList.remove('active');
            });

            // Show target view
            const targetView = document.getElementById(targetViewId + 'View');
            if (targetView) {
                targetView.classList.add('active');
            }

            // Close sidebar on mobile/desktop after selection
            closeSidebarMenu();
        });
    });

    // --- Drag & Drop Upload Logic ---
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.background = 'var(--accent-glow)';
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.style.background = '';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.background = '';

            if (e.dataTransfer.files.length > 0) {
                uploadFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        });
    }

    // Initial binding for Investigate/Reset buttons
    const invBtn = document.getElementById('investigate-btn');
    if (invBtn) invBtn.addEventListener('click', toggleInvestigate);
});

// --- Functional Modules (Backend Integration) ---

// Dot animation for loading state
let dotInterval;
function startDotAnimation() {
    const dotsEl = document.getElementById('loading-dots');
    if (!dotsEl) return;
    let dots = '';
    dotInterval = setInterval(() => {
        dots = dots.length >= 3 ? '' : dots + '.';
        dotsEl.innerText = dots;
    }, 400);
}

function stopDotAnimation() {
    clearInterval(dotInterval);
}

function uploadFile(file) {
    console.log("Starting upload for:", file.name);

    const dropZone = document.getElementById('drop-zone');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');

    // Start dots
    startDotAnimation();

    // Create a local preview URL
    const videoUrl = URL.createObjectURL(file);
    const videoPreview = document.getElementById('result-video-preview');
    if (videoPreview) videoPreview.src = videoUrl;

    // UI Update
    if (dropZone) dropZone.style.display = 'none';
    if (loadingState) loadingState.style.display = 'block';
    if (resultState) resultState.style.display = 'none';

    // FormData for upload
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:8000/analyze/video', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                showResults(data);
            } else {
                alert("Error: " + (data.message || "Analysis failed"));
                resetUpload();
            }
        })
        .catch(err => {
            console.error("Upload failed:", err);
            alert("Upload failed. Ensure backend is running at http://127.0.0.1:8000");
            resetUpload();
        });
}

function showResults(data) {
    stopDotAnimation();
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');

    if (loadingState) loadingState.style.display = 'none';
    if (resultState) resultState.style.display = 'block';

    const statusEl = document.getElementById('result-status');
    const scoreEl = document.getElementById('result-score');
    const listEl = document.getElementById('anomaly-list');
    const container = document.getElementById('result-state');
    const overlay = document.getElementById('xai-overlay');
    const video = document.getElementById('result-video-preview');
    const investBtn = document.getElementById('investigate-btn');

    if (overlay) overlay.innerHTML = '';
    if (listEl) listEl.innerHTML = '';

    // Handle Investigation Button Visibility
    if (data.evidence && data.evidence.length > 0) {
        if (investBtn) investBtn.style.display = 'inline-flex';

        const performDraw = () => drawEvidence(data.evidence, video, overlay);
        if (video.readyState >= 1) {
            performDraw();
        } else {
            video.onloadedmetadata = performDraw;
        }
    }

    // Set Status and Score
    if (statusEl) statusEl.innerText = data.verdict;
    if (scoreEl) scoreEl.innerText = data.risk_score;

    // List Anomalies
    if (listEl && data.anomalies) {
        data.anomalies.forEach(anom => {
            const li = document.createElement('div');
            li.style.padding = '8px 12px';
            li.style.background = 'rgba(239, 68, 68, 0.1)';
            li.style.borderLeft = '3px solid #ef4444';
            li.style.borderRadius = '4px';
            li.style.fontSize = '0.85rem';
            li.style.marginBottom = '8px';
            li.style.color = 'var(--text-main)';
            li.innerHTML = `⚠️ ${anom}`;
            listEl.appendChild(li);
        });
    }

    // Colors based on risk
    if (container) {
        container.classList.remove('result-crit', 'result-susp');
        if (data.risk_score > 75) {
            container.classList.add('result-crit');
            if (statusEl) statusEl.style.color = '#ef4444';
        } else if (data.risk_score > 40) {
            container.classList.add('result-susp');
            if (statusEl) statusEl.style.color = '#f59e0b';
        } else {
            if (statusEl) statusEl.style.color = '#10b981';
        }
    }

    // Details
    const videoRiskEl = document.getElementById('video-risk-detail');
    const audioRiskEl = document.getElementById('audio-risk-detail');
    if (videoRiskEl) videoRiskEl.innerText = `Video Risk: ${data.details.video_risk}%`;
    if (audioRiskEl) audioRiskEl.innerText = `Audio Risk: ${data.details.audio_risk}%`;
}

function drawEvidence(evidence, video, overlay) {
    if (!overlay || !video) return;
    overlay.innerHTML = '';
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const rect = video.getBoundingClientRect();

    if (vw === 0 || vh === 0) return;

    const videoRatio = vw / vh;
    const displayRatio = rect.width / rect.height;

    let actualW, actualH, offsetX = 0, offsetY = 0;

    if (videoRatio > displayRatio) {
        actualW = rect.width;
        actualH = rect.width / videoRatio;
        offsetY = (rect.height - actualH) / 2;
    } else {
        actualH = rect.height;
        actualW = rect.height * videoRatio;
        offsetX = (rect.width - actualW) / 2;
    }

    const scale = actualW / vw;

    evidence.forEach(ev => {
        const box = document.createElement('div');
        box.style.position = 'absolute';
        box.style.left = `${offsetX + (ev.box[0] * scale)}px`;
        box.style.top = `${offsetY + (ev.box[1] * scale)}px`;
        box.style.width = `${ev.box[2] * scale}px`;
        box.style.height = `${ev.box[3] * scale}px`;
        box.style.border = '2px solid #ef4444';
        box.style.boxShadow = '0 0 10px rgba(239, 68, 68, 0.5)';
        box.style.backgroundColor = 'rgba(239, 68, 68, 0.15)';
        box.title = ev.type.replace('_', ' ').toUpperCase();
        overlay.appendChild(box);
    });
}

function toggleInvestigate() {
    const overlay = document.getElementById('xai-overlay');
    const label = document.getElementById('heatmap-label');
    const btn = document.getElementById('investigate-btn');

    if (!overlay || !btn) return;

    const isHidden = overlay.style.display === 'none' || overlay.style.display === '';

    if (isHidden) {
        overlay.style.display = 'block';
        if (label) label.style.display = 'block';
        btn.innerText = '🛡️ Hide Analysis Proof';
        btn.style.background = 'linear-gradient(135deg, #1e293b, #0f172a)';
    } else {
        overlay.style.display = 'none';
        if (label) label.style.display = 'none';
        btn.innerText = '🔍 Investigate Forensic Proof';
        btn.style.background = 'linear-gradient(135deg, #ef4444, #b91c1c)';
    }
}

function resetUpload() {
    console.log("[DeepGuard] Resetting analysis.");
    stopDotAnimation();

    const dropZone = document.getElementById('drop-zone');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');
    const fileInput = document.getElementById('file-input');
    const overlay = document.getElementById('xai-overlay');
    const label = document.getElementById('heatmap-label');
    const btn = document.getElementById('investigate-btn');

    if (dropZone) dropZone.style.display = 'block';
    if (loadingState) loadingState.style.display = 'none';
    if (resultState) resultState.style.display = 'none';
    if (fileInput) fileInput.value = '';

    if (overlay) {
        overlay.style.display = 'none';
        overlay.innerHTML = '';
    }
    if (label) label.style.display = 'none';
    if (btn) {
        btn.innerText = '🔍 Investigate Forensic Proof';
        btn.style.background = 'linear-gradient(135deg, #ef4444, #b91c1c)';
        btn.style.display = 'none';
    }
}
