// DeepGuard-AI Stream Interceptor
// This script is designed to be injected by the Browser Extension (manifest.json)
// It hooks into the browser's Media API to transparently capture video/audio 
// without interrupting the user's call (Google Meet, Zoom Web, Teams).

console.log("[DeepGuard] Interceptor Loaded. Connecting to Analysis Engine...");

let socket = null;
const BACKEND_URL = "ws://localhost:8000/ws/analyze";

function connectToBackend() {
    socket = new WebSocket(BACKEND_URL);

    socket.onopen = () => {
        console.log("[DeepGuard] WebSocket Connected! Secure Pipeline Established.");
    };

    socket.onmessage = (event) => {
        const report = JSON.parse(event.data);
        const display = document.getElementById('live-risk-display');
        const details = document.getElementById('risk-details');

        if (report.risk_score > 50) {
            console.warn("[DeepGuard ALERT] High Risk Detected:", report);
            if (display) {
                display.innerText = `⚠️ HIGH RISK: ${report.risk_score}%`;
                display.style.color = "red";
                details.innerText = "Possible Deepfake Detected: " + report.anomalies.join(", ");
                if (report.anomalies.some(a => a.includes("Statue") || a.includes("Stillness"))) {
                    details.innerText = "⚠️ UNNATURAL STILLNESS DETECTED. " + details.innerText;
                }
            }
            alertUserOfRisk(report);
        } else {
            console.log("[DeepGuard] Frame Safe. Risk Score:", report.risk_score);
            if (display) {
                display.innerText = `✅ SECURE: ${report.risk_score}% Risk`;
                display.style.color = "green";
                details.innerText = "Biological signals verified. No anomalies.";
            }
            // Auto-hide the warn overlay if it was shown
            hideOverlay();
        }
    };

    socket.onclose = () => {
        console.log("[DeepGuard] WebSocket Disconnected. Reconnecting in 3s...");
        const display = document.getElementById('live-risk-display');
        if (display) display.innerText = "🔌 Connecting to Brain...";
        setTimeout(connectToBackend, 3000);
    };

    socket.onerror = (error) => {
        console.error("[DeepGuard] WebSocket Error:", error);
    };
}

// Initialize Connection
// Initialize Connection
connectToBackend();

// --- UI Overlay Logic ---

let overlayContainer = null;
let overlayTimeout = null;

createOverlayUI();

function createOverlayUI() {
    if (document.getElementById('deepguard-overlay')) return;

    // Create container
    overlayContainer = document.createElement('div');
    overlayContainer.id = 'deepguard-overlay';
    overlayContainer.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        width: 380px;
        background: rgba(20, 20, 20, 0.95);
        color: white;
        padding: 24px;
        border-radius: 16px;
        z-index: 999999;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        display: none; /* Hidden by default */
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 59, 48, 0.3);
        transform: translateX(120%);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    `;

    // Inner HTML content
    overlayContainer.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 12px; height: 12px; background: #ff453a; border-radius: 50%; box-shadow: 0 0 10px #ff453a; animation: pulse 1s infinite alternate;"></div>
                <h3 style="margin: 0; font-size: 18px; font-weight: 700; color: #ff453a; letter-spacing: 0.5px;">THREAT DETECTED</h3>
            </div>
            <span id="dg-risk-score" style="font-size: 24px; font-weight: 800; color: white;">94%</span>
        </div>
        
        <div style="background: rgba(255, 69, 58, 0.15); border-left: 4px solid #ff453a; padding: 12px; border-radius: 4px; margin-bottom: 16px;">
            <p id="dg-alert-msg" style="margin: 0; font-size: 14px; line-height: 1.4; color: #ffdede;">
                AI has detected unnatural facial artifacts consistent with deepfake manipulation.
            </p>
        </div>

        <div style="display: flex; gap: 8px;">
            <button onclick="document.getElementById('deepguard-overlay').style.transform='translateX(120%)'" 
                style="flex: 1; padding: 10px; border: none; background: rgba(255,255,255,0.1); color: white; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 13px; transition: background 0.2s;">
                Ignore
            </button>
            <button style="flex: 1; padding: 10px; border: none; background: #ff453a; color: white; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 13px; box-shadow: 0 4px 15px rgba(255, 69, 58, 0.4);">
                Block Stream
            </button>
        </div>
        <style>
            @keyframes pulse { from { opacity: 0.6; transform: scale(1); } to { opacity: 1; transform: scale(1.2); } }
        </style>
    `;

    document.body.appendChild(overlayContainer);
    console.log("[DeepGuard] UI Overlay injected into DOM.");
}

function showOverlay(report) {
    if (!overlayContainer) createOverlayUI();

    // Update Content
    const scoreEl = document.getElementById('dg-risk-score');
    const msgEl = document.getElementById('dg-alert-msg');

    if (scoreEl) scoreEl.innerText = report.risk_score + "%";
    if (msgEl) {
        let message = "Risk Factors: " + report.anomalies.join(", ");
        if (report.anomalies.some(a => a.includes("Statue") || a.includes("Stillness"))) {
            message = "⚠️ UNNATURAL LACK OF MOVEMENT DETECTED (Statue-like). " + message;
        }
        msgEl.innerText = message;
    }

    // Show Animation
    overlayContainer.style.display = "block";
    // Small delay to allow 'display: block' to apply before transform for animation
    requestAnimationFrame(() => {
        overlayContainer.style.transform = "translateX(0)";
    });

    // Auto-hide if safe again (debounced in real app, simple timeout here)
    if (overlayTimeout) clearTimeout(overlayTimeout);
    overlayTimeout = setTimeout(() => {
        // Optional: Auto dismiss after 10s if no new threats? 
        // For now, we keep it until 'hideOverlay' is called by a Safe frame logic
    }, 5000);
}

function hideOverlay() {
    if (overlayContainer) {
        overlayContainer.style.transform = "translateX(120%)";
    }
}

function alertUserOfRisk(report) {
    // Trigger the beautiful UI
    showOverlay(report);

    // Also keep the console log for debugging
    console.error("%c DEEPFAKE DETECTED! ", "background: red; color: white; font-size: 20px");
}


// 1. Store the original browser function
const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

// 2. Overwrite the function with our proxy
navigator.mediaDevices.getUserMedia = async (constraints) => {
    console.log("[DeepGuard] Call detected! Intercepting media stream...");

    // Get the actual camera/mic stream requested by the calling app (e.g., Google Meet)
    const mediaStream = await originalGetUserMedia(constraints);

    // 3. "Tee" the stream: Split it into two paths
    // Path A: Goes back to the calling app (so the user can still talk)
    // Path B: Goes to our Analysis Engine
    processStreamForAnalysis(mediaStream);

    // Return Path A to the app seamlessly
    return mediaStream;
};

function processStreamForAnalysis(stream) {
    const videoTrack = stream.getVideoTracks()[0];
    const audioTrack = stream.getAudioTracks()[0];

    if (videoTrack) {
        console.log("[DeepGuard] Capturing Video Track:", videoTrack.label);

        // Robust Frame Capture: Use a hidden video element
        const hiddenVideo = document.createElement('video');
        hiddenVideo.srcObject = stream;
        hiddenVideo.muted = true;
        hiddenVideo.play().catch(e => console.error("Hidden video play error:", e));

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // Processing Loop
        setInterval(() => {
            if (videoTrack.readyState === 'live' && socket && socket.readyState === WebSocket.OPEN) {
                try {
                    // Set canvas size to video size (or downscale for performance)
                    if (hiddenVideo.videoWidth > 0) {
                        canvas.width = 320; // Lower res for faster processing
                        canvas.height = 240;

                        // Draw current video frame to canvas
                        ctx.drawImage(hiddenVideo, 0, 0, canvas.width, canvas.height);

                        // Convert to Base64
                        const base64Frame = canvas.toDataURL('image/jpeg', 0.6); // 60% quality

                        // Send!
                        sendFrameToBackend(base64Frame);
                    }
                } catch (err) {
                    console.error("[DeepGuard] Frame capture error:", err);
                }
            }
        }, 1000); // Check every 1 second
    }

    if (audioTrack) {
        console.log("[DeepGuard] Capturing Audio Track:", audioTrack.label);
    }
}

function sendFrameToBackend(frameData) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        // Send simply textual data for the mock server to process
        socket.send(JSON.stringify({
            type: "video_frame",
            payload: frameData
        }));
    }
}
