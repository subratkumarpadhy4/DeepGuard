# DeepGuard-AI: Real-Time Deepfake & Scam Detection Architecture

## 1. System Architecture

The DeepGuard-AI system follows a real-time stream processing pipeline designed to intercept media, analyze it for synthetic artifacts, and alert the user immediately.

### Architecture Flow Diagram

```mermaid
graph TD
    User[User / Victim] -->|Video Call / Browser| Stream[Media Stream Interceptor]
    
    subgraph Client_Side [Client Side - Extension/Plugin]
        Stream -->|Capture| WebRTC_Handler[WebRTC Media Handler]
        WebRTC_Handler -->|Extract Audio/Video| Preprocessor[Pre-processor (Frame extraction, Audio normalization)]
        Preprocessor -->|Secure Websocket| Backend_API[Analysis Server API]
        Backend_API <-->|Alert Data| Overlay[Overlay UI Manager]
        Overlay -->|Visual/Audio Warning| User
    end

    subgraph Server_Side [DeepGuard Intelligence Engine]
        Backend_API -->|Video Frames| Vision_Model[Vision Core (CNN + Landmark)]
        Backend_API -->|Audio Chunks| Audio_Model[Audio Core (Spectrogram + MFCC)]
        Backend_API -->|Live Transcript| NLP_Module[NLP Phishing Scanner]
        
        Vision_Model -->|Blink/Desync Score| Risk_Engine[Risk Aggregation Engine]
        Audio_Model -->|Wavenet/Spectral Score| Risk_Engine
        NLP_Module -->|Social Eng. Score| Risk_Engine
        
        Risk_Engine -->|Final Risk JSON| Backend_API
    end
```

### Flow Description
1.  **Media Input**: A Browser Extension or Zoom Plugin intercepts the WebRTC media stream (video/audio) or DOM elements.
2.  **Pre-processing**: The client-side logic captures frames (e.g., 5fps) and audio buffers (e.g., 2sec chunks) to reduce bandwidth.
3.  **ML Inference**:
    *   **Vision Core**: Analyzes blinking patterns, mouth-to-audio sync, and facial artifacts (e.g., strange shadows).
    *   **Audio Core**: Analyzes spectrograms to detect TTS (Text-to-Speech) frequencies and lack of natural breath pauses.
    *   **NLP Scanner**: Converts speech to text (STT) and scans for semantic red flags (e.g., "gift cars", "urgent", "SSN request").
4.  **Frontend Alerting**: If the Risk Aggregation Engine calculates a score > Threshold, the UI Overlay triggers a "High Risk" warning with explainable reasons.

---

## 2. Tech Stack Recommendation

### Backend (Intelligence Engine)
*   **Language**: **Python 3.10+** (Industry standard for AI/ML libraries).
*   **ML Framework**: **TensorFlow / Keras** or **PyTorch** (for running Deepfake detection models like XceptionNet or MesoNet).
*   **Computer Vision**: **OpenCV** & **dlib** (Face extraction, landmark detection).
*   **Audio Processing**: **Librosa** (Spectral analysis).
*   **API Framework**: **FastAPI** (High-performance async support, crucial for real-time inference).

### Frontend / Communication Layer
*   **Runtime**: **Node.js** (for handling high-concurrency websocket connections if using a relay server).
*   **Client Core**: **JavaScript / TypeScript**.
*   **Communication**: **WebRTC** (for media handling) & **Socket.io / WebSocket** (for low-latency risk score transmission).
*   **Browser API**: **Chrome Extension Manifest V3** (offscreen documents for audio processing).

### Deployment & Ops
*   **Containerization**: Docker (for consistent ML environments).
*   **Acceleration**: CUDA/GPU support for real-time inference.

---

## 3. Explainable AI (XAI) UI Concept

The UI is designed to be unobtrusive until a threat is detected.

*   **Status Indicator**: A small floating shield icon in the corner of the video call.
    *   🟢 **Green**: Secure / Real Human Verified.
    *   🟡 **Yellow**: analyzing... or Uncertainty.
    *   🔴 **Red**: Deepfake / Scam Detected.

*   **Alert Overlay**: When 'Red' status is triggered:
    *   **Headline**: "POTENTIAL DEEPFAKE DETECTED" (Blinking Red).
    *   **Risk Score**: "94% Risk Level".
    *   **Evidence List (The 'Why')**:
        *   "⚠️ **Lip Sync Error**: Mouth movements do not match audio track."
        *   "⚠️ **Unnatural Blinking**: Eye blink rate is statistically impossible (0.1 blinks/min)."
        *   "⚠️ **Scam Keyword**: Caller requested 'Gift Card' payment."
    *   **Action Buttons**: "Block Call", "Ignore", "Report".
