import cv2
import numpy as np
import base64
import json
import time
import mediapipe as mp
import librosa
import scipy.signal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Crucial for Iris & refined eye points
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Haar Cascade Face Detector (Fast Pre-filter) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- EAR (Eye Aspect Ratio) Logic ---
# MediaPipe Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, width, height, eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR) using Euclidean distances.
    Vertical distance / Horizontal distance.
    """
    # Helper to get coordinates
    def get_point(index):
        point = landmarks[index]
        return np.array([point.x * width, point.y * height])

    # Vertical landmarks (roughly top and bottom of eye)
    p2 = get_point(eye_indices[1])
    p6 = get_point(eye_indices[5])
    p3 = get_point(eye_indices[2])
    p5 = get_point(eye_indices[4])

    # Horizontal landmarks (corners)
    p1 = get_point(eye_indices[0]) # Outer corner?
    p4 = get_point(eye_indices[3]) # Inner corner?

    # Euclidean Distances
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# State Tracking per Connection
client_states = {}

print("[DeepGuard] Advanced Vision Server (MediaPipe) Starting...")

def decode_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def analyze_audio_liveness(audio_data, sample_rate, state):
    """
    Analyzes audio for AI-generated voice detection using spectral analysis.
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: sampling rate (typically 16000 or 44100)
        state: client state dictionary for temporal tracking
    
    Returns:
        risk_score: int (0-100)
        anomalies: list of detected issues
    """
    risk_score = 0
    anomalies = []
    
    try:
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 1. Spectral Centroid Analysis
        # AI voices often have unnatural spectral distribution
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        avg_centroid = np.mean(spectral_centroids)
        centroid_variance = np.var(spectral_centroids)
        
        # Store history for temporal analysis
        if "audio_centroid_history" not in state:
            state["audio_centroid_history"] = []
        
        state["audio_centroid_history"].append(avg_centroid)
        state["audio_centroid_history"] = state["audio_centroid_history"][-30:]  # Keep last 30 samples
        
        # AI voices have unusually consistent spectral centroids
        if len(state["audio_centroid_history"]) > 10:
            temporal_variance = np.var(state["audio_centroid_history"])
            if temporal_variance < 100000:  # Very low variance
                risk_score += 35
                anomalies.append(f"Unnatural Voice Consistency (Spectral Var: {temporal_variance:.0f})")
                print(f"[DeepGuard] ⚠️ AI Voice Detected - Low Spectral Variance: {temporal_variance:.0f}")
        
        # 2. Zero-Crossing Rate (ZCR)
        # Measures how often the signal changes sign
        # AI voices are often too smooth (low ZCR variance)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        zcr_variance = np.var(zcr)
        
        if zcr_variance < 0.0001:  # Unnaturally smooth
            risk_score += 25
            anomalies.append(f"Synthetic Audio Pattern (ZCR Var: {zcr_variance:.6f})")
            print(f"[DeepGuard] ⚠️ Synthetic Audio - Low ZCR Variance: {zcr_variance:.6f}")
        
        # 3. Pitch Variance Analysis
        # Real human voices have natural pitch fluctuations
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 5:
            pitch_variance = np.var(pitch_values)
            
            # AI voices often have too consistent pitch
            if pitch_variance < 50:
                risk_score += 20
                anomalies.append(f"Robotic Pitch Pattern (Var: {pitch_variance:.1f})")
                print(f"[DeepGuard] ⚠️ Robotic Voice - Low Pitch Variance: {pitch_variance:.1f}")
        
        # 4. Spectral Rolloff
        # Point where 85% of spectral energy is below this frequency
        # AI voices often have unnatural high-frequency cutoffs
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        avg_rolloff = np.mean(rolloff)
        
        # Suspiciously sharp cutoffs (common in AI voices)
        if avg_rolloff > 7000 or avg_rolloff < 2000:
            risk_score += 15
            anomalies.append(f"Unnatural Frequency Range (Rolloff: {avg_rolloff:.0f} Hz)")
        
        print(f"[DeepGuard] Audio Analysis - Centroid: {avg_centroid:.0f}, ZCR Var: {zcr_variance:.6f}, Pitch Var: {pitch_variance if len(pitch_values) > 5 else 0:.1f}")
        
    except Exception as e:
        print(f"[DeepGuard] Audio analysis error: {e}")
        return 0, []
    
    return min(risk_score, 100), anomalies

def analyze_liveness(frame, state):
    current_time = time.time()
    risk_score = 0
    anomalies = []
    
    # Get frame dimensions
    h, w, _ = frame.shape
    
    # --- STEP 1: Fast Face Detection (Haar Cascade Pre-filter) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # No face detected by Haar Cascade -> Skip expensive MediaPipe processing
    if len(faces) == 0:
        print("[DeepGuard] No face detected by Haar Cascade - skipping analysis")
        debug_info = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "variance": 0.0
        }
        return 0, ["No face detected"], debug_info
    
    print(f"[DeepGuard] Face detected: {len(faces)} face(s) found")
    
    # --- STEP 2: Pixel Variance Liveness Check ---
    # Extract the largest face region
    (x, y, fw, fh) = max(faces, key=lambda f: f[2] * f[3])  # Largest face by area
    face_region = gray[y:y+fh, x:x+fw]
    
    # Calculate pixel variance in the face region
    pixel_variance = np.var(face_region)
    
    # Store variance history for temporal analysis
    if "pixel_variance_history" not in state:
        state["pixel_variance_history"] = []
    
    state["pixel_variance_history"].append((current_time, pixel_variance))
    state["pixel_variance_history"] = [p for p in state["pixel_variance_history"] if current_time - p[0] < 3.0]
    
    # Check if variance is too low (static image indicator)
    if len(state["pixel_variance_history"]) > 10:
        avg_pixel_var = np.mean([p[1] for p in state["pixel_variance_history"]])
        
        print(f"[DeepGuard] Pixel Variance: {avg_pixel_var:.2f} (Current: {pixel_variance:.2f})")
        
        # Very low variance = likely a static image or low-quality deepfake
        if avg_pixel_var < 5.0:
            risk_score += 40
            anomalies.append(f"Static Image Detected (Pixel Var: {avg_pixel_var:.1f})")
            print(f"[DeepGuard] ⚠️ STATIC IMAGE DETECTED! Avg Variance: {avg_pixel_var:.2f}")
        elif avg_pixel_var < 15.0:
            risk_score += 15
            anomalies.append("Low Pixel Variance (Suspicious)")
            print(f"[DeepGuard] ⚠️ Low Pixel Variance (Suspicious): {avg_pixel_var:.2f}")
    
    # --- STEP 3: MediaPipe Detailed Analysis ---
    # Preprocessing for landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        state["face_lost_time"] = current_time
        return 0, [] # No face found

    # Get landmarks
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Calculate EAR for both eyes
    left_ear = calculate_ear(landmarks, w, h, LEFT_EYE)
    right_ear = calculate_ear(landmarks, w, h, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Blink Threshold (Typical Human EAR threshold is ~0.2 - 0.25)
    # Increased to 0.22 to be more sensitive to partial blinks
    BLINK_THRESHOLD = 0.22
    
    # State Machine for Blinking
    if avg_ear < BLINK_THRESHOLD:
        if not state.get("eyes_closed", False):
            state["eyes_closed"] = True
            state["last_blink"] = current_time
            state["blink_count"] += 1
            print(f"[DeepGuard] Blink Detected! (EAR: {avg_ear:.2f})")
    else:
        state["eyes_closed"] = False

    # Liveness Checks
    time_since_last_blink = current_time - state.get("last_blink", current_time)
    
    # Rule 1: Abnormal Stare (Deepfake symptom)
    # Humans can focus for 15-20 seconds, so increased threshold to 20s
    if time_since_last_blink > 20.0:
        risk_score += 70
        anomalies.append(f"Abnormal Stare (No blink for {int(time_since_last_blink)}s)")
    elif time_since_last_blink > 15.0:
        risk_score += 25
        anomalies.append("Low Blink Rate")

    # Rule 2: Head Rotation Analysis (Statue Check)
    pitch, yaw, roll = get_head_pose(landmarks, w, h)
    
    # Store pose history for "Statue Check" (Time-based variance analysis)
    state["pose_history"].append((current_time, pitch, yaw, roll))
    
    # Keep only last 5 seconds of data
    state["pose_history"] = [p for p in state["pose_history"] if current_time - p[0] < 5.0]
    
    if len(state["pose_history"]) > 50:  # Need ~5 seconds of data for reliable variance
        # Calculate standard deviation (variance) of movement
        poses = np.array([p[1:] for p in state["pose_history"]])  # [[p, y, r], ...]
        std_dev = np.std(poses, axis=0)  # [std_pitch, std_yaw, std_roll]
        
        # Thresholds: Humans naturally move slightly (micro-movements)
        # Even when focused, humans have subtle movements from breathing, balance
        # Only truly static images/deepfakes have variance < 0.15°
        
        avg_movement = np.mean(std_dev[:2])  # Check Pitch & Yaw mainly
        
        if avg_movement < 0.15:  # Extremely still - likely deepfake/image
            risk_score += 50
            anomalies.append(f"Unnatural Stillness (Statue-like: {avg_movement:.2f}° var)")
        elif avg_movement < 0.3:  # Very still - suspicious but could be focused human
            risk_score += 15
            anomalies.append("Low Head Movement")

    # Prepare debug info
    debug_info = {
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
        "variance": float(avg_movement) if len(state["pose_history"]) > 30 else 0.0
    }

    return min(risk_score, 100), anomalies, debug_info

def get_head_pose(landmarks, img_w, img_h):
    """
    Estimates Head Pose (Pitch, Yaw, Roll) using solvePnP
    Returns: pitch, yaw, roll in degrees
    """
    # 2D Image Points from MediaPipe
    face_2d = []
    face_3d = []  # We'll construct a generic 3D face model

    # Key Landmarks for Pose Estimation
    # Nose tip (1), Chin (152), Left Eye Left Corner (33), Right Eye Right Corner (263), Left Mouth Corner (61), Right Mouth Corner (291)
    key_landmarks = [1, 152, 33, 263, 61, 291]

    for idx in key_landmarks:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        
        # Generic 3D model points (approximate relative positions)
        if idx == 1: face_3d.append([0, -1.12, 4.25])      # Nose tip
        elif idx == 152: face_3d.append([0, -8.70, -0.65]) # Chin
        elif idx == 33: face_3d.append([-3.5, 1.80, 0.20]) # Left Eye Corner
        elif idx == 263: face_3d.append([3.5, 1.80, 0.20]) # Right Eye Corner
        elif idx == 61: face_3d.append([-2.0, -4.50, 0.10])# Left Mouth Corner
        elif idx == 291: face_3d.append([2.0, -4.50, 0.10])# Right Mouth Corner

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera Matrix (Approximate)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])

    # Distortion Matrix (Assuming no distortion)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get Rotational Matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Calculate Euler Angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Convert to degrees
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360  
    
    # Adjust for intuitive values
    # Pitch: Down +, Up -
    # Yaw: Right +, Left -
    # Roll: Tilt Right +, Tilt Left -

    return pitch, yaw, roll

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    client_states[client_id] = {
        "last_blink": time.time(),
        "blink_count": 0,
        "eyes_closed": False,
        "pose_history": []  # Stores (timestamp, pitch, yaw, roll)
    }
    
    print(f"[DeepGuard] Client #{client_id} connected.")
    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            video_risk = 0
            audio_risk = 0
            all_anomalies = []
            debug_info = {}
            
            # Process video frame
            if "payload" in payload:
                image = decode_image(payload["payload"])
                
                if image is not None:
                    video_risk, video_anomalies, debug_info = analyze_liveness(image, client_states[client_id])
                    all_anomalies.extend(video_anomalies)
            
            # Process audio chunk (if present)
            if "audio_payload" in payload:
                try:
                    # Decode base64 audio data
                    audio_b64 = payload["audio_payload"]
                    if "," in audio_b64:
                        audio_b64 = audio_b64.split(",")[1]
                    
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Convert to numpy array (assuming 16-bit PCM, mono, 16kHz)
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    sample_rate = payload.get("sample_rate", 16000)
                    
                    # Analyze audio
                    audio_risk, audio_anomalies = analyze_audio_liveness(audio_data, sample_rate, client_states[client_id])
                    all_anomalies.extend(audio_anomalies)
                    
                except Exception as e:
                    print(f"[DeepGuard] Audio processing error: {e}")
            
            # Combined risk score (weighted average: 60% video, 40% audio)
            combined_risk = int(video_risk * 0.6 + audio_risk * 0.4)
            
            response = {
                "status": "processed",
                "risk_score": combined_risk,
                "video_risk": int(video_risk),
                "audio_risk": int(audio_risk),
                "anomalies": all_anomalies,
                "debug_info": debug_info,
                "timestamp": time.time()
            }
            
            await websocket.send_json(response)
                
    except WebSocketDisconnect:
        print(f"[DeepGuard] Client #{client_id} disconnected")
        # del client_states[client_id]
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
