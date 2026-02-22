import cv2
import numpy as np
import base64
import json
import time
import mediapipe as mp
import librosa
import scipy.signal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile

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
            # Softened from 250,000 to 80,000. 
            # Real humans can be consistent, but AI is extremely flat.
            if temporal_variance < 80000: 
                risk_score += 40 
                anomalies.append(f"Unnatural Voice Timbre (Spectral Var: {temporal_variance:.0f})")
        
        # 2. Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        zcr_variance = np.var(zcr)
        
        # Softened from 0.002 to 0.0005. 
        if zcr_variance < 0.0005: 
            risk_score += 25
            anomalies.append(f"Synthetic Audio Pattern (ZCR Var: {zcr_variance:.6f})")
        
        # 3. Pitch Variance Analysis
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 5:
            pitch_variance = np.var(pitch_values)
            # Softened from 150 to 80. Real humans can be monotonic.
            if pitch_variance < 80:
                risk_score += 30
                anomalies.append(f"Robotic Pitch Pattern (Var: {pitch_variance:.1f})")
        
        # 4. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        avg_rolloff = np.mean(rolloff)
        
        # Suspiciously sharp cutoffs (usually indicative of low-bitrate AI or specific vocoders)
        if avg_rolloff > 8500 or avg_rolloff < 1500:
            risk_score += 20
            anomalies.append(f"Unnatural Frequency Range (Rolloff: {avg_rolloff:.0f} Hz)")
        
        print(f"[DeepGuard] Audio Analysis - Centroid: {avg_centroid:.0f}, ZCR Var: {zcr_variance:.6f}, Pitch Var: {pitch_variance if len(pitch_values) > 5 else 0:.1f}")
        
    except Exception as e:
        print(f"[DeepGuard] Audio analysis error: {e}")
        return 0, []
    
    return min(risk_score, 100), anomalies

def analyze_liveness(frame, state, override_timestamp=None):
    current_time = override_timestamp if override_timestamp is not None else time.time()
    risk_score = 0
    anomalies = []
    
    # Get frame dimensions
    h, w, _ = frame.shape
    
    # --- STEP 1: MediaPipe Detailed Analysis (Primary) ---
    # We removed the weak Haar Cascade pre-filter which was missing faces.
    
    # Preprocessing for landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        # Fallback/Return if no face found by MediaPipe
        state["face_lost_time"] = current_time
        return 0, ["No face detected"], {}

    # Get landmarks
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Get frame dimensions for later use
    h, w, _ = frame.shape
    
    print(f"[DeepGuard] Face detected by MediaPipe.")
    
    # --- STEP 2: Pixel Variance Liveness Check (Optional but kept for static images) ---
    # We can still do pixel variance on a cropped region if needed, 
    # but let's derive the bounding box from MediaPipe landmarks now for accuracy.
    
    # Estimate bounding box from landmarks
    x_min = min([lm.x for lm in landmarks]) * w
    x_max = max([lm.x for lm in landmarks]) * w
    y_min = min([lm.y for lm in landmarks]) * h
    y_max = max([lm.y for lm in landmarks]) * h
    
    # Clamp to frame
    x_min, x_max = max(0, int(x_min)), min(w, int(x_max))
    y_min, y_max = max(0, int(y_min)), min(h, int(y_max))
    
    try:
        if x_max > x_min and y_max > y_min:
            face_region = cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
            pixel_variance = np.var(face_region)
            
             # Store variance history for temporal analysis
            if "pixel_variance_history" not in state:
                state["pixel_variance_history"] = []
            
            state["pixel_variance_history"].append((current_time, pixel_variance))
            state["pixel_variance_history"] = [p for p in state["pixel_variance_history"] if current_time - p[0] < 3.0]
            
            # Check if variance is too low (static image indicator)
            if len(state["pixel_variance_history"]) > 10:
                avg_pixel_var = np.mean([p[1] for p in state["pixel_variance_history"]])
                
                # Very low variance = likely a static image or low-quality deepfake
                if avg_pixel_var < 5.0:
                    risk_score += 40
                    anomalies.append(f"Static Image Detected (Pixel Var: {avg_pixel_var:.1f})")
                elif avg_pixel_var < 15.0:
                    risk_score += 15
                    anomalies.append("Low Pixel Variance (Suspicious)")
                    
    except Exception as e:
        print(f"[DeepGuard] Variance check failed: {e}")
        
    # --- STEP 3: Continue with EAR and Pose ---
    
    # Calculate EAR for both eyes
    left_ear = calculate_ear(landmarks, w, h, LEFT_EYE)
    right_ear = calculate_ear(landmarks, w, h, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # --- CHECK 4: Mouth Aspect Ratio (MAR) & Lip Sync Proxy ---
    # Inner lip landmarks for better region isolation
    p_upper = landmarks[13]
    p_lower = landmarks[14]
    p_left = landmarks[78]
    p_right = landmarks[308]
    
    vertical_mouth = np.linalg.norm(np.array([p_upper.x*w, p_upper.y*h]) - np.array([p_lower.x*w, p_lower.y*h]))
    horizontal_mouth = np.linalg.norm(np.array([p_left.x*w, p_left.y*h]) - np.array([p_right.x*w, p_right.y*h]))
    
    mar = vertical_mouth / horizontal_mouth if horizontal_mouth > 0 else 0
    
    # Track MAR history
    if "mar_history" not in state: state["mar_history"] = []
    state["mar_history"].append(mar)
    state["mar_history"] = state["mar_history"][-300:]
    
    # NEW: Mouth Content Analysis (Teeth Check)
    # If mouth is open (MAR > 0.2), check the variance in the mouth region
    if mar > 0.3:
        try:
            # Crop mouth region
            mx1, my1 = int(p_left.x * w), int(p_upper.y * h)
            mx2, my2 = int(p_right.x * w), int(p_lower.y * h)
            
            # Pad slightly
            pad = 5
            mouth_roi = gray_frame[max(0, my1-pad):min(h, my2+pad), max(0, mx1-pad):min(w, mx2+pad)]
            
            if mouth_roi.size > 0:
                # Calculate variance. Real teeth + shadows have high variance.
                # AI "blob" teeth have very low internal variance in the white area.
                m_var = np.var(mouth_roi)
                
                # Check for "Flat Teeth" signature
                if m_var < 800: # Heuristic: Real open mouths usually > 1500 due to teeth/tongue/shadows
                    risk_score += 35
                    anomalies.append(f"Unnatural Mouth Rendering (Mouth Var: {m_var:.1f})")
        except Exception:
            pass

    # Check for "Frozen Mouth" (Robot)
    if len(state["mar_history"]) > 30:
        mar_variance = np.var(state["mar_history"])
        # If talking is expected (e.g. audio present) but mouth is frozen
        # This is handled in the aggregation, but we track here.
        pass 

    # --- CHECK 5: Gaze & Pupil Tracking ("Soulless Eyes") ---
    # Iris landmarks: Left [468-472], Right [473-477]
    # We track the center of the iris relative to the eye corners to detect movement.
    try:
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        
        # Simple Gaze Variance Check
        if "iris_history" not in state: state["iris_history"] = []
        state["iris_history"].append((left_iris.x, left_iris.y, right_iris.x, right_iris.y))
        state["iris_history"] = state["iris_history"][-60:] # 2 seconds
        
        if len(state["iris_history"]) > 10:
            # Calculate movement variance of irises
            iris_data = np.array(state["iris_history"])
            iris_var = np.var(iris_data, axis=0) # [lx_var, ly_var, rx_var, ry_var]
            avg_iris_var = np.mean(iris_var) * 100000 # Scale up
            
            # "Dead Eyes" check: Real eyes constantly saccade (micro-movements)
            # Increased threshold to 1.0 because 0.2 was too strict for some AI models
            if avg_iris_var < 1.0: 
                risk_score += 25
                anomalies.append(f"Soulless Eyes (No saccadic movement: {avg_iris_var:.2f})")
    except IndexError:
        pass # Old MediaPipe model might not have iris landmarks
        
    # --- CHECK 6: Skin Texture Analysis (Plastic/Blurry Skin & GAN Noise) ---
    # Crop a cheek region and check for high-frequency noise (pores, skin texture)
    # and GAN-specific periodic patterns.
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cheek_lm = landmarks[205] # Left cheek
        cx, cy = int(cheek_lm.x * w), int(cheek_lm.y * h)
        patch_size = int(w * 0.08) # 8% of width
        
        y1, y2 = max(0, cy-patch_size), min(h, cy+patch_size)
        x1, x2 = max(0, cx-patch_size), min(w, cx+patch_size)
        
        if y2>y1 and x2>x1:
            cheek_roi = gray_frame[y1:y2, x1:x2]
            
            # 1. Laplacian variance (Blurriness)
            texture_score = cv2.Laplacian(cheek_roi, cv2.CV_64F).var()
            
            # 2. FFT Analysis (Frequency Domain Artifacts)
            f = np.fft.fft2(cheek_roi)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            
            # GANs often have "regular" repeating patterns (ringing/checkered)
            # We check for high-frequency energy distribution
            h_fft, w_fft = magnitude_spectrum.shape
            center_y, center_x = h_fft // 2, w_fft // 2
            
            # Mask out the low frequencies (center)
            mask_size = 5
            magnitude_spectrum[center_y-mask_size:center_y+mask_size, center_x-mask_size:center_x+mask_size] = 0
            
            high_freq_val = np.mean(magnitude_spectrum)
            
            if "texture_history" not in state: state["texture_history"] = []
            if "fft_history" not in state: state["fft_history"] = []
            
            state["texture_history"].append(texture_score)
            state["fft_history"].append(high_freq_val)
            
            state["texture_history"] = state["texture_history"][-30:]
            state["fft_history"] = state["fft_history"][-30:]
            
            avg_texture = np.mean(state["texture_history"])
            avg_fft = np.mean(state["fft_history"])
            
            # DETECTOR LOGIC:
            # - Real humans have natural "noise" (pores/imperfections) -> High high-freq energy
            # - AI/Deepfakes are often too smooth (Low Laplacian) OR have weird periodic noise (Abnormal FFT)
            
            # Thresholds:
            # EXTREME: Only trigger if skin is UNSETTLINGLY smooth (like 25)
            # This prevents false positives from standard beauty filters or high-end webcams.
            if avg_texture < 30.0: 
                risk_score += 10 # Reduced from 15
                anomalies.append(f"Unnatural Skin Smoothness (Blur Score: {avg_texture:.1f})")
            
            # FFT anomaly: Toughened further
            if avg_fft < 1.5: 
                risk_score += 5
                anomalies.append("Lack of Natural Skin Micro-texture (FFT)")
            elif avg_fft > 75.0: 
                risk_score += 25
                anomalies.append("High-Frequency Periodic Artifacts (Deepfake Noise)")
                
    except Exception as e:
        print(f"[DeepGuard] Skin analysis failed: {e}")
    
    # Blink Threshold
    BLINK_THRESHOLD = 0.24
    
    # State Machine for Blinking
    if avg_ear < BLINK_THRESHOLD:
        if not state.get("eyes_closed", False):
            state["eyes_closed"] = True
            state["last_blink"] = current_time
            state["blink_count"] += 1
            print(f"[DeepGuard] Blink Detected! (EAR: {avg_ear:.2f})")
    else:
        state["eyes_closed"] = False

    # Rule 2: Head Rotation Analysis (Pose Estimation)
    pitch, yaw, roll = get_head_pose(landmarks, w, h)
    
    # Store pose history for "Statue Check" (Time-based variance analysis)
    state["pose_history"].append((current_time, pitch, yaw, roll))
    
    # Keep only last 5 seconds of data (approx 150 frames at 30fps)
    state["pose_history"] = [p for p in state["pose_history"] if current_time - p[0] < 5.0]
    
    avg_movement = 0
    if len(state["pose_history"]) > 15: 
        # Calculate standard deviation (variance) of movement
        poses = np.array([p[1:] for p in state["pose_history"]])  # [[p, y, r], ...]
        std_dev = np.std(poses, axis=0)  # [std_pitch, std_yaw, std_roll]
        avg_movement = np.mean(std_dev[:2])  # Check Pitch & Yaw mainly
        
        # --- CHECK 7: Jerky Motion / Glitch Detection ---
        velocities = np.diff(poses, axis=0) # [d_pitch, d_yaw, d_roll]
        if len(velocities) > 0:
             max_jerk = np.max(np.abs(velocities))
             if max_jerk > 12.0: 
                 risk_score += 40
                 anomalies.append(f"Robotic/Jerky Motion (Max Jerk: {max_jerk:.1f}°)")

    # --- AGGREGATION & HUMANITY CREDIT ---
    humanity_credit = 0
    
    # Credit for natural eye micro-movements (Even subtle ones)
    try:
        # Relaxed from 1.0 to 0.7 to catch subtle saccades
        if 0.7 < avg_iris_var < 8.0: 
            humanity_credit += 25
    except NameError:
        avg_iris_var = 0
    
    # Credit for natural head sway
    # Relaxed from 0.8 to 0.5 to catch very stable but living humans
    if 0.5 < avg_movement < 5.0:
        humanity_credit += 20

    # Re-calculate blink risk (EXTREMELY forgiving)
    time_since_last_blink = current_time - state.get("last_blink", current_time)
    blink_risk = 0
    if time_since_last_blink > 25.0: # Deepfakes often NEVER blink
        blink_risk = 30
        anomalies.append(f"Abnormal Stare (No blink for {int(time_since_last_blink)}s)")
    
    # Re-calculate motion risk
    motion_risk = 0
    if len(state["pose_history"]) > 90: # Need 3 seconds of stillness
        if avg_movement < 0.2:
            motion_risk = 20
            anomalies.append(f"Unnatural Stillness (Statue-like: {avg_movement:.2f}° var)")

    # Combine risks
    final_risk = risk_score + blink_risk + motion_risk
    
    # Apply humanity credit
    final_risk = max(0, final_risk - humanity_credit)
    
    # Cap "Absence of behavior" risk if no "Positive AI" artifacts are found
    has_positive_ai_artifact = any(a in ["Unnatural Skin Smoothness", "High-Frequency Periodic Artifacts", "Unnatural Mouth Rendering", "Robotic/Jerky Motion"] for a in anomalies)
    
    if not has_positive_ai_artifact:
        # If it's just "stillness" and "low blinking", cap it at 35% (Suspicious but not Critical)
        final_risk = min(final_risk, 35)
    else:
        # If we have POSITIVE AI indicators, ensure the risk doesn't drop too low from credit
        final_risk = max(final_risk, 40)

    # Prepare debug info
    debug_info = {
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
        "variance": float(avg_movement),
        "humanity_credit": humanity_credit
    }

    return min(final_risk, 100), anomalies, debug_info

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

@app.post("/analyze/video")
async def analyze_video_upload(file: UploadFile = File(...)):
    """
    Endpoint to upload a video file and analyze it for deepfake signatures.
    """
    print(f"[DeepGuard] Received video upload: {file.filename}")
    
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        shutil.copyfileobj(file.file, temp_video)
        temp_video_path = temp_video.name
    
    try:
        # --- VIDEO ANALYSIS ---
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        FRAME_SKIP = 5 
        
        analysis_state = {
            "last_blink": time.time(),
            "blink_count": 0,
            "eyes_closed": False,
            "pose_history": [],
            "pixel_variance_history": [],
            "mar_history": [],
            "iris_history": [],
            "texture_history": [],
            "fft_history": []
        }
        
        video_risk_accum = 0
        humanity_credit_accum = 0
        anomaly_counts = {}
        frame_count = 0
        max_frame_risk = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0: continue
            
            current_video_time = frame_count / fps
            risk, anomalies, debug = analyze_liveness(frame, analysis_state, override_timestamp=current_video_time)
            
            # Use raw risk logic for the aggregation
            video_risk_accum += risk
            humanity_credit_accum += debug.get("humanity_credit", 0)
            
            for a in anomalies:
                anomaly_counts[a] = anomaly_counts.get(a, 0) + 1

        cap.release()
        
        frames_processed = frame_count // FRAME_SKIP
        if frames_processed == 0: return {"status": "error", "message": "No frames processed"}
        
        avg_video_risk = video_risk_accum / frames_processed
        avg_humanity_credit = humanity_credit_accum / frames_processed
        
        # --- POLARIZED AGGREGATION ---
        robust_anomalies = []
        critical_indicators = 0
        
        # SUSTAINED ANOMALIES (> 50% of video duration)
        # Deepfakes carry artifacts throughout. Humans only have glitches periodically.
        for anomaly, count in anomaly_counts.items():
            frequency = count / frames_processed
            if frequency > 0.50:
                robust_anomalies.append(anomaly)
                if any(x in anomaly for x in ["Periodic Artifacts", "Mouth Rendering", "Skin Micro-texture", "Jerky Motion", "Soulless Eyes"]):
                    critical_indicators += 1
                if any(x in anomaly for x in ["Abnormal Stare", "Statue-like"]) and frequency > 0.85:
                    critical_indicators += 1

        # SCORE SCALING: Force the gap
        # Force risk down for humans log-linearly
        if avg_humanity_credit > 5:
            # Exponential decay of risk for humans
            avg_video_risk = avg_video_risk * (0.8 ** (avg_humanity_credit / 2))
            
        if critical_indicators == 0:
            avg_video_risk = min(avg_video_risk, 10) 
        
        # Boost only for multiple pieces of evidence
        if critical_indicators >= 3: 
            avg_video_risk = max(avg_video_risk, 96)
        elif critical_indicators == 2:
            avg_video_risk = max(avg_video_risk, 85)
        elif critical_indicators == 1:
            # ONE STRIKE: Can only make it "Suspicious" (max 35)
            # This prevents the 78% result for humans.
            avg_video_risk = min(max(avg_video_risk, 35), 45)
            
        # Polarization: Only push UP if it's already very high (> 80)
        # Otherwise, if it's moderate, push it DOWN.
        if avg_video_risk < 60:
            avg_video_risk = avg_video_risk * 0.3 # Stronger push down for potential humans
        elif avg_video_risk > 80:
            avg_video_risk = min(99, avg_video_risk * 1.1)
            
        # "Dynamic Human" Override: If they move and blink, they are likely human
        if avg_humanity_credit > 30 and critical_indicators < 2:
            avg_video_risk = min(avg_video_risk, 10)
            
        # Final sanity check: If humanity credit is high, push it down even more
        if avg_humanity_credit > 25:
            avg_video_risk = min(avg_video_risk, 15)

        # --- AUDIO ANALYSIS ---
        audio_risk = 0
        audio_anomalies = []
        try:
            y, sr = librosa.load(temp_video_path, sr=None, duration=30)
            if len(y) > 0:
                audio_risk, audio_anomalies = analyze_audio_liveness(y, sr, {})
        except Exception: pass
        
        # Aggregate final score
        if audio_risk > 0:
            final_risk = (avg_video_risk * 0.7) + (audio_risk * 0.3)
        else:
            final_risk = avg_video_risk
            
        final_risk = min(int(final_risk), 99)
        status = "SAFE"
        if final_risk > 70: status = "CRITICAL DEEPFAKE"
        elif final_risk > 35: status = "SUSPICIOUS"
            
        return {
            "filename": file.filename,
            "risk_score": final_risk,
            "status": status,
            "anomalies": robust_anomalies + audio_anomalies,
            "details": {
                "video_risk": int(avg_video_risk),
                "audio_risk": int(audio_risk),
                "humanity_score": int(avg_humanity_credit * 2.8) # Normalized to 0-100 logic
            }
        }

    except Exception as e:
        print(f"[DeepGuard] Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"[DeepGuard] Cleaned up temp file: {temp_video_path}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

