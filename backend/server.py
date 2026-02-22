import cv2
import numpy as np
import base64
import json
import time
import shutil
import os
import tempfile
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Engine Module Imports
from backend.engines.vision import face_mesh, calculate_ear, get_head_pose, analyze_rppg, LEFT_EYE, RIGHT_EYE
from backend.engines.forensics import analyze_skin_texture, analyze_perimeter_artifacts, analyze_brand_watermarks, analyze_face_swap_seams, analyze_moire_patterns
from backend.engines.audio import analyze_audio_liveness, check_emotional_consistency
from backend.engines.integrity import embed_watermark, check_watermark
from backend.engines.text import PhishingTranscriptAnalyzer

app = FastAPI()
phishing_analyzer = PhishingTranscriptAnalyzer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client_states = {}

def decode_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except: return None

def analyze_liveness(frame, state, override_timestamp=None):
    current_time = override_timestamp if override_timestamp is not None else time.time()
    h, w, _ = frame.shape
    risk_score = 0
    humanity_credit = 0
    anomalies = []
    evidence = []
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        state["face_lost_time"] = current_time
        return 0, ["No face detected"], {}, []

    landmarks = results.multi_face_landmarks[0].landmark

    # 1. Vision Metrics
    avg_ear = (calculate_ear(landmarks, w, h, LEFT_EYE) + calculate_ear(landmarks, w, h, RIGHT_EYE)) / 2.0
    pitch, yaw, roll = get_head_pose(landmarks, w, h)
    
    # 2. rPPG (Heartbeat)
    p_risk, p_anoms, p_credit, p_evid = analyze_rppg(frame, landmarks, w, h, state, current_time)
    risk_score += p_risk
    anomalies.extend(p_anoms)
    humanity_credit += p_credit
    evidence.extend(p_evid)

    # 3. Forensics (Skin & FFT)
    s_risk, s_anoms, s_evid = analyze_skin_texture(gray_frame, landmarks, w, h, state)
    risk_score += s_risk
    anomalies.extend(s_anoms)
    evidence.extend(s_evid)

    # 4. Perimeter Artifacts
    perim_risk, perim_anoms, perim_evid = analyze_perimeter_artifacts(gray_frame, landmarks, w, h)
    risk_score += perim_risk
    anomalies.extend(perim_anoms)
    evidence.extend(perim_evid)

    # 5. Brand Watermarks
    brand_risk, brand_anoms, brand_evid = analyze_brand_watermarks(gray_frame, w, h)
    risk_score += brand_risk
    anomalies.extend(brand_anoms)
    evidence.extend(brand_evid)

    # 6. Face-Swap Seams (Edge Blending)
    fs_risk, fs_anoms, fs_evid = analyze_face_swap_seams(gray_frame, landmarks, w, h)
    risk_score += fs_risk
    anomalies.extend(fs_anoms)
    evidence.extend(fs_evid)

    # 7. Moire Pattern (Anti-Replay)
    m_risk, m_anoms, m_evid = analyze_moire_patterns(gray_frame, w, h)
    risk_score += m_risk
    anomalies.extend(m_anoms)
    evidence.extend(m_evid)

    # 6. Eye & Motion Dynamics
    state["pose_history"].append((current_time, pitch, yaw, roll))
    state["pose_history"] = [p for p in state["pose_history"] if current_time - p[0] < 5.0]
    
    avg_movement = 0
    if len(state["pose_history"]) > 15:
        history = state["pose_history"]
        times = np.array([h[0] for h in history])
        poses = np.array([h[1:] for h in history])
        
        # Handle angle wraparound (e.g. 179 to -179 should be a 2 deg jump, not 358)
        diffs = np.diff(poses, axis=0)
        diffs = (diffs + 180) % 360 - 180 
        
        dts = np.diff(times)
        valid_dts = dts > 0
        
        if np.any(valid_dts):
            # Calculate velocities in deg/sec
            velocities = np.abs(diffs[valid_dts]) / dts[valid_dts][:, np.newaxis]
            # Use 95th percentile to ignore single-frame tracking glitches
            max_v = np.percentile(velocities, 95)
            
            if max_v > 650.0: # High threshold for synthetic jitter
                risk_score += 40
                anomalies.append(f"Robotic/Jerky Motion (Velocity: {max_v:.1f}\u00b0/s)")
                evidence.append({"type": "jerky_motion", "box": [int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8)]})

    # Blink Logic
    if avg_ear < 0.24:
        if not state.get("eyes_closed", False):
            state["eyes_closed"] = True
            state["last_blink"] = current_time
            state["blink_count"] += 1
    else: state["eyes_closed"] = False

    time_since_last_blink = current_time - state.get("last_blink", current_time)
    if time_since_last_blink > 25.0:
        risk_score += 30
        anomalies.append(f"Abnormal Stare (No blink for {int(time_since_last_blink)}s)")
        # Box the eyes
        evidence.append({"type": "stare", "box": [int(landmarks[33].x * w), int(landmarks[159].y * h), int((landmarks[263].x - landmarks[33].x) * w), int((landmarks[145].y - landmarks[159].y) * h)]})

    if len(state["pose_history"]) > 90 and avg_movement < 0.2:
        risk_score += 20
        anomalies.append(f"Unnatural Stillness (Statue-like: {avg_movement:.2f}\u00b0 var)")
        evidence.append({"type": "statue", "box": [int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6)]})

    # Humanity Credits
    if 0.5 < avg_movement < 5.0: humanity_credit += 20
    
    # Final Scoring
    final_risk = risk_score
    final_risk = max(0, final_risk - humanity_credit)
    
    # Cap non-critical risk
    pos_ai = ["Skin Smoothness", "Stability", "Cleanliness", "Periodic Artifacts", "Mouth Rendering", "Jerky Motion", "Soulless Eyes", "Pulse", "Image Detected", "Perimeter", "AI Brand", "Edge Blending", "Replay Attack"]
    has_pos = any(any(ind in a for ind in pos_ai) for a in anomalies)
    
    if not has_pos: final_risk = min(final_risk, 35)
    else: final_risk = max(final_risk, 40)

    debug_info = {
        "pitch": float(pitch), "yaw": float(yaw), "roll": float(roll),
        "variance": float(avg_movement), "humanity_credit": humanity_credit,
        "mar": float(np.linalg.norm(np.array([landmarks[13].x*w, landmarks[13].y*h]) - np.array([landmarks[14].x*w, landmarks[14].y*h])) / 
                    np.linalg.norm(np.array([landmarks[78].x*w, landmarks[78].y*h]) - np.array([landmarks[308].x*w, landmarks[308].y*h]))),
        "brow_dist": float(np.linalg.norm(np.array([landmarks[70].x, landmarks[70].y]) - np.array([landmarks[107].x, landmarks[107].y])))
    }

    return min(final_risk, 100), anomalies, debug_info, evidence

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cid = id(websocket)
    client_states[cid] = {"last_blink": time.time(), "blink_count": 0, "eyes_closed": False, "pose_history": []}
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            v_risk, a_risk, anoms, debug, evidence = 0, 0, [], {}, []
            
            if "payload" in payload:
                img = decode_image(payload["payload"])
                if img is not None:
                    v_risk, v_anoms, debug, v_evid = analyze_liveness(img, client_states[cid])
                    anoms.extend(v_anoms)
                    evidence.extend(v_evid)
            
            if "audio_payload" in payload:
                ab64 = payload["audio_payload"]
                audio_bytes = base64.b64decode(ab64.split(",")[1] if "," in ab64 else ab64)
                a_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                a_risk, a_anoms = analyze_audio_liveness(a_data, payload.get("sample_rate", 16000), client_states[cid])
                anoms.extend(a_anoms)
            
            await websocket.send_json({
                "status": "processed", "risk_score": int(v_risk * 0.6 + a_risk * 0.4),
                "video_risk": int(v_risk), "audio_risk": int(a_risk), "anomalies": anoms,
                "debug_info": debug, "evidence": evidence, "timestamp": time.time()
            })
    except WebSocketDisconnect: pass

@app.post("/analyze/video")
async def analyze_video_upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        state = {"last_blink": 0.0, "blink_count": 0, "eyes_closed": False, "pose_history": [], "mar_history": [], "iris_history": [], "texture_history": [], "fft_history": []}
        v_risk_acc, hum_acc, anom_counts, f_count, is_wm, t_vis = 0, 0, {}, 0, False, []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if f_count == 0: is_wm = check_watermark(frame)
            f_count += 1
            if f_count % 2 != 0: continue
            
            risk, anoms, debug, evid = analyze_liveness(frame, state, override_timestamp=f_count/fps)
            v_risk_acc += risk
            hum_acc += debug.get("humanity_credit", 0)
            t_vis.append((f_count/fps, debug.get("mar", 0) + debug.get("brow_dist", 0)))
            for a in anoms: anom_counts[a] = anom_counts.get(a, 0) + 1
            if risk > 30: # Only save evidence for suspicious frames
                state["last_evidence"] = evid
            
        cap.release()
        processed = f_count // 2
        if processed == 0: 
            return {"status": "error", "message": "Video too short for analysis."}
            
        avg_v_risk = v_risk_acc / processed
        avg_hum = hum_acc / processed
        
        robust_anoms = []
        h_art, s_art, b_ind = 0, 0, 0
        for anom, count in anom_counts.items():
            freq = count / processed
            # Critical descriptors for high-end AI
            is_critical = any(x in anom for x in ["Rendering", "Motion", "Eyes", "Artifacts", "Signature", "Stability", "Ghosting", "Cleanliness", "Smoothness", "Grain"])
            is_suspicious = any(x in anom for x in ["Pulse", "Detected", "Perimeter", "Alpha", "Stare", "Stillness"])
            
            if freq > 0.05 and is_critical: # 5% is enough to catch high-end generative signatures
                h_art += 1
                robust_anoms.append(anom)
            elif freq > 0.20 and is_suspicious:
                s_art += 1
                robust_anoms.append(anom)

        # Scale and Normalize
        if avg_hum > 5: avg_v_risk = max(2, avg_v_risk - (avg_hum * 2.2))
        
        # If absolutely NO critical AI indicators were consistently found, keep risk low
        if (h_art + s_art + b_ind) == 0: 
            avg_v_risk = min(avg_v_risk, 12)
        
        # More gradual risk floors
        is_ai_branded = any("AI Brand Signature" in a for a in robust_anoms)
        
        if is_ai_branded:
            avg_v_risk = max(avg_v_risk, 95) # Instant critical for AI branding
        elif h_art >= 2: 
            avg_v_risk = max(avg_v_risk, 92)
        elif h_art == 1:
            avg_v_risk = max(avg_v_risk, 65) 
        elif (s_art + b_ind) >= 3:
            avg_v_risk = max(avg_v_risk, 55)
        elif (s_art + b_ind) >= 1:
            avg_v_risk = min(max(avg_v_risk, 25), 45)
        
        # Audio & Emotion
        y, sr = librosa.load(tmp_path, sr=None, duration=30)
        a_risk, a_anoms = analyze_audio_liveness(y, sr, {})
        e_risk, e_anoms = check_emotional_consistency(y, sr, t_vis)
        
        # Aggregate final score
        # If video risk is very high, don't let clean audio dilute it (Critical Override)
        if avg_v_risk > 75:
            final_risk = int(avg_v_risk)
        elif a_risk > 0 or e_risk > 0:
            final_risk = int((avg_v_risk * 0.7) + (max(a_risk, e_risk) * 0.3))
        else:
            final_risk = int(avg_v_risk)
        if is_wm: 
            return {
                "status": "success",
                "risk_score": 0, 
                "verdict": "AUTHENTICATED HUMAN", 
                "anomalies": ["Verified Digital Identity"],
                "details": {"video_risk": 0, "audio_risk": 0, "humanity_score": 100}
            }
        
        status = "SAFE"
        if final_risk > 70: status = "CRITICAL DEEPFAKE"
        elif final_risk > 35: status = "SUSPICIOUS"
            
        return {
            "status": "success",
            "risk_score": min(final_risk, 99), 
            "verdict": status, 
            "anomalies": robust_anoms + a_anoms + e_anoms,
            "evidence": state.get("last_evidence", []),
            "details": {
                "video_risk": int(avg_v_risk),
                "audio_risk": int(max(a_risk, e_risk)),
                "humanity_score": int(avg_hum * 2.8)
            }
        }
    except Exception as e:
        print(f"[DeepGuard] Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

@app.post("/verify/sign-video")
async def sign_video_endpoint(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        in_p = tmp.name
    out_p = in_p.replace(".mp4", "_signed.mp4")
    try:
        cap = cv2.VideoCapture(in_p)
        out = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS) or 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if f_idx == 0: frame = embed_watermark(frame)
            out.write(frame)
            f_idx += 1
        cap.release(); out.release()
        return FileResponse(out_p, filename=f"signed_{file.filename}")
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
