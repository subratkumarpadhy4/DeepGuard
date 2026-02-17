import cv2
import numpy as np
import base64
import json
import time
import mediapipe as mp
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

def analyze_liveness(frame, state):
    current_time = time.time()
    risk_score = 0
    anomalies = []
    
    # Preprocessing for landmarks
    h, w, _ = frame.shape
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
    
    # Blink Threshold (Typical Human EAR threshold is ~0.2 - 0.3)
    # MediaPipe is very precise. < 0.2 usually means blink.
    BLINK_THRESHOLD = 0.2
    
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
    # Increased threshold to 12s to be more forgiving for focused humans
    if time_since_last_blink > 12.0:
        risk_score += 70
        anomalies.append(f"Abnormal Stare (No blink for {int(time_since_last_blink)}s)")
    elif time_since_last_blink > 8.0:
        risk_score += 30
        anomalies.append("Low Blink Rate")

    # Rule 2: Head Rotation Analysis (Statue Check)
    pitch, yaw, roll = get_head_pose(landmarks, w, h)
    
    # Store pose history for "Statue Check" (Time-based variance analysis)
    state["pose_history"].append((current_time, pitch, yaw, roll))
    
    # Keep only last 5 seconds of data
    state["pose_history"] = [p for p in state["pose_history"] if current_time - p[0] < 5.0]
    
    if len(state["pose_history"]) > 30: # Need sufficient frames (~1-2 sec min)
        # Calculate standard deviation (variance) of movement
        poses = np.array([p[1:] for p in state["pose_history"]]) # [[p, y, r], ...]
        std_dev = np.std(poses, axis=0) # [std_pitch, std_yaw, std_roll]
        
        # Thresholds: Humans naturally move slightly (micro-movements)
        # Deepfakes/Images are often perfectly still (std_dev near 0)
        # However, be careful not to flag a very focused person too aggressively.
        # A statue-like person usually has std_dev < 0.2 degrees over 5s
        
        avg_movement = np.mean(std_dev[:2]) # Check Pitch & Yaw mainly
        
        if avg_movement < 0.2: 
            risk_score += 60 # High risk for perfect stillness
            anomalies.append(f"Unnatural Stillness (Statue-like: {avg_movement:.2f}° var)")
        elif avg_movement < 0.5:
            risk_score += 20
            anomalies.append("Low Head Movement")

    return min(risk_score, 100), anomalies

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
            
            if "payload" in payload:
                image = decode_image(payload["payload"])
                
                if image is not None:
                    risk, reasons = analyze_liveness(image, client_states[client_id])
                    
                    response = {
                        "status": "processed",
                        "risk_score": int(risk),
                        "anomalies": reasons,
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
