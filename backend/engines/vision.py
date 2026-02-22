import cv2
import numpy as np
import scipy.signal
import mediapipe as mp

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, width, height, eye_indices):
    def get_point(index):
        point = landmarks[index]
        return np.array([point.x * width, point.y * height])

    p2, p6 = get_point(eye_indices[1]), get_point(eye_indices[5])
    p3, p5 = get_point(eye_indices[2]), get_point(eye_indices[4])
    p1, p4 = get_point(eye_indices[0]), get_point(eye_indices[3])

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def get_head_pose(landmarks, img_w, img_h):
    face_2d = []
    face_3d = []
    key_landmarks = [1, 152, 33, 263, 61, 291]

    for idx in key_landmarks:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        
        if idx == 1: face_3d.append([0, -1.12, 4.25])
        elif idx == 152: face_3d.append([0, -8.70, -0.65])
        elif idx == 33: face_3d.append([-3.5, 1.80, 0.20])
        elif idx == 263: face_3d.append([3.5, 1.80, 0.20])
        elif idx == 61: face_3d.append([-2.0, -4.50, 0.10])
        elif idx == 291: face_3d.append([2.0, -4.50, 0.10])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles[0], angles[1], angles[2]

def analyze_rppg(frame, landmarks, w, h, state, current_time):
    forehead_points = [10, 151, 67, 297]
    roi_pts = np.array([[landmarks[idx].x * w, landmarks[idx].y * h] for idx in forehead_points], dtype=np.int32)
    x, y, rw, rh = cv2.boundingRect(roi_pts)
    evidence = []
    
    if rw > 5 and rh > 5:
        roi_img = frame[y:y+rh, x:x+rw]
        pulse_val = np.mean(roi_img[:, :, 1])
        
        if "pulse_history" not in state: state["pulse_history"] = []
        state["pulse_history"].append((current_time, pulse_val))
        state["pulse_history"] = [p for p in state["pulse_history"] if current_time - p[0] < 10.0]
        
        if len(state["pulse_history"]) > 90:
            history_vals = scipy.signal.detrend(np.array([p[1] for p in state["pulse_history"]]))
            duration = state["pulse_history"][-1][0] - state["pulse_history"][0][0]
            if duration <= 0: return 0, [], 0, []
            
            fs = len(history_vals) / duration
            try:
                nyq = 0.5 * fs
                b, a = scipy.signal.butter(3, [0.7/nyq, 3.0/nyq], btype='band')
                filtered = scipy.signal.filtfilt(b, a, history_vals)
                fft_vals = np.abs(np.fft.rfft(filtered))
                freqs = np.fft.rfftfreq(len(filtered), 1/fs)
                
                peak_idx = np.argmax(fft_vals)
                dom_freq = freqs[peak_idx]
                snr = fft_vals[peak_idx] / np.mean(fft_vals)
                
                if 0.7 < dom_freq < 3.0 and snr > 2.5:
                    state["heart_bpm"] = int(dom_freq * 60)
                    return 0, [], 30, [] # Risk, Anomalies, Humanity Credit, Evidence
                elif snr < 1.2:
                    evidence.append({"type": "rppg_fail", "box": [x, y, rw, rh]})
                    return 15, ["Absence of Physiological Pulse (rPPG)"], 0, evidence
            except: pass
    return 0, [], 0, []
