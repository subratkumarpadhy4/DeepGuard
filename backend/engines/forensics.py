import cv2
import numpy as np

def analyze_skin_texture(gray_frame, landmarks, w, h, state):
    risk = 0
    anomalies = []
    evidence = []
    try:
        cheek_lm = landmarks[205]
        cx, cy = int(cheek_lm.x * w), int(cheek_lm.y * h)
        patch_size = int(w * 0.08)
        
        y1, y2 = max(0, cy-patch_size), min(h, cy+patch_size)
        x1, x2 = max(0, cx-patch_size), min(w, cx+patch_size)
        
        if y2>y1 and x2>x1:
            cheek_roi = gray_frame[y1:y2, x1:x2]
            texture_score = cv2.Laplacian(cheek_roi, cv2.CV_64F).var()
            
            fshift = np.fft.fftshift(np.fft.fft2(cheek_roi))
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
            magnitude_spectrum[center_y-5:center_y+5, center_x-5:center_x+5] = 0
            
            high_freq_val = np.mean(magnitude_spectrum)
            
            if "texture_history" not in state: state["texture_history"] = []
            if "fft_history" not in state: state["fft_history"] = []
            state["texture_history"].append(texture_score)
            state["fft_history"].append(high_freq_val)
            state["texture_history"] = state["texture_history"][-30:]
            state["fft_history"] = state["fft_history"][-30:]
            
            # High-end AI (Veo/Gemini) adds fake grain to mimic texture, 
            # but it is mathematically uniform (Zero Noise Floor).
            # We look for "Synthetic Stability" - pixels that are too stable over time.
            if "stability_history" not in state: state["stability_history"] = []
            frame_diff = texture_score # Using variance as a proxy for stability
            state["stability_history"].append(frame_diff)
            state["stability_history"] = state["stability_history"][-60:]
            
            avg_texture = np.mean(state["texture_history"])
            avg_fft = np.mean(state["fft_history"])
            stability_var = np.var(state["stability_history"]) if len(state["stability_history"]) > 20 else 100
            
            # Real human skin has imperfections/pores. AI skin is "Perfectly Smooth" (Gauss-like).
            if avg_texture < 130.0: # Tightened from 95.0
                risk += 40
                anomalies.append(f"Unnatural Skin Smoothness (AI Gradient: {avg_texture:.1f})")
                evidence.append({"type": "texture", "box": [x1, y1, x2-x1, y2-y1]})
            
            if stability_var < 6.5: # Real skin on camera has thermal/sensor jitter. AI is too stable.
                risk += 45
                anomalies.append("Synthetic Image Stability (Neural Ghosting detected)")
                evidence.append({"type": "stability", "box": [x1, y1, x2-x1, y2-y1]})

            if avg_fft < 5.5: 
                risk += 35
                anomalies.append("Lack of Natural Image Grain (AI Cleanliness)")
                evidence.append({"type": "fft_low", "box": [x1, y1, x2-x1, y2-y1]})
    except Exception as e:
        print(f"[DeepGuard] Skin analysis failed: {e}")
    return risk, anomalies, evidence

def analyze_perimeter_artifacts(gray_frame, landmarks, w, h):
    risk = 0
    anomalies = []
    evidence = []
    try:
        # Simplified perimeter box calculation: bounding box of oval landmarks
        oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        edge_map = cv2.Canny(gray_frame, 100, 200)
        
        perimeter_edges = []
        xs, ys = [], []
        for idx in oval_indices:
            p = landmarks[idx]
            lx, ly = int(p.x * w), int(p.y * h)
            if 0 <= lx < w and 0 <= ly < h:
                roi = edge_map[max(0, ly-2):min(h, ly+2), max(0, lx-2):min(w, lx+2)]
                perimeter_edges.append(np.mean(roi))
                xs.append(lx)
                ys.append(ly)
        
        if len(perimeter_edges) > 10:
            avg_edge = np.mean(perimeter_edges)
            # Real-world background contrast causes high variance. 
            # AI masking causes "Ringing" (thin, ultra-bright edges).
            # We check if a significant portion of the perimeter has "extreme" edges.
            edge_peaks = [e for e in perimeter_edges if e > 200]
            peak_ratio = len(edge_peaks) / len(perimeter_edges)

            box = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
            
            if peak_ratio > 0.15: # Over 15% of the perimeter is "pure white" edge (ringing)
                risk += 30
                anomalies.append("Mask Seam Artifact (Perimeter Ringing)")
                evidence.append({"type": "mask_seam", "box": box})
            elif avg_edge < 2.0: 
                risk += 20
                anomalies.append("Face-to-Neck Blend Alpha (Perimeter Blur)")
                evidence.append({"type": "perimeter_blur", "box": box})
    except Exception as e:
        print(f"[DeepGuard] Perimeter analysis failed: {e}")
    return risk, anomalies, evidence

def analyze_brand_watermarks(gray_frame, w, h):
    risk = 0
    anomalies = []
    evidence = []
    try:
        cw, ch = int(w*0.30), int(h*0.30)
        # regions mapped to their coordinates
        region_coords = [
            (0, 0, cw, ch), (w-cw, 0, cw, ch),
            (0, h-ch, cw, ch), (w-cw, h-ch, cw, ch),
            (int(w*0.35), 0, int(w*0.3), int(h*0.15)),
            (int(w*0.35), int(h*0.85), int(w*0.3), int(h*0.15))
        ]
        
        center_roi = gray_frame[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
        base_edge_density = np.mean(cv2.Canny(center_roi, 100, 200)) + 0.1
        
        for (rx, ry, rw, rh) in region_coords:
            roi = gray_frame[ry:ry+rh, rx:rx+rw]
            if roi.size == 0: continue
            
            # High-fidelity digital logos are extremely sharp
            edges = cv2.Canny(roi, 100, 180)
            roi_density = np.mean(edges)
            
            # Use adaptive threshold to find small text blobs
            binary_roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Text-based logos like 'Veo' are typically small blobs
                if 15 < area < 4000:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    ar = float(bw)/bh if bh > 0 else 1
                    
                    # LOGO SIGNATURE: 
                    # 1. Distinct contrast relative to surroundings
                    # 2. Text-like aspect ratio
                    if (0.3 < ar < 6.0) and (roi_density > base_edge_density * 2.5):
                        found = True
                        evidence.append({"type": "ai_logo", "box": [rx+bx, ry+by, bw, bh]})
                        break
            if found:
                risk += 80 # Very high risk for overt AI branding
                anomalies.append("Generative AI Brand Signature detected (Veo/Sora Signature)")
                break
    except Exception as e:
        print(f"[DeepGuard] Logo analysis failed: {e}")
    return risk, anomalies, evidence

def analyze_face_swap_seams(gray_frame, landmarks, w, h):
    risk = 0
    anomalies = []
    evidence = []
    try:
        # Landmarks specifically for the upper face/hairline and jawline transition
        seam_indices = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338]
        
        edge_samples = []
        box_points = []
        
        for idx in seam_indices:
            p = landmarks[idx]
            lx, ly = int(p.x * w), int(p.y * h)
            if 10 <= lx < w-10 and 10 <= ly < h-10:
                # Sample a small patch around the seam
                patch = gray_frame[ly-5:ly+5, lx-5:lx+5]
                # Calculate Laplacian variance to detect blur
                blur_score = cv2.Laplacian(patch, cv2.CV_64F).var()
                edge_samples.append(blur_score)
                box_points.append((lx, ly))
        
        if len(edge_samples) > 15:
            avg_blur = np.mean(edge_samples)
            # If the seam is unnaturally blurry compared to normal sharp facial edges
            if avg_blur < 15.0:
                risk += 35
                anomalies.append(f"Face-Swap Edge Blending (Seam Blur: {avg_blur:.1f})")
                
                # Create evidence boxes for the detected blurry segments
                xs, ys = zip(*box_points)
                evidence.append({"type": "edge_seam", "box": [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]})
    except Exception as e:
        print(f"[DeepGuard] Face-swap seam analysis failed: {e}")
    return risk, anomalies, evidence

def analyze_moire_patterns(gray_frame, w, h):
    risk = 0
    anomalies = []
    evidence = []
    try:
        # Sample center region
        cw, ch = int(w*0.4), int(h*0.4)
        rx, ry = int(w*0.3), int(h*0.3)
        roi = gray_frame[ry:ry+ch, rx:rx+cw]
        
        # 2D FFT
        dft = np.fft.fft2(roi)
        dft_shift = np.fft.fftshift(dft)
        mag = np.abs(dft_shift)
        
        # Log scale for analysis
        mag = 20 * np.log(mag + 1)
        
        # Mask center (low frequencies)
        rows, cols = mag.shape
        crow, ccol = rows//2, cols//2
        mask_size = 15
        mag[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
        
        # Natural images have high std deviation in FFT because they are random/complex.
        # Screen grids produce VERY sharp peaks. We look for freq concentration.
        peak_val = np.max(mag)
        avg_val = np.mean(mag)
        
        # The Ratio of max/avg is much higher in structured grids
        peak_ratio = peak_val / (avg_val + 0.0001)
        
        if peak_val > 230 and peak_ratio > 3.3:
            risk += 45
            anomalies.append("Replay Attack Detected (Screen Interference)")
            evidence.append({"type": "moire_grid", "box": [rx, ry, cw, ch]})
    except Exception as e:
        print(f"[DeepGuard] Moire analysis failed: {e}")
    return risk, anomalies, evidence
