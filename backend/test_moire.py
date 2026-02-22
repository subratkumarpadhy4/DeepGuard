import cv2
import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.engines.forensics import analyze_moire_patterns

def test_moire_detection():
    print("--- Testing Moire Pattern (Anti-Replay) Detector ---")
    w, h = 640, 480
    
    # 1. Test with a clean "Natural" image (Real portrait-like content)
    # We create a gradient/face-like shape instead of random noise
    clean_img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            clean_img[y, x] = int(128 + 64 * np.sin(y/100) * np.cos(x/100))
    
    risk_clean, anoms_clean, _ = analyze_moire_patterns(clean_img, w, h)
    print(f"Natural Image Risk: {risk_clean}")

    # 2. Test with a synthetic Moire Pattern (Digital Screen Grid)
    # High frequency grid
    moire_img = clean_img.copy()
    for i in range(0, h, 4):
        moire_img[i:i+1, :] = 255
    for j in range(0, w, 4):
        moire_img[:, j:j+1] = 255
        
    risk_moire, anoms_moire, evid_moire = analyze_moire_patterns(moire_img, w, h)
    
    print(f"Moire Image Risk: {risk_moire}")
    print(f"Anomalies: {anoms_moire}")
    print(f"Evidence Boxes: {len(evid_moire)}")

    if risk_moire > 0 and risk_clean == 0:
        print("✅ SUCCESS: Moire pattern detected selectively.")
    else:
        print("❌ FAILURE: Calibration needed. Natural Risk: {}, Moire Risk: {}".format(risk_clean, risk_moire))

if __name__ == "__main__":
    test_moire_detection()
