import cv2
import numpy as np
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.engines.forensics import analyze_face_swap_seams

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def test_deepfake_seam():
    print("--- Testing Face-Swap Seam Detector ---")
    w, h = 640, 480
    
    # Create a very flat image with almost no edges
    img = np.ones((h, w), dtype=np.uint8) * 128
    
    # Add a VERY SUBTLE "face" rectangle
    cv2.rectangle(img, (200, 100), (440, 380), 130, -1)
    
    # Apply massive blur across the whole image to kill variance
    img = cv2.GaussianBlur(img, (51, 51), 0)

    # Create mock landmarks along the "seam"
    landmarks = [MockLandmark(0.5, 0.5)] * 500 
    
    seam_indices = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338]
    
    for i, idx in enumerate(seam_indices):
        angle = (i / len(seam_indices)) * 2 * np.pi
        lx = 320 + int(120 * np.cos(angle))
        ly = 240 + int(140 * np.sin(angle))
        landmarks[idx] = MockLandmark(lx / w, ly / h)

    risk, anoms, evid = analyze_face_swap_seams(img, landmarks, w, h)

    print(f"Result Risk: {risk}")
    print(f"Anomalies: {anoms}")
    print(f"Evidence Boxes: {len(evid)}")
    
    if risk > 0:
        print("✅ SUCCESS: Deepfake seam detected with low variance.")
    else:
        print("❌ FAILURE: Seam was not detected. Threshold might be too strict or test data too sharp.")

if __name__ == "__main__":
    test_deepfake_seam()
