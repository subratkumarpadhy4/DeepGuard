import cv2
import numpy as np

# --- DIGITAL WATERMARKING (Integrity Feature) ---
# Hex signature for 'DEEPGUARD_VERIFIED'
WATERMARK_SIG = [0x44, 0x47, 0x55, 0x41, 0x52, 0x44, 0x5F, 0x56] 

def embed_watermark(frame):
    """Embeds a hidden signature in the LSB of the first few pixels."""
    h, w, _ = frame.shape
    for i, byte in enumerate(WATERMARK_SIG):
        for bit in range(8):
            pixel_idx = i * 8 + bit
            if pixel_idx >= w: break
            # Embed bit in Blue channel LSB
            bit_val = (byte >> (7 - bit)) & 1
            frame[0, pixel_idx, 0] = (frame[0, pixel_idx, 0] & 0xFE) | bit_val
    return frame

def check_watermark(frame):
    """Extracts the hidden signature from the LSB."""
    try:
        extracted_sig = []
        for i in range(len(WATERMARK_SIG)):
            byte = 0
            for bit in range(8):
                pixel_idx = i * 8 + bit
                bit_val = frame[0, pixel_idx, 0] & 1
                byte = (byte << 1) | bit_val
            extracted_sig.append(byte)
        return extracted_sig == WATERMARK_SIG
    except Exception:
        return False
