import numpy as np
import librosa

def analyze_audio_liveness(audio_data, sample_rate, state):
    """
    Analyzes audio for AI-generated voice detection using spectral analysis.
    """
    risk_score = 0
    anomalies = []
    
    try:
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 1. Spectral Centroid Analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        avg_centroid = np.mean(spectral_centroids)
        
        if "audio_centroid_history" not in state:
            state["audio_centroid_history"] = []
        
        state["audio_centroid_history"].append(avg_centroid)
        state["audio_centroid_history"] = state["audio_centroid_history"][-30:]
        
        if len(state["audio_centroid_history"]) > 10:
            temporal_variance = np.var(state["audio_centroid_history"])
            if temporal_variance < 80000: 
                risk_score += 40 
                anomalies.append(f"Unnatural Voice Timbre (Spectral Var: {temporal_variance:.0f})")
        
        # 2. Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        zcr_variance = np.var(zcr)
        
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
            if pitch_variance < 80:
                risk_score += 30
                anomalies.append(f"Robotic Pitch Pattern (Var: {pitch_variance:.1f})")
        
        # 4. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        avg_rolloff = np.mean(rolloff)
        
        if avg_rolloff > 8500 or avg_rolloff < 1500:
            risk_score += 20
            anomalies.append(f"Unnatural Frequency Range (Rolloff: {avg_rolloff:.0f} Hz)")
            
    except Exception as e:
        print(f"[DeepGuard] Audio analysis error: {e}")
        return 0, []
    
    return min(risk_score, 100), anomalies

def check_emotional_consistency(y, sr, temporal_visual_energy):
    """
    Checks if audio loudness matches visual expression intensity.
    """
    try:
        if len(y) == 0: return 0, []
        
        # Sample audio energy (RMS)
        S = librosa.magphase(librosa.stft(y))[0]
        rms = librosa.feature.rms(S=S)[0]
        times_audio = librosa.times_like(rms, sr=sr)
        
        v_times = [p[0] for p in temporal_visual_energy]
        v_vals = [p[1] for p in temporal_visual_energy]
        
        if len(v_vals) > 10:
            audio_energy_at_frames = np.interp(v_times, times_audio, rms)
            avg_audio_energy = np.mean(audio_energy_at_frames)
            avg_visual_expression = np.var(v_vals)
            
            if avg_audio_energy > 0.05 and avg_visual_expression < 0.0005:
                return 40, ["Emotional Dissonance (High audio volume, Flat face)"]
    except Exception as e:
        print(f"[DeepGuard] Emotional consistency check failed: {e}")
        
    return 0, []
