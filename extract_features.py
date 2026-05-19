import numpy as np
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def compute_magnitude(data):
    """Convert (frames, H, W, 2) vector field to magnitude (frames, H, W)."""
    return np.linalg.norm(data, axis=-1)


def spatial_mean(data):
    """Average over all spatial pixels: (frames,)"""
    return data.mean(axis=(1, 2))


def spatial_max(data):
    """Max over all pixels per frame: (frames,)"""
    return data.max(axis=(1, 2))


def detect_contact(signal_1d, threshold_factor=3.0):
    """
    Detect press start/end using threshold = factor * noise_std.
    signal_1d: 1D array (frames,)
    Returns: (start_idx, end_idx) or (None, None)
    """
    # Estimate noise from first 20 frames (assume no contact)
    noise_std = np.std(signal_1d[:20])
    threshold = threshold_factor * noise_std
    
    above = signal_1d > threshold
    if not np.any(above):
        return None, None
    
    idx = np.where(above)[0]
    return idx[0], idx[-1]


# ------------------------------------------------------------
# Main feature extraction function
# ------------------------------------------------------------

def extract_features(deform_array, shear_array):
    """
    Input:
        deform_array: (frames, 240, 320, 2) from sensor.getDeformation2D()
        shear_array:  (frames, 240, 320, 2) from sensor.getShear()
    Returns:
        dict of ~25 features
    """
    
    # --- Compute magnitudes and spatial averages ---
    d_mag = compute_magnitude(deform_array)       # (frames, 240, 320)
    s_mag = compute_magnitude(shear_array)         # (frames, 240, 320)
    
    d_avg = spatial_mean(d_mag)                   # (frames,)
    d_max = spatial_max(d_mag)                     # (frames,)
    s_avg = spatial_mean(s_mag)                    # (frames,)
    
    # --- Contact detection ---
    start, end = detect_contact(d_avg)
    
    if start is None or end is None or (end - start) < 5:
        return None   # no valid press detected
    
    d_seg = d_avg[start:end+1]
    s_seg = s_avg[start:end+1]
    n_frames = len(d_seg)
    
    # --- Time axis (relative frames) ---
    t = np.arange(n_frames)
    
    # ====================================================
    # FRUIT PAPER FEATURES (hardness from force dynamics)
    # ====================================================
    
    # 1. Maximum deformation during press
    max_deformation = np.max(d_seg)
    
    # 2. Time to peak (rise time)
    peak_idx = np.argmax(d_seg)
    time_to_peak = peak_idx  # in frames
    
    # 3. Rise slope (dF/dt as per fruit paper Eq.3)
    baseline = np.mean(d_avg[:20])  # pre-contact baseline
    if time_to_peak > 0:
        rise_slope = (d_seg[peak_idx] - baseline) / time_to_peak
    else:
        rise_slope = 0.0
    
    # 4. Hold mean (middle 50% of contact)
    q1 = int(0.25 * n_frames)
    q3 = int(0.75 * n_frames)
    hold_mean = np.mean(d_seg[q1:q3])
    
    # 5. Recovery slope (after peak to end)
    post_peak = d_seg[peak_idx:]
    if len(post_peak) > 1:
        recovery_slope = (post_peak[-1] - post_peak[0]) / len(post_peak)
    else:
        recovery_slope = 0.0
    
    # 6. Deformation energy (area under curve)
    deformation_energy = np.trapz(d_seg - baseline)
    
    # 7. Normalized energy (energy / n_frames)
    deformation_energy_norm = deformation_energy / n_frames
    
    # 8. Second derivative mean (d²F/dt² — fruit paper ripeness metric)
    if n_frames > 2:
        d2 = np.diff(d_seg, n=2)
        second_deriv_mean = np.mean(np.abs(d2))
    else:
        second_deriv_mean = 0.0
    
    # ====================================================
    # FG-CLTP INSPIRED FEATURES (shear)
    # ====================================================
    
    # 9. Shear magnitude mean during contact
    shear_mean = np.mean(s_seg)
    
    # 10. Shear magnitude std
    shear_std = np.std(s_seg)
    
    # 11. Shear variance during hold phase
    shear_hold_std = np.std(s_seg[q1:q3])
    
    # 12. Shear-to-deformation ratio (relative tangential force)
    shear_deform_ratio = shear_mean / (max_deformation + 1e-8)
    
    # 13. Shear peak
    shear_peak = np.max(s_seg)
    
    # ====================================================
    # GENERAL STATISTICAL FEATURES
    # ====================================================
    
    # Deformation statistics
    # 14-19
    d_mean = np.mean(d_seg)
    d_std = np.std(d_seg)
    d_min = np.min(d_seg)
    d_skew = skew(d_seg)
    d_kurt = kurtosis(d_seg)
    d_rms = np.sqrt(np.mean(d_seg ** 2))
    
    # Shear statistics
    # 20-25
    s_mean = np.mean(s_seg)
    s_std = np.std(s_seg)
    s_min = np.min(s_seg)
    s_skew = skew(s_seg)
    s_kurt = kurtosis(s_seg)
    s_rms = np.sqrt(np.mean(s_seg ** 2))
    
    # ====================================================
    # SPATIAL FEATURES
    # ====================================================
    
    # 26. Spatial deformation gradient (texture indicator)
    # Mean absolute difference between neighbouring taxels over contact frames
    d_spatial_grad = np.mean(np.abs(np.diff(d_mag[start:end+1], axis=1))) + \
                     np.mean(np.abs(np.diff(d_mag[start:end+1], axis=2)))
    
    # 27. Spatial shear heterogeneity
    s_spatial_std = np.mean(np.std(s_mag[start:end+1], axis=(1, 2)))
    
    # ====================================================
    # FREQUENCY FEATURES
    # ====================================================
    
    # 28. Dominant frequency (FFT peak)
    if n_frames > 4:
        fft_vals = np.abs(fft(d_seg - np.mean(d_seg)))
        fft_vals = fft_vals[:n_frames // 2]
        dom_freq = np.argmax(fft_vals[1:]) + 1 if len(fft_vals) > 1 else 0
        dom_freq_energy = fft_vals[dom_freq] / (np.sum(fft_vals) + 1e-8)
    else:
        dom_freq = 0
        dom_freq_energy = 0.0
    
    # ====================================================
    # ASSEMBLE FEATURE DICT
    # ====================================================
    
    features = {
        # Fruit paper (hardness)
        'max_deformation':       max_deformation,
        'time_to_peak':          time_to_peak,
        'rise_slope':            rise_slope,
        'hold_mean':             hold_mean,
        'recovery_slope':        recovery_slope,
        'deformation_energy':    deformation_energy,
        'deformation_energy_norm': deformation_energy_norm,
        'second_deriv_mean':     second_deriv_mean,
        
        # FG-CLTP (shear)
        'shear_mean':            shear_mean,
        'shear_std':             shear_std,
        'shear_hold_std':        shear_hold_std,
        'shear_deform_ratio':    shear_deform_ratio,
        'shear_peak':            shear_peak,
        
        # Statistical — deformation
        'd_mean':                d_mean,
        'd_std':                 d_std,
        'd_min':                 d_min,
        'd_skew':                d_skew,
        'd_kurt':                d_kurt,
        'd_rms':                 d_rms,
        
        # Statistical — shear
        's_mean':                s_mean,
        's_std':                 s_std,
        's_min':                 s_min,
        's_skew':                s_skew,
        's_kurt':                s_kurt,
        's_rms':                 s_rms,
        
        # Spatial
        'd_spatial_grad':        d_spatial_grad,
        's_spatial_std':         s_spatial_std,
        
        # Frequency
        'dom_freq':              dom_freq,
        'dom_freq_energy':       dom_freq_energy,
        
        # Meta
        'n_frames':              n_frames,
    }
    
    return features


# ------------------------------------------------------------
# Quick test on your captured data
# ------------------------------------------------------------
if __name__ == "__main__":
    d = np.load('test_deformation.npy')
    s = np.load('test_shear.npy')
    
    feats = extract_features(d, s)
    
    if feats is not None:
        print("=== Extracted Features ===")
        for key, value in feats.items():
            print(f"  {key:25s}: {value:.4f}" if isinstance(value, float) else f"  {key:25s}: {value}")
    else:
        print("No valid press detected in the test data.")
