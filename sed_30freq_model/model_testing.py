import h5py
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==============================================================================
# 0. SETUP AND CONSTANTS
# ==============================================================================
c = 2.9979e10
num_freqs = 30  
frequencies = np.logspace(9, 15, num_freqs)

# ==============================================================================
# 1. LOAD THE BRAND NEW MOCK DATA
# ==============================================================================
print("Loading the Final Exam data...")
file_path = r"C:\Users\kzore\Desktop\jet_training_data_MACRO.TEST100.h5" 

with h5py.File(file_path, "r") as hf:
    X_raw = hf["X_flux_profiles"][:].astype(np.float64)
    y_true = hf["y_parameters"][:].astype(np.float64)

X_new = np.log10(X_raw + 1e-30)
print(f"Test data loaded! X shape: {X_new.shape}, y shape: {y_true.shape}")
print(f"Model Log-Flux Average: {np.mean(X_new):.4f}")
print(f"Model Log-Flux Max: {np.max(X_new):.4f}")

# ==============================================================================
# 2. EXACT MATCHING FEATURE ENGINEERING (THE FIX)
# ==============================================================================
print("Applying spectral feature engineering...")

sed_peak_flux = np.max(X_new, axis=1)
sed_peak_idx = np.argmax(X_new, axis=1)
sed_peak_freq = frequencies[sed_peak_idx]

# THE FIX: We MUST use the exact same column indices you used during training!
# If you changed these to something else in your training script, change them here too!
idx_mid = 4
idx_high = 9

slope_low_mid = (X_new[:, idx_mid] - X_new[:, 0]) / (np.log10(frequencies[idx_mid]) - np.log10(frequencies[0]))
slope_mid_high = (X_new[:, idx_high] - X_new[:, idx_mid]) / (np.log10(frequencies[idx_high]) - np.log10(frequencies[idx_mid]))

sed_mean = np.mean(X_new, axis=1)
sed_std = np.std(X_new, axis=1)
sed_range = np.max(X_new, axis=1) - np.min(X_new, axis=1)

engineered_features = np.column_stack([
    sed_peak_flux, sed_peak_freq, slope_low_mid, slope_mid_high,
    sed_mean, sed_std, sed_range
])

X_combined = np.hstack([X_new, engineered_features])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

# ==============================================================================
# 3. WAKE UP THE AI
# ==============================================================================
print("Waking up the trained model and scalers...")
# Make sure these filenames point to your 30-frequency trained model!
reg_xgb = joblib.load(r"C:\Users\kzore\Desktop\trained_xgb_model5.joblib")
X_scaler = joblib.load(r"C:\Users\kzore\Desktop\X_scaler5.joblib")
y_scaler = joblib.load(r"C:\Users\kzore\Desktop\y_scaler5.joblib")

# ==============================================================================
# 4. PREDICT AND REVERSE PHYSICS
# ==============================================================================
print("Generating predictions...")
X_scaled = X_scaler.transform(X_combined)
y_pred_scaled = reg_xgb.predict(X_scaled)

y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
y_pred_real = y_scaler.inverse_transform(y_pred_scaled)

# ---> THE MISSING FIX <---
# 1. Undo the log10 on Mass Accretion Rate (Column 0)
y_pred_real[:, 0] = 10**(y_pred_real[:, 0])

# 2. Undo the Lorentz Factor back to Initial Velocity in cm/s (Column 2)
pred_gamma = y_pred_real[:, 2]
y_pred_real[:, 2] = c * np.sqrt(np.maximum(0, 1.0 - (1.0 / (pred_gamma**2 + 1e-10))))

# ==============================================================================
# 5. PLOT RESULTS 
# ==============================================================================
param_names = ["Mass Accretion Rate (M_dot)", "Enthalpy Factor (h_gamma)", 
               "Initial Velocity (v0)", "Viewing Angle (theta_deg)"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Final Exam: Performance on 100 Unseen Jets (FIXED)', fontsize=16, fontweight='bold')

for idx in range(4):
    ax = axes[idx // 2, idx % 2]
    
    if idx == 2:  
        true_vals = y_true[:, idx] / c
        pred_vals = y_pred_real[:, idx] / c
        unit_label = "[c]"
    else:
        true_vals = y_true[:, idx]
        pred_vals = y_pred_real[:, idx]
        unit_label = ""
    
    ax.scatter(true_vals, pred_vals, s=50, c="seagreen", 
               edgecolor="k", alpha=0.7, label="AI Predictions")
    
    min_val = np.min([np.min(true_vals), np.min(pred_vals)])
    max_val = np.max([np.max(true_vals), np.max(pred_vals)])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Accuracy")
    
    ss_res = np.sum((pred_vals - true_vals)**2)
    ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    param_label = param_names[idx].split('(')[0].strip()
    ax.set_xlabel(f"True {param_label} {unit_label}", fontsize=11)
    ax.set_ylabel(f"Predicted {param_label} {unit_label}", fontsize=11)
    ax.set_title(f"{param_label} (R² = {r2:.4f})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()