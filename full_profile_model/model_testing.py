import h5py
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# ==============================================================================
# 0. CONSTANTS
# ==============================================================================
c = 2.9979e10

# ==============================================================================
# 1. LOAD THE NEW UNSEEN DATA (7,000 Slices)
# ==============================================================================
print("Loading the Final Exam data (7,000 slices)...")
file_path = r"C:\Users\kzore\Desktop\jet_training_data.TEST100.h5" 

with h5py.File(file_path, "r") as hf:
    X_raw = hf["X_flux_profiles"][:].astype(np.float64)
    # The targets are completely raw straight from the physics engine
    y_true = hf["y_parameters"][:].astype(np.float64)

print("Applying log10 transform to raw fluxes...")
X_new = np.log10(X_raw + 1e-30)

print(f"Test data loaded! X shape: {X_new.shape}, y shape: {y_true.shape}")

# ==============================================================================
# 2. EXACT MATCHING FEATURE ENGINEERING (For 7000 Slices)
# ==============================================================================
print("Engineering features from flux profiles...")

flux_mean = np.mean(X_new, axis=1)
flux_std = np.std(X_new, axis=1)
flux_min = np.min(X_new, axis=1)
flux_max = np.max(X_new, axis=1)
flux_range = flux_max - flux_min
flux_median = np.median(X_new, axis=1)
flux_q25 = np.percentile(X_new, 25, axis=1)
flux_q75 = np.percentile(X_new, 75, axis=1)
flux_peak_idx = np.argmax(X_new, axis=1) / X_new.shape[1] 
flux_peak_value = flux_max

# Shift for centroid/entropy math
X_shifted = X_new - np.min(X_new, axis=1, keepdims=True) + 1e-10

freq_indices = np.arange(X_new.shape[1])
flux_centroid = np.sum(X_shifted * freq_indices[np.newaxis, :], axis=1) / np.sum(X_shifted, axis=1)

flux_skewness = skew(X_new, axis=1)
flux_kurtosis = kurtosis(X_new, axis=1)

flux_normalized = X_shifted / np.sum(X_shifted, axis=1, keepdims=True)
flux_entropy = -np.sum(flux_normalized * np.log(flux_normalized + 1e-10), axis=1)

flux_slope = np.mean(np.diff(X_new, axis=1), axis=1)
flux_curvature = np.std(np.diff(X_new, axis=1), axis=1)

engineered_features = np.column_stack([
    flux_mean, flux_std, flux_min, flux_max, flux_range, flux_median,
    flux_q25, flux_q75, flux_peak_idx, flux_peak_value, flux_centroid,
    flux_skewness, flux_kurtosis, flux_entropy, flux_slope, flux_curvature
])

X_combined = np.hstack([X_new, engineered_features])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Combined feature shape ready for AI: {X_combined.shape}")

# ==============================================================================
# 3. WAKE UP THE AI AND SCALERS
# ==============================================================================
print("\nWaking up the trained model and scalers...")
# IMPORTANT: Ensure these are the ones saved from the 7000-slice model!
reg_xgb = joblib.load(r"C:\Users\kzore\Desktop\trained_xgb_model1old.joblib")
X_scaler = joblib.load(r"C:\Users\kzore\Desktop\X_scaler1old.joblib")
y_scaler = joblib.load(r"C:\Users\kzore\Desktop\y_scaler1old.joblib")

# ==============================================================================
# 4. PREDICT AND REVERSE PHYSICS
# ==============================================================================
print("Generating predictions...")
X_scaled = X_scaler.transform(X_combined)
y_pred_scaled = reg_xgb.predict(X_scaled)

# Clip to prevent math explosions, then unscale
y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
y_pred_real = y_scaler.inverse_transform(y_pred_scaled)

print("Reversing physics transformations on predictions...")
# 1. Undo the log10 on Mass Accretion Rate (Column 0)
y_pred_real[:, 0] = 10**(y_pred_real[:, 0])

# 2. Undo the Lorentz Factor back to Initial Velocity in cm/s (Column 2)
pred_gamma = y_pred_real[:, 2]
y_pred_real[:, 2] = c * np.sqrt(np.maximum(0, 1.0 - (1.0 / (pred_gamma**2 + 1e-10))))

# NOTE: y_true is loaded raw from the test generator, so it needs ZERO reversing.
# It is already in standard M_dot and v0 (cm/s).

# ==============================================================================
# 5. PLOT RESULTS FOR CARLA
# ==============================================================================
param_names = ["Mass Accretion Rate (M_dot)", "Enthalpy Factor (h_gamma)", 
               "Initial Velocity (v0)", "Viewing Angle (theta_deg)"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Final Exam (7000 Spatial Slices): AI Performance on 100 Unseen Jets', fontsize=16, fontweight='bold')

for idx in range(4):
    ax = axes[idx // 2, idx % 2]
    
    if idx == 2:  # Velocity (Convert to c for plotting)
        true_vals = y_true[:, idx] / c
        pred_vals = y_pred_real[:, idx] / c
        unit_label = "[c]"
    else:
        true_vals = y_true[:, idx]
        pred_vals = y_pred_real[:, idx]
        unit_label = ""
    
    ax.scatter(true_vals, pred_vals, s=50, c="crimson", 
               edgecolor="k", alpha=0.7, label="AI Predictions")
    
    min_val = np.min([np.min(true_vals), np.min(pred_vals)])
    max_val = np.max([np.max(true_vals), np.max(pred_vals)])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Perfect Accuracy")
    
    # Calculate R2
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
plt.savefig(r"C:\Users\kzore\Desktop\Final_Exam_7000.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nValidation complete! Plots saved to Desktop.")