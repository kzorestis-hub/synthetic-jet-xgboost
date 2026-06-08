import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. CONSTANTS & SETUP
# ==============================================================================
c = 2.9979e10

# ---> IMPORTANT: Update this path to your original 7,000 slice dataset!
file_path = r"C:\Users\kzore\Desktop\jet_training_data.h5" 

print("Loading 7,000-slice dataset...")
with h5py.File(file_path, "r") as hf:
    # Assuming X_flux_profiles contains the 7000 slices
    X_raw = hf["X_flux_profiles"][:]
    y_raw = hf["y_parameters"][:]

print(f"Data loaded. X shape: {X_raw.shape}, y shape: {y_raw.shape}")

# ==============================================================================
# 1. FEATURE ENGINEERING (Must match your original training script)
# ==============================================================================
print("Applying Feature Engineering...")

# (Note: If you used specific features for the 7000-slice model, like PCA 
# or specific slice indices, you must put that logic here. I am using a 
# generic statistical extraction assuming X_raw is [jets, 7000])

X_new = np.log10(X_raw + 1e-30)

# Example Generic Features (Update to match your specific 7000-slice logic)
sed_mean = np.mean(X_new, axis=1, keepdims=True)
sed_std = np.std(X_new, axis=1, keepdims=True)
sed_max = np.max(X_new, axis=1, keepdims=True)
sed_min = np.min(X_new, axis=1, keepdims=True)

X_combined = np.hstack([X_new, sed_mean, sed_std, sed_max, sed_min])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

# Target Scaling
y_scaled = np.copy(y_raw)
y_scaled[:, 0] = np.log10(y_scaled[:, 0]) # Log M_dot
gamma = 1.0 / np.sqrt(1.0 - (y_scaled[:, 2] / c)**2)
y_scaled[:, 2] = gamma # Convert v0 to Lorentz factor

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled_full = X_scaler.fit_transform(X_combined)
y_scaled_full = y_scaler.fit_transform(y_scaled)

# ==============================================================================
# 2. K-FOLD CROSS-VALIDATION (FIXED)
# ==============================================================================
print("\n--- Starting 5-Fold Cross Validation ---")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Set these parameters to match your BEST run perfectly.
base_xgb = XGBRegressor(
    n_estimators=1500,     # <-- Set this to the EXACT number of trees your best run finished at

    max_depth=4,           # <-- Set to your best run's max_depth
    learning_rate=0.02,    # <-- Set to your best run's learning_rate
    subsample=0.8,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-2
    # Notice: early_stopping_rounds is completely removed!
)

model = MultiOutputRegressor(base_xgb)

fold_r2_scores = []
fold_number = 1

all_y_true = np.zeros_like(y_scaled_full)
all_y_pred = np.zeros_like(y_scaled_full)

for train_index, test_index in kf.split(X_scaled_full):
    print(f"Training Fold {fold_number}/5...")
    X_train, X_test = X_scaled_full[train_index], X_scaled_full[test_index]
    y_train, y_test = y_scaled_full[train_index], y_scaled_full[test_index]
    
    # Train the model normally (no eval_set needed since we fixed the tree count)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Store predictions for the residual histograms
    all_y_true[test_index] = y_test
    all_y_pred[test_index] = y_pred
    
    # Evaluate
    score = r2_score(y_test, y_pred)
    fold_r2_scores.append(score)
    print(f"  Fold {fold_number} R² Score: {score:.4f}")
    fold_number += 1

mean_r2 = np.mean(fold_r2_scores)
std_r2 = np.std(fold_r2_scores)
print(f"\n--- Cross-Validation Complete ---")
print(f"Global Average R²: {mean_r2:.4f} ± {std_r2:.4f}")
# ==============================================================================
# 3. RESIDUAL HISTOGRAMS
# ==============================================================================
print("\nGenerating Residual Plots...")

# Calculate residuals (True - Predicted)
# We calculate this in the scaled space to compare standard deviations fairly
residuals = all_y_true - all_y_pred

param_names = ["Log(M_dot)", "h_gamma", "Lorentz Factor (Gamma)", "Viewing Angle (Theta)"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Residual Error Distributions (7,000-Slice Baseline)\nGlobal CV R² = {mean_r2:.4f} ± {std_r2:.4f}', fontsize=16, fontweight='bold')

for idx in range(4):
    ax = axes[idx // 2, idx % 2]
    
    res_data = residuals[:, idx]
    
    # Plot histogram
    ax.hist(res_data, bins=50, color='royalblue', edgecolor='black', alpha=0.7, density=True)
    
    # Add a vertical line at 0 (Perfect Prediction)
    ax.axvline(0, color='red', linestyle='dashed', linewidth=2, label="Zero Error")
    
    # Calculate stats for the title
    mean_res = np.mean(res_data)
    std_res = np.std(res_data)
    
    ax.set_title(f"{param_names[idx]}\nMean Error: {mean_res:.4f}, Std: {std_res:.4f}", fontsize=12)
    ax.set_xlabel("Residual Error (Scaled Units)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()