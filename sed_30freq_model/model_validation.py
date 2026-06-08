import h5py
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. PHYSICS CONSTANTS & SETUP
# ==============================================================================
c = 2.9979e10

# ---> IMPORTANT: Verify this path points to your 30-frequency dataset!
file_path = r"C:\Users\kzore\Desktop\jet_training_data_MACRO.h5" 

print("Loading 30-frequency synthetic jet data...")
with h5py.File(file_path, "r") as hf:
    X_raw = hf["X_flux_profiles"][:].astype(np.float64)
    y_raw = hf["y_parameters"][:].astype(np.float64)

print(f"Data loaded. X shape: {X_raw.shape}, y shape: {y_raw.shape}")

# ==============================================================================
# 1. FEATURE ENGINEERING (Exactly matching your original script)
# ==============================================================================
print("Applying Physics Transformations and Feature Engineering...")

# 1. Log10 on Flux
X = np.log10(X_raw + 1e-30)

# 2. Extract Spectral Features
frequencies = np.logspace(9, 15, 30)

sed_peak_flux = np.max(X, axis=1)
sed_peak_idx = np.argmax(X, axis=1)
sed_peak_freq = frequencies[sed_peak_idx]

slope_low_mid = (X[:, 4] - X[:, 0]) / (np.log10(frequencies[4]) - np.log10(frequencies[0]))
slope_mid_high = (X[:, 9] - X[:, 4]) / (np.log10(frequencies[9]) - np.log10(frequencies[4]))

sed_mean = np.mean(X, axis=1)
sed_std = np.std(X, axis=1)
sed_range = np.max(X, axis=1) - np.min(X, axis=1)

engineered_features = np.column_stack([
    sed_peak_flux, sed_peak_freq, slope_low_mid, slope_mid_high,
    sed_mean, sed_std, sed_range
])

X_combined = np.hstack([X, engineered_features])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

# 3. Target Transforms
y_transformed = np.copy(y_raw)
y_transformed[:, 0] = np.log10(y_transformed[:, 0] + 1e-30) # Log M_dot
v0_c = y_transformed[:, 2] / c
y_transformed[:, 2] = 1.0 / np.sqrt(1.0 - v0_c**2 + 1e-10) # Lorentz Factor

# 4. Scaling
X_scaler = StandardScaler()
y_scaler = MinMaxScaler() # Using MinMaxScaler matching your script

X_scaled_full = X_scaler.fit_transform(X_combined)
y_scaled_full = y_scaler.fit_transform(y_transformed)

print(f"  ✓ Engineered feature shape: {X_scaled_full.shape}")

# ==============================================================================
# 2. K-FOLD CROSS-VALIDATION
# ==============================================================================
print("\n--- Starting 5-Fold Cross Validation (30-Frequency Model) ---")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Using the exact best-run parameters from your script
# (Note: early_stopping_rounds removed for K-Fold stability)
model = xgb.XGBRegressor(
    tree_method="hist",  
    n_estimators=2500,       # Exact tree count from your script
    max_depth=7, 
    learning_rate=0.03,
    subsample=1.0, 
    colsample_bytree=1.0, 
    reg_alpha=0.001, 
    reg_lambda=0.1, 
    multi_strategy="one_output_per_tree", # Handles multi-output natively in modern XGBoost
    n_jobs=-1, 
    random_state=42 
)

fold_r2_scores = []
fold_number = 1

# Arrays to store out-of-fold predictions for the residual analysis
all_y_true = np.zeros_like(y_scaled_full)
all_y_pred = np.zeros_like(y_scaled_full)

for train_index, test_index in kf.split(X_scaled_full):
    print(f"Training Fold {fold_number}/5...")
    X_train, X_test = X_scaled_full[train_index], X_scaled_full[test_index]
    y_train, y_test = y_scaled_full[train_index], y_scaled_full[test_index]
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Store predictions for residuals
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

# Calculate residuals (True - Predicted) in scaled space
residuals = all_y_true - all_y_pred

param_names = ["Log(M_dot)", "h_gamma", "Lorentz Factor (Gamma)", "Viewing Angle (Theta)"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Residual Error Distributions (30-Frequency Model)\nGlobal CV R² = {mean_r2:.4f} ± {std_r2:.4f}', fontsize=16, fontweight='bold')

for idx in range(4):
    ax = axes[idx // 2, idx % 2]
    
    res_data = residuals[:, idx]
    
    # Plot histogram
    ax.hist(res_data, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7, density=True)
    
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