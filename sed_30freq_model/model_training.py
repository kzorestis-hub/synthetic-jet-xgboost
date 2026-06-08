import h5py
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import skew, kurtosis
import joblib

# ==============================================================================
# 0. PHYSICS CONSTANTS
# ==============================================================================
c = 2.9979e10  # Speed of light (cm/s)
Z_0 = 3.086e21  # Reference distance (cm)
R_0 = 3.086e20  # Reference radius (cm)
gamma_ad = 4.0/3.0  # Adiabatic index for relativistic gas

# ==============================================================================
# 1. LOAD THE DATA
# ==============================================================================
print("Loading synthetic jet data...")
file_path = r"C:\Users\kzore\Desktop\jet_training_data_MACRO.h5" 
with h5py.File(file_path, "r") as hf:
    # Load as float64 for full precision with M_dot log transformation
    X_raw = hf["X_flux_profiles"][:].astype(np.float64)
    y = hf["y_parameters"][:].astype(np.float64)

print("Applying log10 transform to flux profiles...")
# Log10 shrinks massive physics numbers into small, safe numbers to prevent scaling crashes
X = np.log10(X_raw + 1e-30)

print(f"Data loaded! X shape: {X.shape}, y shape: {y.shape}")

# ==============================================================================
# 1.5 SPECTRAL FEATURE ENGINEERING (FOR 10-BAND MACROSCOPIC SEDs)
# ==============================================================================
print("Engineering spectral features from Integrated SEDs...")

# The EXACT 30 frequencies you generated in the mock data (10^9 to 10^15 Hz)
frequencies = np.logspace(9, 15, 30)

# 1. Peak Characteristics
sed_peak_flux = np.max(X, axis=1)
sed_peak_idx = np.argmax(X, axis=1)
# Map the index to the actual frequency value
sed_peak_freq = frequencies[sed_peak_idx]

# 2. Broadband Spectral Indices (Slopes)
# Slope between low freq (Radio, index 0) and mid freq (Infrared, index 4)
slope_low_mid = (X[:, 4] - X[:, 0]) / (np.log10(frequencies[4]) - np.log10(frequencies[0]))
# Slope between mid freq (Infrared, index 4) and high freq (Optical/UV, index 9)
slope_mid_high = (X[:, 9] - X[:, 4]) / (np.log10(frequencies[9]) - np.log10(frequencies[4]))

# 3. Overall Statistics
sed_mean = np.mean(X, axis=1)
sed_std = np.std(X, axis=1)
sed_range = np.max(X, axis=1) - np.min(X, axis=1)

# Stack the new engineered features
engineered_features = np.column_stack([
    sed_peak_flux, sed_peak_freq, slope_low_mid, slope_mid_high,
    sed_mean, sed_std, sed_range
])

feature_names = ['sed_peak_flux', 'sed_peak_freq', 'slope_low_mid', 'slope_mid_high',
                 'sed_mean', 'sed_std', 'sed_range']

X_combined = np.hstack([X, engineered_features])
X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

print(f"  ✓ Engineered {engineered_features.shape[1]} spectral features")
print(f"  ✓ Combined feature shape: {X_combined.shape} (10 Freqs + {len(feature_names)} Engineered)")

X = X_combined

# ==============================================================================
# 2. DATA SCALING & PHYSICS TRANSFORMS
# ==============================================================================
print("Applying Physics Transformations...")

# 1. Log transform Mass Accretion Rate (Column 0)
y[:, 0] = np.log10(y[:, 0] + 1e-30)

# 2. Transform Initial Velocity (v0) into Lorentz Factor (Column 2)
v0_c = y[:, 2] / c
y[:, 2] = 1.0 / np.sqrt(1.0 - v0_c**2 + 1e-10) 

print("Scaling features and targets...")
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

# ==============================================================================
# 3. TRAIN/TEST SPLIT
# ==============================================================================
# We hold back 20% of the data to test the AI on jets it has NEVER seen
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ==============================================================================
# 4. BUILD AND TRAIN THE XGBOOST MODEL WITH REGULARIZATION
# ==============================================================================
print("Building the XGBoost Regressor with regularization...")

reg_xgb = xgb.XGBRegressor(
    tree_method="hist",  
    n_estimators=2500, 
    early_stopping_rounds=20,  
    max_depth=7, 
    learning_rate=0.03,
    subsample=1, 
    colsample_bytree=1.0, 
    reg_alpha=0.001, 
    reg_lambda=0.1, 
    multi_strategy="one_output_per_tree",
    n_jobs=-2, 
    random_state=42 
)   

print("Training started...")
eval_set = [(X_train, y_train), (X_test, y_test)]

reg_xgb.fit(
    X_train, y_train, 
    eval_set=eval_set,
    verbose=50 
)

# ==============================================================================
# 4.5 CROSS-VALIDATION ON TRAINING DATA
# ==============================================================================
print("\nPerforming 5-fold cross-validation on training set...")
cv_results = cross_validate(
    xgb.XGBRegressor(
        tree_method="hist", 
        n_estimators=2500,
        max_depth=7,
        learning_rate=0.03,
        subsample=1,
        colsample_bytree=1.0,
        reg_alpha=0.001,
        reg_lambda=0.1,
        multi_strategy="one_output_per_tree",
        n_jobs=-2,
        random_state=42
    ),
    X_train, y_train,
    cv=5, 
    scoring='r2',
    n_jobs=1,  # CRITICAL: Set to 1 to avoid nested parallelization causing memory exhaustion
    verbose=1
)

print(f"\nCross-Validation Results:")
print(f"  Mean CV R² Score: {cv_results['test_score'].mean():.4f} (+/- {cv_results['test_score'].std():.4f})")
print(f"  Individual CV scores: {cv_results['test_score']}")

# ==============================================================================
# 5. INFERENCE AND EVALUATION
# ==============================================================================
print("\nMaking predictions on the test set...")
y_pred_scaled = reg_xgb.predict(X_test)

# CRITICAL FIX: Clip predictions to valid [0,1] range to prevent overflow after exponentiating
print(f"\nDiagnostics:")
print(f"  y_pred_scaled min: {y_pred_scaled.min():.4f}, max: {y_pred_scaled.max():.4f}")
print(f"  Expected range: [0, 1]")

# Clip predictions that fall outside [0, 1] to prevent inf/nan after exponentiation
y_pred_scaled_clipped = np.clip(y_pred_scaled, 0, 1)
out_of_bounds = np.sum((y_pred_scaled < 0) | (y_pred_scaled > 1))
print(f"  Predictions outside [0,1]: {out_of_bounds} / {y_pred_scaled.size}")

# Un-scale the data back from [0,1] range
y_pred_real = y_scaler.inverse_transform(y_pred_scaled_clipped)
y_test_real = y_scaler.inverse_transform(y_test)

print("Reversing physics transformations for accurate stats/plots...")

# 1. Undo the log10 on Mass Accretion Rate (Column 0)
y_pred_real[:, 0] = 10**(y_pred_real[:, 0])
y_test_real[:, 0] = 10**(y_test_real[:, 0])

# 2. Undo the Lorentz Factor back to Initial Velocity in cm/s (Column 2)
true_gamma = y_test_real[:, 2]
y_test_real[:, 2] = c * np.sqrt(np.maximum(0, 1.0 - (1.0 / (true_gamma**2 + 1e-10))))

pred_gamma = y_pred_real[:, 2]
y_pred_real[:, 2] = c * np.sqrt(np.maximum(0, 1.0 - (1.0 / (pred_gamma**2 + 1e-10))))

# ==============================================================================
# 6. PLOT TRUE VS PREDICTED (The Money Shot)
# ==============================================================================
param_names = ["Mass Accretion Rate (M_dot) [g/s]", "Enthalpy Factor (h_gamma) [dimensionless]", 
               "Initial Velocity (v0) [cm/s]", "Viewing Angle (theta_deg) [degrees]"]

# Set to 0 to plot Mass Accretion Rate. 
# Change to 1 for h_gamma, 2 for v0, 3 for theta_deg.
idx = 0 

plt.figure(figsize=(8, 6))  

# Special handling for velocity (idx == 2): convert to units of c
if idx == 2:
    true_v0_c = y_test_real[:, idx] / c
    pred_v0_c = y_pred_real[:, idx] / c
    
    plt.scatter(true_v0_c, pred_v0_c, 
                s=50, c="cornflowerblue", edgecolor="k", alpha=0.7, label="AI Predictions")
    
    min_val = np.min(true_v0_c)
    max_val = np.max(true_v0_c)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Accuracy")
    
    plt.xlabel("True Initial Velocity (v0) [c]", fontsize=12)
    plt.ylabel("AI Predicted Initial Velocity (v0) [c]", fontsize=12)
else:
    # Plot the AI's guesses vs the real answers
    plt.scatter(y_test_real[:, idx], y_pred_real[:, idx], 
                s=50, c="cornflowerblue", edgecolor="k", alpha=0.7, label="AI Predictions")
    
    # Draw the "Perfect Guess" diagonal line
    min_val = np.min(y_test_real[:, idx])
    max_val = np.max(y_test_real[:, idx])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Accuracy")
    
    plt.xlabel(f"True {param_names[idx]}", fontsize=12)
    plt.ylabel(f"AI Predicted {param_names[idx]}", fontsize=12)

plt.title(f"XGBoost Performance: {param_names[idx]}", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ==============================================================================
# 7. TRAINING CURVES - Shows model convergence over boosting rounds
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Training Curve: Overall Model Convergence', fontsize=14, fontweight='bold')

# Extract training history
results = reg_xgb.evals_result()

try:
    # validation_0 is the training set, validation_1 is the test set
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE', color='blue', lw=2)
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test RMSE', color='red', lw=2)
    
    ax.legend(fontsize=12)
    ax.set_ylabel('Global RMSE (Scaled Units)', fontsize=12)
    ax.set_xlabel('Boosting Round (Number of Trees)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
except KeyError as e:
    # Fallback just in case a different metric was used
    print(f"\nWarning: Could not find 'rmse' in evals_result().")
    print(f"Available top-level keys: {list(results.keys())}")
    if 'validation_0' in results:
        print(f"Metrics tracked: {list(results['validation_0'].keys())}")
        
    ax.text(0.5, 0.5, f"Data structure mismatch.\nCheck console for details.", 
            ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig(r"C:\Users\kzore\Desktop\learningcurve2.png", dpi=300, bbox_inches='tight')

plt.show()
# ==============================================================================
# 8. ALL 4 PARAMETERS - True vs Predicted comparison (2x2 grid)
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('All Parameters: True vs Predicted (Test Set)', fontsize=14, fontweight='bold')

for idx in range(4):
    ax = axes[idx // 2, idx % 2]
    
    if idx == 2:  # Velocity: convert to units of c
        true_vals = y_test_real[:, idx] / c
        pred_vals = y_pred_real[:, idx] / c
        unit_label = "[c]"
    else:
        true_vals = y_test_real[:, idx]
        pred_vals = y_pred_real[:, idx]
        unit_label = ""
    
    # Scatter plot
    ax.scatter(true_vals, pred_vals, s=30, c="cornflowerblue", 
              edgecolor="k", alpha=0.5, label="Predictions")
    
    # Perfect prediction line
    min_val = np.min(true_vals)
    max_val = np.max(true_vals)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Accuracy")
    
    # Calculate R² score
    ss_res = np.sum((pred_vals - true_vals)**2)
    ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
    r2_score = 1 - (ss_res / ss_tot)
    
    param_label = param_names[idx].split('[')[0].strip()
    ax.set_xlabel(f"True {param_label} {unit_label}", fontsize=11)
    ax.set_ylabel(f"Predicted {param_label} {unit_label}", fontsize=11)
    ax.set_title(f"{param_label} (R² = {r2_score:.4f})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\kzore\Desktop\resultsrun2.png", dpi=300, bbox_inches='tight')

plt.show()
# ==============================================================================
# 9. FEATURE IMPORTANCE - Which frequencies matter most?
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Get feature importance from the model
importance = reg_xgb.feature_importances_
indices = np.argsort(importance)[-20:]  # Top 20 features

# DYNAMICALLY generate labels based on the actual shape of X
num_raw_freqs = X.shape[1] - len(feature_names) 
raw_freq_names = [f"F{i}" for i in range(num_raw_freqs)]
all_feature_names = raw_freq_names + feature_names

# Map indices to their correct names
freq_labels = [all_feature_names[i] for i in indices]

ax.barh(freq_labels, importance[indices], color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Feature Importance (Gain)', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature Name', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most Important Features for Predictions', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(r"C:\Users\kzore\Desktop\feature_importance2.png", dpi=300, bbox_inches='tight')

plt.show()
# ==============================================================================
# 10. SUMMARY STATISTICS
# ==============================================================================
print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)
for idx, param_name in enumerate(param_names):
    true_vals = y_test_real[:, idx]
    pred_vals = y_pred_real[:, idx]
    
    # Convert v0 to c for display
    if idx == 2:
        true_vals = true_vals / c
        pred_vals = pred_vals / c
    
    mae = np.mean(np.abs(pred_vals - true_vals))
    rmse = np.sqrt(np.mean((pred_vals - true_vals)**2))
    r2 = 1 - (np.sum((pred_vals - true_vals)**2) / np.sum((true_vals - np.mean(true_vals))**2))
    
    print(f"\n{param_name}:")
    print(f"  MAE:  {mae:.4e}")
    print(f"  RMSE: {rmse:.4e}")
    print(f"  R²:   {r2:.4f}")

print("="*70)

print("Saving the model and scalers for future use...")
# Save the trained model
joblib.dump(reg_xgb, r"C:\Users\kzore\Desktop\trained_xgb_model5.joblib")

# Save the scalers (THIS IS CRITICAL!)
joblib.dump(X_scaler, r"C:\Users\kzore\Desktop\X_scaler5.joblib")
joblib.dump(y_scaler, r"C:\Users\kzore\Desktop\y_scaler5.joblib")

print("Saved successfully!") 