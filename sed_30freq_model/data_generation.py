import numpy as np
import h5py
from tqdm import tqdm

# ==============================================================================
# 1. CONSTANTS & NEW TELESCOPE FREQUENCIES
# ==============================================================================
c = 2.9979e10
Z_0 = 3.086e21
R_0 = 3.086e20
gamma_ad = 4.0/3.0
z_values = np.geomspace(Z_0, 300.0 * Z_0, 1000)

# The "Observatory" Approach: 15 logarithmically spaced frequencies 
# covering Radio (1 GHz) to Gamma-Ray (10^20 Hz)
target_freqs = np.logspace(9, 15, 30)
num_features = len(target_freqs)

# ==============================================================================
# 2. YOUR PHYSICS ENGINE
# ==============================================================================
class RelativisticJetCalculator:
    def __init__(self, pressure_law_func, p_total_0, rho0, v0, r0, b0, gamma_ad, theta_deg):
        self.pressure_law_func = pressure_law_func
        self.P_total_0 = p_total_0
        self.rho0 = rho0
        self.v0 = v0
        self.r0 = r0
        self.B_0 = b0
        self.c = 2.9979e10
        self.theta_rad = theta_deg * np.pi / 180.0

        self.gamma_adiabatic = gamma_ad
        self.p_exponent = 1.0 / self.gamma_adiabatic
        self.h_factor = self.gamma_adiabatic / (self.gamma_adiabatic - 1.0)

        self.gamma0 = 1.0 / np.sqrt(1.0 - (self.v0 / self.c)**2)
        self.A0 = np.pi * self.r0**2

        self.P_th_0 = self.P_total_0
        self.B_const = self.rho0 / (self.P_th_0 ** self.p_exponent)
        self.h0_hydro = 1.0 + (self.h_factor * self.P_th_0) / (self.rho0 * self.c**2)

        self.C1_const = self.h0_hydro * self.gamma0
        self.C2_const = self.A0 * self.gamma0 * self.v0 * self.rho0
        self.C3_B_model = self.B_0 * np.sqrt(self.v0 * self.gamma0 / self.rho0)
        
        self.conical_locked = False
        self.z_prev = None
        self.r_prev = None
        self.gamma_prev = self.gamma0

    def get_properties(self, z):
        P_z_test = self.pressure_law_func(z)
        if P_z_test < 0.0: return None

        rho_test = self.B_const * (P_z_test ** self.p_exponent)
        h_test = 1.0 + (self.h_factor * P_z_test) / (rho_test * self.c**2)
        gamma_test = self.C1_const / h_test
        v_test = self.c * np.sqrt(1.0 - 1.0 / gamma_test**2)
        denom_test = rho_test * gamma_test * v_test
        area_test = self.C2_const / denom_test
        radius_test = np.sqrt(area_test / np.pi)
        cs_test = np.sqrt(self.gamma_adiabatic * P_z_test / (rho_test * h_test))
        gamma_cs_test = np.sqrt(1 / (1 - (cs_test / self.c)**2))
        theta_M = gamma_cs_test * cs_test / (gamma_test * v_test)

        if not self.conical_locked and self.z_prev is not None and self.r_prev is not None:
            dr_dz = (radius_test - self.r_prev) / (z - self.z_prev)
            if dr_dz >= theta_M:   
                self.conical_locked = True
                self.r_z_lock = self.r_prev / self.z_prev

        if self.conical_locked:
            radius = (self.r_prev / self.z_prev) * z
            area = np.pi * radius**2
            gamma_low = self.gamma_prev  
            gamma_high = self.C1_const  

            for _ in range(40):
                gamma_mid = 0.5 * (gamma_low + gamma_high)
                v_mid = self.c * np.sqrt(1.0 - 1.0 / gamma_mid**2)
                rho_mid = self.C2_const / (area * gamma_mid * v_mid)
                P_mid = (rho_mid / self.B_const)**(1.0 / self.p_exponent)
                h_mid = 1.0 + (self.h_factor * P_mid) / (rho_mid * self.c**2)

                if h_mid * gamma_mid > self.C1_const:
                    gamma_high = gamma_mid
                else:
                    gamma_low = gamma_mid

            gamma = 0.5 * (gamma_low + gamma_high)
            v = self.c * np.sqrt(1.0 - 1.0 / gamma**2)
            rho = self.C2_const / (area * gamma * v)
            P_z = (rho / self.B_const)**(1.0 / self.p_exponent)
        else:
            P_z = P_z_test
            rho = rho_test
            gamma = gamma_test
            v = v_test
            radius = radius_test

        self.z_prev = z
        self.r_prev = radius
        self.gamma_prev = gamma  

        B_prime = self.C3_B_model * np.sqrt(rho / (v * gamma))

        # Spectrum Generation
        dz_array = np.gradient(z_values)
        d = 30 * 3.086e24  # 30 Mpc
        area_constant = 4.0 * np.pi * d**2.0
        z_index = np.searchsorted(z_values, z)
        z_index = min(z_index, len(z_values) - 1) 
        dz = dz_array[z_index]
        bita = v / self.c
        eta = 0.1
        U_int = 3.0 * P_z
        U_e_cell = eta * U_int * dz * np.pi * radius**2
        m_e = 9.10938356e-28

        E_e_min = m_e * self.c**2.0
        E_e_max = 1e6 * m_e * self.c**2.0 * (rho / self.rho0)**(1.0 / 3.0)
        sigma_T = 6.6524587158e-25
        N_o = U_e_cell / (np.log(E_e_max / E_e_min))
        q_e = 4.80320425e-10  
        E_values = np.geomspace(E_e_min, E_e_max, 100)
        
        n_c_list_thita1 = []
        nLn_app_list_thita1 = []
        
        for i in range(len(E_values)):
            E_e_1 = E_values[i]
            gamma_e = E_e_1 / (m_e * self.c**2.0)
            bita_e = np.sqrt(1.0 - 1.0 / gamma_e**2.0)
            DEradDt = (4.0 / 3.0) * sigma_T * bita_e**2.0 * self.c * (B_prime**2.0 / (8.0 * np.pi)) * (E_e_1 / (m_e * self.c**2.0))**2.0
            N_e = N_o * (E_e_1**-2.0)
            nLn = 0.5 * E_e_1 * N_e * DEradDt  
            n_c = 3.0 / (4.0 * np.pi) * q_e * B_prime * E_e_1**2.0 / (m_e**3.0 * self.c**5.0)

            thita = self.theta_rad
            delta = 1.0 / (gamma * (1.0 - bita * np.cos(thita))) 
            nLn_obs = nLn * delta**3.0 / gamma  
            n_c_obs = n_c * delta  
            
            n_c_list_thita1.append(n_c_obs)
            nLn_app_list_thita1.append(nLn_obs)

        Ln_array_1 = np.array(nLn_app_list_thita1)
        nu_array_1 = np.array(n_c_list_thita1)
        F_obs_1 = Ln_array_1 / (nu_array_1 * area_constant) * 10**26  
        F_obs_1_z = F_obs_1 * z / dz

        return {'n_c_list_thita1': n_c_list_thita1, 'F_obs_1_z': F_obs_1_z}


# ==============================================================================
# 3. CHECKPOINT & SAVE SYSTEM
# ==============================================================================
def save_checkpoint(checkpoint_num):
    """Save current progress to HDF5 file."""
    X_array = np.array(X_features)
    y_array = np.array(y_targets)
    
    if checkpoint_num == 1:
        with h5py.File(desktop_path, "w") as hf:
            # SHAPE UPDATED: Now saves 15 features instead of 7000
            hf.create_dataset("X_flux_profiles", data=X_array, maxshape=(None, num_features))
            hf.create_dataset("y_parameters", data=y_array, maxshape=(None, 4))
        print(f"  ✓ Checkpoint {checkpoint_num}: Saved {len(X_features)} samples")
    else:
        with h5py.File(desktop_path, "a") as hf:
            total_samples = len(X_features)
            hf["X_flux_profiles"].resize((total_samples, num_features))
            hf["y_parameters"].resize((total_samples, 4)) 
            
            start_idx = total_samples - checkpoint_interval
            hf["X_flux_profiles"][start_idx:] = X_array[start_idx:]
            hf["y_parameters"][start_idx:] = y_array[start_idx:]
        print(f"  ✓ Checkpoint {checkpoint_num}: Appended to {total_samples} total samples")

# ==============================================================================
# 4. AI DATA GENERATION PIPELINE
# ==============================================================================
num_samples = 100
X_features = [] 
y_targets = []  

# Checkpoint configuration
desktop_path = r"C:\Users\kzore\Desktop\jet_training_data_MACRO.TEST100.h5" 
checkpoint_interval = 500 
checkpoint_counter = 0

print(f"Starting generation of {num_samples} Macroscopic SEDs...")
print(f"Tracking {num_features} frequencies per jet.")

for i in tqdm(range(num_samples)):
    try:
        rand_v0 = np.random.uniform(0.4 * c, 0.99 * c)
        rand_gamma0 = 1.0 / np.sqrt(1.0 - (rand_v0/c)**2)
        rand_M_dot_base = 10**np.random.uniform(43, 45) 
        rand_h_gamma = np.random.uniform(8.0, 20.0)
        rand_theta_deg = np.random.uniform(10.0, 30.0)
        
        rand_M_dot = rand_M_dot_base / ((rand_h_gamma - 1.0) * c**2)
        rand_h0 = rand_h_gamma / rand_gamma0
        rand_rho0 = rand_M_dot / (rand_v0 * rand_gamma0 * np.pi * R_0**2)
        rand_p0 = (rand_h0 - 1.0) * (rand_rho0 * c**2) / 4.0
        rand_b0 = np.sqrt(0.01 * 8.0 * np.pi * 3.0 * rand_p0)
        
        def pressure_law(z):
            return rand_p0 * (z / Z_0)**(-2.0)

        engine = RelativisticJetCalculator(
            pressure_law_func=pressure_law,
            p_total_0=rand_p0, rho0=rand_rho0, v0=rand_v0, r0=R_0, b0=rand_b0, gamma_ad=gamma_ad,
            theta_deg=rand_theta_deg
        )
        
        # INTEGRATION START: Create an empty array to sum the flux across all z-slices
        integrated_sed = np.zeros(num_features)
        freq_coverage_valid = True
        
        for z in z_values:
            data = engine.get_properties(z)
            if data and 'n_c_list_thita1' in data and len(data['n_c_list_thita1']) > 0:
                freqs = np.array(data['n_c_list_thita1'])
                fluxes = np.array(data['F_obs_1_z']) 
                
                val = np.interp(
                    target_freqs, 
                    freqs, 
                    fluxes,
                    left=0.0,  
                    right=0.0  
                )
                # INTEGRATION CONTINUED: Summing the flux instead of appending it
                integrated_sed += val
            else:
                freq_coverage_valid = False
                break
        
        # Quality Check
        if (freq_coverage_valid and not np.isnan(integrated_sed).any() and np.max(integrated_sed) > 0):
            X_features.append(integrated_sed)
            y_targets.append([rand_M_dot, rand_h_gamma, rand_v0, rand_theta_deg])
            
            if len(X_features) % checkpoint_interval == 0:
                checkpoint_counter += 1
                save_checkpoint(checkpoint_counter)
            
    except Exception as e:
        print(f"Sample {i} failed: {e}") 
        continue 

# Final save
if len(X_features) > 0 and (len(X_features) % checkpoint_interval != 0):
    checkpoint_counter += 1
    save_checkpoint(checkpoint_counter)
    
print(f"\n{'='*70}")
print(f"✓ GENERATION COMPLETE!")
print(f"{'='*70}")
print(f"Generated {len(X_features)} valid samples out of {num_samples} attempted")
print(f"X shape: {np.array(X_features).shape}")
print(f"y shape: {np.array(y_targets).shape}")
print(f"File saved to: {desktop_path}")
print(f"{'='*70}")