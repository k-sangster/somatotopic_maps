from koulakov_based_model import Conditions
import pandas as pd
import os
import pyabc
import numpy as np

# load data
BASE_DIR = "./data/"
PROFILE_DIR = os.path.join(BASE_DIR, "profiles/")

def load_profile(filename: str, profile_dir: str) -> np.ndarray:
    """ Load the profile from a csv file.
    """
    profile_path = os.path.join(profile_dir, filename)
    return pd.read_csv(profile_path, header=None).to_numpy().flatten()

OBSERVED_PROFILES = {
    f"{Conditions.WT.value}_Distal_profile": load_profile("WT_distal_avg.csv", PROFILE_DIR), 
    f"{Conditions.WT.value}_Proximal_profile": load_profile("WT_proximal_avg.csv", PROFILE_DIR), 
    f"{Conditions.TEN3_DRG_KO.value}_Proximal_profile": load_profile("bulk_avg_cKO_Ten3 DRG cKO.csv", PROFILE_DIR),
    f"{Conditions.LPHN2_DRG_KO.value}_Distal_profile": load_profile("bulk_avg_cKO_Lphn2 DRG cKO.csv", PROFILE_DIR),
    f"{Conditions.TEN3_DH_KO.value}_Distal_profile": load_profile("bulk_avg_cKO_Ten3 DH cKO (Distal).csv", PROFILE_DIR), 
    f"{Conditions.TEN3_DH_KO.value}_Proximal_profile": load_profile("bulk_avg_cKO_Ten3 DH cKO (Proximal).csv", PROFILE_DIR),
    f"{Conditions.LPHN2_DH_KO.value}_Proximal_profile": load_profile("bulk_avg_cKO_Lphn2 DH cKO.csv", PROFILE_DIR)
}

# priors
PRIORS_WT = {
    # Activity‑dependent parameters (Eq. 1.2)
    "gamma": pyabc.RV("uniform", 0, 10),  # range also based on original Koulakov et al. 2006 paper
    # Competition parameters (Eqs. 1.4‑1.5)
    "A_comp": pyabc.RV("uniform", 0, 1000),
    "B_comp": pyabc.RV("uniform", 0, 10),
    "D_comp": pyabc.RV("uniform", 0, 10),
    # Chemoaffinity parameters (Eq. 1.11)
    "alpha_ca": pyabc.RV("uniform", 0, 100),
    # Additional binding and rate constants
    "k_tt": pyabc.RV("uniform", 0, 1),
    "k_lt": pyabc.RV("uniform", 0, 1),
    "k_tl": pyabc.RV("uniform", 0, 1),
    # Parameters for how cKO affects E_activity
    "a_t_drg": pyabc.RV("uniform", 0, 5),
    "b_t_drg": pyabc.RV("uniform", 0, 2),
    "a_l_drg": pyabc.RV("uniform", 0, 5),
    "b_l_drg": pyabc.RV("uniform", 0, 2),
    "a_t_dh": pyabc.RV("uniform", 0, 5),
    "b_t_dh": pyabc.RV("uniform", 0, 2),
    "a_l_dh": pyabc.RV("uniform", 0, 5),
    "b_l_dh": pyabc.RV("uniform", 0, 2)
}
PRIORS_ALL_CONDITIONS = PRIORS_WT.copy()  
