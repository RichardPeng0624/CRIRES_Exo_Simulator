import numpy as np
import pandas as pd
import os

# Define the CSV file name
csv_file = 'simulation_log.csv'

# Define the column names
columns = [
    'mass_p', 'mass_s',
    'r_p', 'r_s',
    'teff_p', 'teff_s',
    'logg_p', 'logg_s',
    'M_p', 'M_s',
    'vsini_p', 'vsini_s',
    'age', 'v_sys',
    'SNR_ccf_full', 'SNR_ccf_H2O', 'SNR_ccf_CO', 'SNR_ccf_CH4',
    'Delta_L_full'
]

# Function to initialize the CSV if it doesn't exist
def initialize_log(path):
    if not os.path.exists(path+csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
        print(f"Initialized new log file: {csv_file}")
    else:
        print(f"Log file already exists: {csv_file}")

# Function to log a single simulation
def log_simulation(data_dict):
    df = pd.DataFrame([data_dict])
    df.to_csv(csv_file, mode='a', header=not, os.path.exists(csv_file), index=False)
    print("Simulation logged.")