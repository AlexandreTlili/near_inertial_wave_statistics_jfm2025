import os
import numpy as np
from tqdm import tqdm

import scipy.fft as fft

# Parameters
pathExperiment = './'
path_to_fields = os.path.join(pathExperiment, 'fields0')
path_to_setup = os.path.join(pathExperiment, 'setup0.npz')
path_to_save_all = os.path.join(pathExperiment, 'output_simu_0.npz')

assert not os.path.isfile(path_to_save_all)

# Get all files
key_sort = (lambda name: int(name[len('field_'):-len('.npz')]))
fieldFiles = [entry for entry in os.listdir(path_to_fields) if entry.startswith("field") and os.path.isfile(os.path.join(path_to_fields, entry))]
fieldFiles = sorted(fieldFiles, key=key_sort)
assert len(fieldFiles) > 0

# Read all files for concatenation
time_list = []
M_hat_list = []

for file in tqdm(fieldFiles):
    filepath = os.path.join(path_to_fields, file)
    data_file = np.load(filepath)

    time_list.append(data_file['t'])
    M_hat_list.append(data_file['M_hat'])

time = np.array(time_list).reshape((len(fieldFiles,)))
M_hat = np.array(M_hat_list)

# Read the setup
data_setup = np.load(path_to_setup)
xx, yy = data_setup['x'], data_setup['y']
kx, ky = data_setup['kx'], data_setup['ky']
psi_rms = data_setup['psi_rms']

if 'psi_hat' in data_setup.files:
    psi_hat = data_setup['psi_hat']
else:
    psi_hat = fft.fft2(data_setup['psi'], norm='forward')

# Save all in one file
print(f'Saving all in file {path_to_save_all}')
dict_to_save = {'M_hat': M_hat, 't': time,
                'x': xx, 'y': yy, 
                'kx': kx, 'ky': ky,
                'psi_hat': psi_hat, 'psi_rms': psi_rms}

if ('u0' in data_setup.files) and ('theta_u0' in data_setup.files):
    dict_to_save['u0'] = data_setup['u0']
    dict_to_save['theta_u0'] = data_setup['theta_u0']

np.savez_compressed(os.path.join(pathExperiment, path_to_save_all), **dict_to_save)
print("Done")
