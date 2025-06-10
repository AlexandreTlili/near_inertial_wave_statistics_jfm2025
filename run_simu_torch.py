import os

import torch
import torch.fft as fft

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tqdm import tqdm

# Parameters for the FFT
kwfft = {'norm': 'forward'}

######################################################
###################   Parameters   ###################
######################################################

# Device for parallelisation (or not)
device = torch.device("cuda:0") # For GPU
#device = torch.device("cpu")    # For CPU

# Resolution
resol = 2**10
dealias = True                  # If True: uses the 2/3 rule

# Definition of psi
path_psi_shape = "./psi.npz"   # Only the shape is used -> it will be normalized
psi_rms = 1.                   # Controls the amplitude of psi (psi_rms/h) -> Ro/Bu 

# Initial condition
restart = False
path_data_restart = None # './output_simu_0.npz'
path_M_initial = None #'./M_0.npz'  # If None, starts with uniform M0 = 1

# Parameters simulation
t_end = 100
dt = 5e-5
freq_save_field = 20_000
freq_save_plot = 1_000
freq_save_scalars = 100
freq_save_timeIntegral = 100_000 #None

# Path to save stuff
path_save_fields = "./fields0"                  # Folder
path_save_snapshots = "./snapshots0"            # Folder
path_save_energies = "./energies0.txt"          # File
path_save_correlations = "./correlations0.txt"  # File
path_save_setup = "./setup0.npz"                # File
path_psi_plot = "./psi.png"                     # File

# Parameters plots
max_KE = 2                    # Initial magnitude for |M|² plots -> is increased when |M|² > max_KE
max_PE = 0.1                  # Initial magnitude for |grad(M)|² plots -> is increased when |grad(M)|² > max_PE
min_PE = 1e-4
cmap = 'RdBu_r'
plotContours = False          # Contours of the background vorticity
maxResol = 1024               # Max resol for plots

#################################
###### CODE FOR SIMULATION ###### ###########################################################################
#################################

## Generate the domain and the frequencies

# Domain (real space)
domain_size = 2 * np.pi
domain_1d = np.linspace(0., domain_size, resol, endpoint=False)
x, y = np.meshgrid(domain_1d, domain_1d, indexing='xy')  # CPU for plots

# Fourier space
k_1d = fft.fftfreq(resol, d=1./resol, device=device)
mask_aliasing_1d = ~(3 * torch.abs(k_1d) < resol)
k_xx = k_1d.detach().clone().reshape((1, resol))    # Use sparse operators for less memory usage (makes a copy)
k_yy = k_1d.detach().clone().reshape((resol, 1))


def dealias_field(field_hat):
    """ Dealiases the field with the 2/3 rule """
    field_hat[mask_aliasing_1d,:] = 0.
    field_hat[:,mask_aliasing_1d] = 0.
    return field_hat

## Function to load field and crop the high wave-number
def load_and_crop(path_field, field='psi', normalize=False):
    '''Load the fft of the field and crops the fft to get the correct resolution
       Also send the result to the computation device (CPU or GPU)
    '''

    assert os.path.isfile(path_field), f"Wrong path for {field} data"
    assert field in ['psi', 'M'], "<field> should be 'M' or 'psi'"

    file_field = np.load(path_field)
    field_hat = file_field[field+'_hat']

    k_max, k_min = k_1d.max().item(), k_1d.min().item()

    mask_kx = (file_field['kx'][0,:] <= k_max) & (file_field['kx'][0,:] >= k_min)
    mask_ky = (file_field['ky'][:,0] <= k_max) & (file_field['ky'][:,0] >= k_min)

    field_hat = field_hat[:, mask_kx]   # Hmm maybe inversion here, but no pb since mask_kx and mask_ky identical
    field_hat = field_hat[mask_ky, :]

    if normalize:
        field_hat /= np.sqrt(np.sum(np.abs(field_hat)**2))  # It is a sum here because we are on Fourier space (Parseval with forward norm: mean_x |f|^2> = sum_k |f_hat|^2)

    # Send it to device
    field_hat = torch.tensor(field_hat, device=device)

    if dealias:
        field_hat = dealias_field(field_hat)

    return field_hat

## Load psi and normalize it
psi_hat = load_and_crop(path_psi_shape, field='psi', normalize=True) # On device
psi = fft.ifft2(psi_hat, **kwfft).real # On device


## Create time-independant matrices

# Derivation matrices (on device) -> all columns/lines for low memory usage
deriv_x = 1j * k_xx
deriv_y = 1j * k_yy
laplacian_xx = -k_xx**2     # Shape (1, resol)
laplacian_yy = -k_yy**2     # Shape (resol, 1)

# Psi derivatives (on device)
lapl_psi = fft.ifft2(laplacian_xx * psi_hat + laplacian_yy * psi_hat, **kwfft).real
dx_psi = fft.ifft2(deriv_x * psi_hat, **kwfft).real
dy_psi = fft.ifft2(deriv_y * psi_hat, **kwfft).real

# Get back psi/psi_hat on CPU
psi_cpu = psi.cpu()             # Keep psi on GPU because usefull when computing advection energy (can be modified)
psi_hat = psi_hat.cpu()         # Not needed on GPU
lapl_psi_cpu = lapl_psi.cpu()   # Usefull for plots with contours
mask_aliasing_1d_cpu = mask_aliasing_1d.cpu()


## Compute background field (usefull for computing correlations) -> all on device
vortPsi_local = lapl_psi
KEpsi_local = torch.abs(dx_psi)**2 + torch.abs(dy_psi)**2
potentialPsi_local = 0.5 * (vortPsi_local - psi_rms * KEpsi_local)
Utot_cpu = torch.sqrt(KEpsi_local).cpu()

vortPsi_mean = vortPsi_local.mean()
KEpsi_mean = KEpsi_local.mean()
potentialPsi_mean = potentialPsi_local.mean()

vortPsi_std = vortPsi_local.std()
KEpsi_std = KEpsi_local.std()
potentialPsi_std = potentialPsi_local.std()


## Plot psi and the vorticity

if resol > maxResol:
    skipSome = resol // maxResol
else:
    skipSome = 1

fig, ax = plt.subplots(figsize=(14,4), ncols=3, tight_layout=True)

mag1 = torch.max(torch.abs(psi_cpu)).item()
mag2 = torch.max(Utot_cpu).item()
mag3 = torch.max(torch.abs(lapl_psi_cpu)).item()

p1 = ax[0].pcolormesh(x[::skipSome,::skipSome], y[::skipSome,::skipSome], psi_cpu[::skipSome,::skipSome], cmap=cmap, vmin=-mag1, vmax=mag1)
p2 = ax[1].pcolormesh(x[::skipSome,::skipSome], y[::skipSome,::skipSome], Utot_cpu[::skipSome,::skipSome], cmap='viridis', vmin=0, vmax=mag2)
p3 = ax[2].pcolormesh(x[::skipSome,::skipSome], y[::skipSome,::skipSome], lapl_psi_cpu[::skipSome,::skipSome], cmap=cmap, vmin=-mag3, vmax=mag3)

fig.colorbar(p1, ax=ax[0])
fig.colorbar(p2, ax=ax[1])
fig.colorbar(p3, ax=ax[2])

# Plot contours
ax[0].contour(x[::skipSome,::skipSome], y[::skipSome,::skipSome], psi_cpu[::skipSome,::skipSome], colors='k')
if plotContours:
    ax[1].contour(x[::skipSome,::skipSome], y[::skipSome,::skipSome], Utot_cpu[::skipSome,::skipSome], colors='k', levels=[1e-1, 1, 2, 3, 4])
    ax[2].contour(x[::skipSome,::skipSome], y[::skipSome,::skipSome], lapl_psi_cpu[::skipSome,::skipSome], colors='k')

ax[0].set_title('Stream-function $\\psi$')
ax[1].set_title('Total velocity $|\\nabla \\psi|$')
ax[2].set_title('Vorticity $\\Delta \\psi$')

for axi in ax:
    axi.set_xlabel('$x$')
    axi.set_ylabel('$y$')
    axi.set_aspect(1)

plt.savefig(path_psi_plot, dpi=200)
plt.close()

## Save the parameters
np.savez_compressed(path_save_setup,
                    x=x, y=y,
                    kx=k_xx.cpu().numpy(), ky=k_yy.cpu().numpy(), 
                    psi=psi_cpu.numpy(), lapl_psi=lapl_psi_cpu.numpy(),
                    psi_hat=psi_hat.numpy(),
                    psi_rms=psi_rms)


## Define operators of the equation

# Operator on M_hat giving dM_hat/dt (YBJ) -> all computed on device
def get_time_deriv_YBJ(M_hat, dealias=True):
    """ Returns the instantaneous rate of change associated to M_hat
    """

    # Dealiasing with 2/3 rule
    if dealias:
        M_hat = dealias_field(M_hat)

    # Computes the advection term (on device) -> real space
    dx_M = fft.ifft2(deriv_x * M_hat, **kwfft)
    dy_M = fft.ifft2(deriv_y * M_hat, **kwfft)
    advection_refraction = dx_psi * dy_M - dy_psi * dx_M

    dx_M = None; dy_M = None; # For less memory usage

    # Refraction term -> real space
    advection_refraction += 1j/2. * lapl_psi * fft.ifft2(M_hat, **kwfft)

    # Dispersion term, with parameter psi_rms = Ro/Bu
    # More operations but less memory usage
    laplacian_M_hat = laplacian_xx * M_hat
    laplacian_M_hat += laplacian_yy * M_hat
    
    return 1j/2. * laplacian_M_hat / psi_rms - fft.fft2(advection_refraction, **kwfft)


def RK4_step_YBJ(dt, M_hat, dealias=True):
    """ Computes the next time-step using RK4 (uses 4 new arrays and modifies M_hat)
        -> unused anymore but easier to understand
    """

    k1 = get_time_deriv_YBJ(M_hat, dealias)
    k2 = get_time_deriv_YBJ(M_hat + 0.5 * dt * k1, dealias)
    k3 = get_time_deriv_YBJ(M_hat + 0.5 * dt * k2, dealias)
    k4 = get_time_deriv_YBJ(M_hat + dt * k3, dealias)

    M_hat += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.


def RK4_step_YBJ_loop(dt, M_hat, dealias=True):
    """ Computes the next time-step using RK4 (uses 2 new arrays and modifies M_hat)
    """

    final_slope = torch.zeros_like(M_hat, device=device)
    current_slope = torch.zeros_like(M_hat, device=device)

    steps = [0., 0.5, 0.5, 1.]
    weights = [1./6, 2./6, 2./6, 1./6]

    for step, weight in zip(steps, weights):
        current_slope = get_time_deriv_YBJ(M_hat + step * dt * current_slope, dealias) 
        final_slope += weight * current_slope

    M_hat += dt * final_slope


def plot_KE_PE_wave(M_hat, path, time=None):
    """ Function used to plot and save the kinetic and potential energy fields
    """

    global max_KE
    global max_PE

    # Compute |M|² and |grad(M)|²
    KE_wave = torch.abs(fft.ifft2(M_hat, **kwfft))**2
    PE_wave = torch.abs(fft.ifft2(M_hat * deriv_x, **kwfft))**2 + torch.abs(fft.ifft2(M_hat * deriv_y, **kwfft))**2

    # Update cmap and norm
    max_KE = max(max_KE, torch.max(KE_wave).item())
    max_PE = max(max_PE, torch.max(PE_wave).item())

    norm_KE = LogNorm(vmin=1./max_KE, vmax=max_KE)
    norm_PE = LogNorm(vmin=min_PE, vmax=max_PE)

    fig, ax = plt.subplots(figsize=(10,4), ncols=2, tight_layout=True)

    p1 = ax[0].pcolormesh(x[::skipSome,::skipSome], y[::skipSome,::skipSome], KE_wave.cpu()[::skipSome,::skipSome], cmap=cmap, norm=norm_KE)
    p2 = ax[1].pcolormesh(x[::skipSome,::skipSome], y[::skipSome,::skipSome], PE_wave.cpu()[::skipSome,::skipSome], cmap=cmap, norm=norm_PE)

    if plotContours:
        for axi in ax:
            axi.contour(x[::skipSome,::skipSome], y[::skipSome,::skipSome], lapl_psi_cpu[::skipSome,::skipSome], colors='k', alpha=0.8)

    fig.colorbar(p1, ax=ax[0], label="$|M|^2$")
    fig.colorbar(p2, ax=ax[1], label="$|\\nabla M|^2$")

    for axi in ax:
        axi.set_xlabel('$x$')
        axi.set_ylabel('$y$')
        axi.set_aspect(1)

    if time is None:
        ax[0].set_title(f'Kinetic energy $|M|^2$')
        ax[1].set_title(f'Potential energy $|\\nabla M|^2$')
    else:
        ax[0].set_title(f'Kinetic energy $|M|^2$ ($Ro_{{\\Psi}} \\times (t f_0) = {time:.2f}$)')
        ax[1].set_title(f'Potential energy $|\\nabla M|^2$ ($Ro_{{\\Psi}} \\times (t f_0) = {time:.2f}$)')

    plt.savefig(path, dpi=200)
    plt.close()


def write_field(path, M_hat, time):
    """ Function used to write the fields in the file given in path
    """
    
    if os.path.isfile(path):
        path = path + '_1'

    Mhat_cpu = M_hat.cpu()

    if dealias:
        # Only save non-zero values
        Mhat_cpu = Mhat_cpu[~mask_aliasing_1d_cpu][:,~mask_aliasing_1d_cpu]
    
    np.savez_compressed(path, M_hat=Mhat_cpu, t=[time], resol=[resol], dealias=[dealias])


def compute_energies(M_hat, verbose=False):
    """ Function to compute the total energies (KE, advection, dispersion, refraction) of the wave field
        -> energies defined by space averaging instead of space integration !
        -> Also: should divide everything by psi_rms to be consistent with the hamiltonian (see Danioux) 
                 but kept this way for consistency with older simulations
    """

    M = fft.ifft2(M_hat, **kwfft)
    
    dx_M = fft.ifft2(deriv_x * M_hat, **kwfft)
    dy_M = fft.ifft2(deriv_y * M_hat, **kwfft)
    jac_Ms_M = torch.conj(dx_M) * dy_M - torch.conj(dy_M) * dx_M

    kineticEnergy = torch.mean(torch.abs(M)**2) # Real space
    advectionEnergy  = torch.real(1j * torch.mean(psi * jac_Ms_M)) # Real space
    dispersionEnergy = 0.5 * torch.mean(torch.abs(dx_M)**2 + torch.abs(dy_M)**2) / psi_rms # Real space
    refractionEnergy = 0.5 * torch.mean(lapl_psi * torch.abs(M)**2) # Real space

    if verbose:
        print(f'KE: 1+{kineticEnergy.item()-1:.2e} - total: {advectionEnergy.item() + dispersionEnergy.item() + refractionEnergy.item():.2e}')

    return kineticEnergy.item(), advectionEnergy.item(), dispersionEnergy.item(), refractionEnergy.item()

def compute_correlation(M_hat):
    """ Function used to compute the normalized cross-correlations between the flow and the background
    """
    
    # Wave kinetic enery
    KEwave_local = torch.abs(fft.ifft2(M_hat, **kwfft))**2
    KEwave_mean = KEwave_local.mean()
    KEwave_std = KEwave_local.std() + 1e-15     # To prevent warnings when dividing by zero

    # Wave potential energy
    PEwave_local = torch.abs(fft.ifft2(M_hat * deriv_x, **kwfft))**2 + torch.abs(fft.ifft2(M_hat * deriv_y, **kwfft))**2
    PEwave_mean = PEwave_local.mean()
    PEwave_std = PEwave_local.std() + 1e-15     # To prevent warnings when dividing by zero

    # Normalized correlations C(a,b) = E((a-E(a))(b-E(b))) / sqrt(E((a-E(a))^2)) * sqrt(E((a-E(a))^2))
    # Note: C(a,b) = (E(ab) - E(a)E(b))/ std(a)std(b)

    # Correlations with wave KE
    corrKE_vortPsi = ((KEwave_local * vortPsi_local).mean(axis=(-2,-1)) - KEwave_mean * vortPsi_mean) / (KEwave_std * vortPsi_std)
    corrKE_KEpsi = ((KEwave_local * KEpsi_local).mean(axis=(-2,-1)) - KEwave_mean * KEpsi_mean) / (KEwave_std * KEpsi_std)
    corrKE_potPsi = ((KEwave_local * potentialPsi_local).mean(axis=(-2,-1)) - KEwave_mean * potentialPsi_mean) / (KEwave_std * potentialPsi_std)

    # Correlations with wave PE
    corrPE_vortPsi = ((PEwave_local * vortPsi_local).mean(axis=(-2,-1)) - PEwave_mean * vortPsi_mean) / (PEwave_std * vortPsi_std)
    corrPE_KEpsi = ((PEwave_local * KEpsi_local).mean(axis=(-2,-1)) - PEwave_mean * KEpsi_mean) / (PEwave_std * KEpsi_std)
    corrPE_potPsi = ((PEwave_local * potentialPsi_local).mean(axis=(-2,-1)) - PEwave_mean * potentialPsi_mean) / (PEwave_std * potentialPsi_std)

    return corrKE_vortPsi.item(), corrKE_KEpsi.item(), corrKE_potPsi.item(), corrPE_vortPsi.item(), corrPE_KEpsi.item(), corrPE_potPsi.item()


def save_energies(path, energies, iteration, time):
    """ Function used to append the energies (KE, advection, dispersion, refraction) to the energy file
    """

    header = 'iteration time kineticEnergy advectionEnergy dispersionEnergy refractionEnergy\n'
    if not os.path.isfile(path):
        write_header = True
    else:
        write_header = False
    
    with open(path, "a") as file:
        if write_header: 
            file.write(header)
        file.write(' '.join([f'{iteration}', f'{time}'] + 
                            [f'{energy:.15e}' for energy in energies]) + '\n')

def save_correlations(path, correlations, iteration, time):
    """ Function used tto append the normalized cross-correlations between the flow and the background
    """

    header = 'iteration time KE_vortPsi KE_KEpsi KE_potPsi PE_vortPsi PE_KEpsi PE_potPsi\n'
    if not os.path.isfile(path):
        write_header = True
    else:
        write_header = False
    
    with open(path, "a") as file:
        if write_header: 
            file.write(header)
        file.write(' '.join([f'{iteration}', f'{time}'] + 
                            [f'{correlation:.8f}' for correlation in correlations]) + '\n') 

def save_timeIntegral(path, fields_tIntegrated, time, t_start):
    """ Function used to write the time-integrated quadratic data
    """
    
    if os.path.isfile(path):
        path = path + '_1'

    fields_tIntegrated_cpu = fields_tIntegrated.cpu()
    
    np.savez_compressed(path, waveKE=fields_tIntegrated_cpu[0], wavePE=fields_tIntegrated_cpu[1],
                        waveStokes_x=fields_tIntegrated_cpu[2], waveStokes_y=fields_tIntegrated_cpu[3], 
                        t_bounds=[t_start, time], resol=[resol])


def solve_dt_fixed(M_hat_0, dt, t_end, t_restart=0.):
    """ Function used to solve the IVP with RK4 (constant dt) from a given initial condition.
    """

    # Number of time steps
    n_steps = int((t_end - t_restart) / dt)

    # Initialize array of saved fields
    M_hat = M_hat_0
    time = t_restart
    if freq_save_timeIntegral is not None:
        fields_tIntegral = torch.zeros((4, resol, resol), device=device)

    # Save first snapshot
    plot_KE_PE_wave(M_hat, os.path.join(path_save_snapshots, 'KE_PE_0.png'), time=time)
    write_field(os.path.join(path_save_fields, f'field_0'), M_hat, time)

    # Time loop
    for i in tqdm(range(n_steps)):

        # Advance one time step
        RK4_step_YBJ_loop(dt, M_hat, dealias)
        time = t_restart + (i + 1) * dt

        if torch.any(torch.isnan(M_hat)):
            raise ValueError(f"NaN values in M_hat at step {i}")

        # Write the scalar metrics
        if (i+1) % freq_save_scalars == 0:
            energies = compute_energies(M_hat, printVal=False)
            save_energies(path_save_energies, energies, i+1, time)

            correlations = compute_correlation(M_hat)
            save_correlations(path_save_correlations, correlations, i+1, time)

        # Update time-integral quadratic
        if freq_save_timeIntegral is not None:
            M = fft.ifft2(M_hat, **kwfft)
            dxM = fft.ifft2(deriv_x * M_hat, **kwfft)
            dyM = fft.ifft2(deriv_y * M_hat, **kwfft)

            fields_tIntegral[0] += dt * torch.abs(M)**2    # Wave KE
            fields_tIntegral[1] += dt * (torch.abs(dxM)**2 + torch.abs(dyM)**2)       # Wave PE
            fields_tIntegral[2] += dt * 0.25 * (2 * torch.imag(M.conj() * dxM) + 2 * torch.real(M.conj() * dyM))
            fields_tIntegral[3] += dt * 0.25 * (2 * torch.imag(M.conj() * dyM) - 2 * torch.real(M.conj() * dxM))
            
            # Save it if needed
            if (i+1) % freq_save_timeIntegral == 0:
                save_timeIntegral(os.path.join(path_save_fields, f'tIntegrated_{i+1}'), fields_tIntegral, time, t_restart)


        # Write the field data
        if (i+1) % freq_save_field == 0:
            write_field(os.path.join(path_save_fields, f'field_{i+1}'), M_hat, time)

        # Save some snapshots KE/PE
        if (i+1) % freq_save_plot == 0:
            plot_KE_PE_wave(M_hat, os.path.join(path_save_snapshots, f'KE_PE_{i+1}.png'), time=time)

    # Compute the last step if needed
    i = n_steps
    if time < t_end:
        dt_last = t_end - time
        RK4_step_YBJ_loop(dt_last, M_hat, dealias)
        time = t_end

        energies = compute_energies(M_hat, printVal=False)
        save_energies(path_save_energies, energies, i+1, time)

        correlations = compute_correlation(M_hat)
        save_correlations(path_save_correlations, correlations, i+1, time)

        write_field(os.path.join(path_save_fields, f'field_{i+1}'), M_hat, time)

        plot_KE_PE_wave(M_hat, os.path.join(path_save_snapshots, f'KE_PE_{i+1}.png'), time=time)

    return 

def pad_with_zeros_fft(field_hat, dtype='complex'):
    """Pad field_hat of a dealiased field with zeros to extend back to aliased size"""

    assert field_hat.ndim == 2, "Only for a single flow field."

    mask_aliasing_1d_numpy = mask_aliasing_1d_cpu.numpy()
    assert np.sum(~mask_aliasing_1d_numpy).item() == field_hat.shape[-1]    # Check that the missing data correspond to aliased data

    field_hat_extended = np.zeros((resol, resol), dtype=dtype)

    j_dealiased = 0
    for j in range(resol):

        # If should be zero, skip  
        if mask_aliasing_1d[j]:
            continue

        # Else, copy the corresponding column 
        field_hat_extended[~mask_aliasing_1d_numpy,j] = field_hat[:,j_dealiased]
        j_dealiased += 1 

    return field_hat_extended

####################################
###### RUNNING THE SIMULATION ###### #################################################################
####################################

# Create directories for saving data
if not os.path.isdir(path_save_snapshots):
    os.makedirs(path_save_snapshots)
if not os.path.isdir(path_save_fields):
    os.makedirs(path_save_fields)

# Load initial condition
if restart: 
    if not (path_M_initial is None):
        raise ValueError("Conflict between restart and initial condition: should set <path_M_initial> to None")
    
    print(f"Reading IC from restart at {path_data_restart}")
    data_restart = np.load(path_data_restart)
    M_hat_0 = data_restart['M_hat']
    t_restart = data_restart['t'][-1]

    if M_hat_0.ndim > 2:
        # Then its a list of fields, only take last one
        M_hat_0 = M_hat_0[-1]

    if M_hat_0.shape[-1] != resol:
        # Zero padding for dealiased fieds
        M_hat_0 = pad_with_zeros_fft(M_hat_0, dtype='complex')    

    # Send array to device
    M_hat_0 = torch.tensor(M_hat_0, device=device)

elif not (path_M_initial is None):
    # Use the given initial condition
    print(f"Reading IC from given at {path_M_initial}")
    M_hat_0 = load_and_crop(path_M_initial, field='M', normalize=True)
    t_restart = 0.

else:
    # Start from uniform initial condition M0 = 1
    print("Starting from uniform IC.")
    M_0 = torch.full_like(psi, 1, device=device)
    M_hat_0 = fft.fft2(M_0, **kwfft)
    t_restart = 0.
    del M_0

## Run the simulation
solve_dt_fixed(M_hat_0, dt, t_end, t_restart=t_restart)

print("Simulation ended successfully")
