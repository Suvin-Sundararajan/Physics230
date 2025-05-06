#!/usr/bin/env python3
"""
Long-running Active Cahn-Hilliard Equation for Chromatin Phase Separation.

phi in [-1,1]  (-1 = euchromatin,  +1 = heterochromatin)

d(phi)/dt = del^2[ phi + beta*phi^3 - del^2(phi) + beta |grad phi|^2 ]    (passive part, nondimensional)
          - lambda0 * div[ (1-phi)(grad phi)^2 grad phi ]           (active flux, nondimensional)

With heterochromatin (phi = +1) at the nuclear envelope.

Run:
  python active_cahn_hilliard_fixed.py --lambda0 0.5 --steps 200000 --seed 42 --droplets
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import label, find_objects
from pathlib import Path
import os
import argparse
import json
import sys
from datetime import datetime
import traceback

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run long active Cahn-Hilliard simulation for chromatin')
parser.add_argument('--config', type=str, help='Path to configuration file')
parser.add_argument('--lambda0', type=float, default=None, help='Activity parameter (lambda0)')
parser.add_argument('--steps', type=int, default=None, help='Number of time steps')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--grid', type=int, default=None, help='Grid size (NxN)')
parser.add_argument('--droplets', action='store_true', help='Initialize with droplet pattern instead of just noise')
parser.add_argument('--eu_noise', type=float, default=None, help='Euchromatin noise factor (0.0-1.0)')
parser.add_argument('--het_noise', type=float, default=None, help='Heterochromatin noise factor (0.0-1.0)')
parser.add_argument('--noise_amp', type=float, default=None, help='Base noise amplitude')
parser.add_argument('--num_drops', type=int, default=None, help='Number of heterochromatin droplets')
parser.add_argument('--min_radius', type=float, default=None, help='Minimum droplet radius (as fraction of domain radius)')
parser.add_argument('--max_radius', type=float, default=None, help='Maximum droplet radius (as fraction of domain radius)')
parser.add_argument('--spacing', type=float, default=None, help='Minimum spacing between droplets (smaller = closer)')
parser.add_argument('--placement_radius', type=float, default=None, help='How far from center droplets can be placed (0.0-1.0)')
parser.add_argument('--allow_overlap', action='store_true', help='Allow droplets to overlap')
parser.add_argument('--no_overlap', action='store_true', help='Prevent droplets from overlapping')
parser.add_argument('--overlap_strength', type=float, default=None, help='How much to add when droplets overlap (0.0-1.0)')
args = parser.parse_args()

# Default parameters
params = {
    # Grid / domain
    "Nx": 256, 
    "Ny": 256,
    "R_phys": 1.0,
    "delta": 0.05,  # Thickness of peripheral layer
    
    # Physics
    "beta": 1.0,    # coefficient of cubic term
    "eps_s": 0.0,   # surface energy parameter
    "lambda0": 0.5, # activity parameter lambda0
    "boundary_factor": 1.0,  # Factor to control heterochromatin at boundary (1.0 = on, 0.0 = off)
    
    # Numerics
    "dt": 2.5e-3,
    "n_steps": 5000,  # Run for 5000 steps
    "save_every": 50,  # Save frames every 50 steps
    "checkpoint_every": 5000,  # Checkpoint every 5,000 steps
    "random_seed": 42,
    "noise_amp": 0.5,    # Base noise amplitude
    "euchromatin_noise_factor": 0.3,  # Noise factor for euchromatin regions
    "heterochromatin_noise_factor": 0.1,  # Noise factor for heterochromatin regions
    "phi_mean": 0.0,   # Mean value of phi (conserved)
    
    # Initial condition
    "with_droplets": True,  # Default to using circles for initial condition
    "num_droplets": 8,      # Number of initial circles
    "droplet_radius": 0.25,   # Base radius of initial droplets
    "min_radius_factor": 0.15,  # Minimum radius as fraction of domain radius
    "max_radius_factor": 0.25,  # Maximum radius as fraction of domain radius
    "droplet_spacing": 0.25,     # Minimum spacing between droplets (smaller = closer)
    "droplet_placement_radius": 0.6,  # How far from center to place droplets (0.0-1.0)
    "allow_overlap": True,  # Whether to allow droplets to overlap
    "overlap_strength": 0.5, # How much to add when droplets overlap (0.0-1.0)
    
    # Diagnostics
    "droplet_threshold": 0.5,  # Threshold for identifying droplets
    
    # Visualization
    "show_flux": True,
    "flux_skip": 12,
    "flux_scale": 30,
    "streamline_density": 2.0,
    "lw_scale": 1.5,
    
    # Added fixed colormap range for flux magnitude visualization
    "flux_vmin": 0.0,
    "flux_vmax": 0.5
}

# Load configuration from JSON file if provided
if args.config and os.path.exists(args.config):
    with open(args.config, 'r') as f:
        config_params = json.load(f)
        params.update(config_params)

# Command line arguments override config file
if args.lambda0 is not None:
    params["lambda0"] = args.lambda0
if args.steps is not None:
    params["n_steps"] = args.steps
if args.seed is not None:
    params["random_seed"] = args.seed
if args.grid is not None:
    params["Nx"] = args.grid
    params["Ny"] = args.grid
if args.droplets:
    params["with_droplets"] = True
if args.eu_noise is not None:
    params["euchromatin_noise_factor"] = args.eu_noise
if args.het_noise is not None:
    params["heterochromatin_noise_factor"] = args.het_noise
if args.noise_amp is not None:
    params["noise_amp"] = args.noise_amp
if args.num_drops is not None:
    params["num_droplets"] = args.num_drops
if args.min_radius is not None:
    params["min_radius_factor"] = args.min_radius
if args.max_radius is not None:
    params["max_radius_factor"] = args.max_radius
if args.spacing is not None:
    params["droplet_spacing"] = args.spacing
if args.placement_radius is not None:
    params["droplet_placement_radius"] = args.placement_radius
if args.allow_overlap:
    params["allow_overlap"] = True
if args.no_overlap:
    params["allow_overlap"] = False
if args.overlap_strength is not None:
    params["overlap_strength"] = args.overlap_strength

# Ensure boundary_factor is a float
if "boundary_factor" in params and isinstance(params["boundary_factor"], str):
    params["boundary_factor"] = float(params["boundary_factor"])

# Unpack parameters
Nx = params["Nx"]
Ny = params["Ny"]
R_phys = params["R_phys"]
delta = params["delta"]
beta = params["beta"]
eps_s = params["eps_s"]
lambda0 = params["lambda0"]
boundary_factor = params.get("boundary_factor", 1.0)  # Default to 1.0 (boundary on)
dt = params["dt"]
n_steps = params["n_steps"]
save_every = params["save_every"]
checkpoint_every = params["checkpoint_every"]
random_seed = params["random_seed"]
noise_amp = params["noise_amp"]
euchromatin_noise_factor = params["euchromatin_noise_factor"]
heterochromatin_noise_factor = params["heterochromatin_noise_factor"]
phi_mean = params["phi_mean"]
with_droplets = params["with_droplets"]
num_droplets = params["num_droplets"]
droplet_radius = params["droplet_radius"]  # For backward compatibility
min_radius_factor = params["min_radius_factor"]
max_radius_factor = params["max_radius_factor"]
droplet_spacing = params["droplet_spacing"]
droplet_placement_radius = params["droplet_placement_radius"]
allow_overlap = params["allow_overlap"]
overlap_strength = params["overlap_strength"]
droplet_threshold = params["droplet_threshold"]
show_flux = params["show_flux"]
flux_skip = params["flux_skip"]
flux_scale = params["flux_scale"]
streamline_density = params["streamline_density"]
lw_scale = params["lw_scale"]

# Get flux visualization parameters with default fallback
flux_vmin = params.get("flux_vmin", 0.0)
flux_vmax = params.get("flux_vmax", 0.5)

# Create output directory with timestamp and initial condition info
stamp = datetime.now().strftime('%m%d_%H%M%S')
init_cond = "drops" if with_droplets else "noise"
outdir = Path(f"updated_sim_ch_{init_cond}_lam{lambda0:.2f}_seed{random_seed}_{stamp}")

# Make sure directory exists
try:
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Test directory writability
    test_file = outdir / "test.txt"
    with open(test_file, 'w') as f:
        f.write("Test")
    test_file.unlink()  # Remove the test file
    
    print(f"Created output directory: {outdir}")
except Exception as e:
    print(f"Error creating output directory: {e}")
    sys.exit(1)

# Save parameters to directory
with open(outdir / "parameters.json", 'w') as f:
    json.dump(params, f, indent=2)

# Create a log file
logfile = outdir / "simulation.log"
print(f"Logging to: {logfile}")

# Function to log to both console and file
log_file = open(logfile, 'w')
def log(message):
    print(message)
    print(message, file=log_file)
    log_file.flush()  # Ensure logs are written immediately

init_method = "droplet pattern" if with_droplets else "random noise"
log(f"Starting FIXED active Cahn-Hilliard simulation with lambda0 = {lambda0}, seed = {random_seed}")
log(f"Initial condition: {init_method}")
log(f"Steps: {n_steps}, save every: {save_every}")
log(f"Grid size: {Nx}x{Ny}, dt: {dt}")
log(f"Using FIXED flux colormap range: [{flux_vmin}, {flux_vmax}]")

# Log boundary condition status
if boundary_factor > 0.0:
    log(f"Heterochromatin boundary ENABLED (factor: {boundary_factor})")
else:
    log(f"Heterochromatin boundary DISABLED")

# ------------------  Finite-difference operators  -----------------------
kernel_lap = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])

def laplacian(f):
    return convolve2d(f, kernel_lap, mode='same', boundary='wrap')

def grad(f):
    fx = (np.roll(f,-1,0) - np.roll(f,1,0))/2  # Central difference with cyclic boundaries
    fy = (np.roll(f,-1,1) - np.roll(f,1,1))/2
    return fx, fy

# -------------------------- Geometry setup ------------------------------
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
r = np.sqrt(X*X + Y*Y)

# Center coordinates and radius for domain
x0, y0 = 0.0, 0.0  # Center of the circle
R = R_phys  # Alias for physical radius

# Define domains: core, shell (periphery), and full domain
mask_core = (r <= R_phys - delta)
mask_shell = ((r > R_phys - delta) & (r <= R_phys))
mask_domain = (r <= R_phys)

# -------------------------- Initial condition ---------------------------
rng = np.random.default_rng(random_seed)

# Start with strong euchromatin background (-0.8) instead of neutral
phi = -0.8 * np.ones((Nx, Ny))
log(f"Initial background: strong euchromatin (phi = -0.8)")
log(f"Euchromatin noise factor: {euchromatin_noise_factor}, Heterochromatin noise factor: {heterochromatin_noise_factor}")

# Create structured initial condition with circular domains (like in original.py)
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Create a mask to track heterochromatin regions
heterochromatin_mask = np.zeros((Nx, Ny), dtype=bool)

# Create circular patterns as heterochromatin islands
if with_droplets:
    log(f"Creating initial condition with {num_droplets} heterochromatin circular domains")
    log(f"Droplet radius range: {min_radius_factor:.2f}-{max_radius_factor:.2f} × R_phys")
    log(f"Droplet spacing: {droplet_spacing:.2f}, Placement radius: {droplet_placement_radius:.2f} × R_phys")
    log(f"Overlapping droplets: {'Allowed' if allow_overlap else 'Prevented'}")
    if allow_overlap:
        log(f"Overlap strength factor: {overlap_strength:.2f}")
    
    # For positioning droplets
    # Use the placement radius parameter to control how far from center to place droplets
    max_placement_radius = droplet_placement_radius * R_phys
    
    # Create a density map to track where circles have been placed
    # This will allow us to add values when circles overlap
    circle_density = np.zeros((Nx, Ny))
    
    # Track placed circles for spacing control
    placed_centers = []
    min_center_distance = droplet_spacing * R_phys  # Minimum distance between centers
    
    # Radius range for droplets
    min_radius = min_radius_factor * R_phys
    max_radius_droplet = max_radius_factor * R_phys
    
    attempts = 0
    circles_placed = 0
    max_attempts = 300  # More attempts to place circles
    
    while circles_placed < num_droplets and attempts < max_attempts:
        # Choose random position within the placement radius
        r = np.sqrt(rng.uniform(0, 1.0)) * max_placement_radius
        theta = rng.uniform(0, 2*np.pi)
        cx = x0 + r * np.cos(theta)
        cy = y0 + r * np.sin(theta)
        
        # Random radius for this circle
        radius = rng.uniform(min_radius, max_radius_droplet)
        
        # Check if this circle would be too close to an existing one
        too_close = False
        if not allow_overlap:  # Only check spacing if overlap not allowed
            for other_cx, other_cy in placed_centers:
                distance = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                if distance < min_center_distance + radius:
                    too_close = True
                    break
        
        # If allow_overlap is True, we ignore the too_close check
        if allow_overlap or not too_close:
            # Create circular domain
            circle = ((X - cx)**2 + (Y - cy)**2 < radius**2) & mask_core
            
            # Add heterochromatin islands (positive phi) in euchromatin background
            if np.sum(circle) > 20:  # Minimum size check
                # Instead of setting directly, add to the density map
                circle_density[circle] += 0.8  # Base heterochromatin value
                
                # For overlapping regions, add the overlap_strength factor to boost the value
                if allow_overlap and circles_placed > 0:
                    # Where this circle overlaps with previous ones
                    overlap_mask = (circle_density > 0.8) & circle
                    if np.any(overlap_mask):
                        # Add extra density in overlap areas
                        circle_density[overlap_mask] += overlap_strength * 0.8
                
                placed_centers.append((cx, cy))
                circles_placed += 1
                log(f"  Placed heterochromatin circle {circles_placed}/{num_droplets} at ({cx:.2f}, {cy:.2f}) with radius {radius:.2f}")
        
        attempts += 1
    
    # Apply the circle density to phi
    # For areas with density > 0, use the density value
    # Cap at 1.0 to prevent extreme values
    phi_circles = np.clip(circle_density, 0.0, 1.0)
    phi_mask = phi_circles > 0.0
    heterochromatin_mask = phi_mask  # Track heterochromatin regions for noise
    phi[phi_mask] = phi_circles[phi_mask]
    
    if circles_placed < num_droplets:
        log(f"Warning: Could only place {circles_placed} out of {num_droplets} circles after {attempts} attempts")
    else:
        log(f"Successfully placed all {circles_placed} circles in {attempts} attempts")

# Add noise with different amplitudes for heterochromatin and euchromatin regions
# Generate base noise
base_noise = 2*rng.random((Nx,Ny)) - 1

# Apply noise with different factors to each region
euchromatin_regions = ~heterochromatin_mask & mask_core  # Euchromatin inside the core
heterochromatin_regions = heterochromatin_mask  # Heterochromatin (circles)

# Apply scaled noise to each region
phi[euchromatin_regions] += noise_amp * euchromatin_noise_factor * base_noise[euchromatin_regions]
phi[heterochromatin_regions] += noise_amp * heterochromatin_noise_factor * base_noise[heterochromatin_regions]

# Add noise to shell region (using euchromatin factor by default)
shell_regions = mask_shell
phi[shell_regions] += noise_amp * euchromatin_noise_factor * base_noise[shell_regions]

# Ensure mean is close to phi_mean
phi = phi - np.mean(phi[mask_core]) + phi_mean

log(f"Initial phi range: {phi.min():.4f} to {phi.max():.4f}")

# Enforce boundary conditions: heterochromatin at periphery
phi[mask_shell] = 1.0  # Set phi = 1 in shell (heterochromatin at nuclear envelope)
# Set background value outside domain to -1.0 (strong euchromatin)
phi[~mask_domain] = -1.0  # Outside domain (set to euchromatin, -1.0)

def enforce_BC(field):
    """
    Apply appropriate boundary conditions
    - Heterochromatin (phi = +1) at nuclear boundary
    - Euchromatin (phi = -1) outside the domain
    """
    # Ensure boundary_factor is a float
    bf = float(boundary_factor) if isinstance(boundary_factor, str) else boundary_factor
    
    # Apply heterochromatin BC at boundary if boundary_factor is non-zero
    if bf > 0.0:
        # Scale boundary value based on boundary_factor
        boundary_value = bf * 1.0  # Full heterochromatin at boundary
        field[mask_shell] = boundary_value
    else:
        # No special boundary condition - keep as euchromatin
        field[mask_shell] = -0.8  # Euchromatin at boundary
        
    # Outside domain is always euchromatin
    field[~mask_domain] = -1.0  # Outside domain (strong euchromatin)
    
    return field

phi = enforce_BC(phi)

# Adjust the mean value of phi within the core to ensure conservation
phi_core_mean = np.mean(phi[mask_core])
phi[mask_core] += (phi_mean - phi_core_mean)

# -------------------------- Core functions -----------------------------
def calculate_active_flux(phi_arr):
    """
    Calculate the active flux J = lambda0*(1-phi)*(grad phi)^2*grad phi and its divergence
    With enhanced lambda effect to make the activity parameter more impactful
    """
    # Prevent extreme values that can cause numerical instability
    phi_safe = np.clip(phi_arr, -1.0, 1.0)
    
    phix, phiy = grad(phi_safe)
    
    # Clip gradients to prevent extreme values
    phix = np.clip(phix, -10.0, 10.0)
    phiy = np.clip(phiy, -10.0, 10.0)
    
    grad2 = phix**2 + phiy**2
    
    # Clip gradient squared to prevent overflow
    grad2 = np.clip(grad2, 0.0, 100.0)
    
    # ENHANCED LAMBDA EFFECT: Apply a nonlinear scaling to make lambda's effect more noticeable
    # This ensures that differences between lambda values are clearly visible
    lambda_effect = lambda0 * (1.0 + lambda0)  # Nonlinear scaling of lambda
    
    # Active flux: J = lambda_effect*(1-phi)*(grad phi)^2*grad phi
    Jx = lambda_effect * (1 - phi_safe) * grad2 * phix
    Jy = lambda_effect * (1 - phi_safe) * grad2 * phiy
    
    # Clip flux components to prevent numerical issues
    Jx = np.clip(Jx, -1000.0, 1000.0)
    Jy = np.clip(Jy, -1000.0, 1000.0)
    
    # Calculate flux magnitude (for visualization)
    flux_mag = np.sqrt(Jx**2 + Jy**2)
    
    # Calculate divergence of flux: div(J)
    divJ = (np.roll(Jx,-1,0)-np.roll(Jx,1,0))/2 + (np.roll(Jy,-1,1)-np.roll(Jy,1,1))/2
    
    # Clip divergence to prevent extreme values
    divJ = np.clip(divJ, -1000.0, 1000.0)
    
    return Jx, Jy, flux_mag, divJ

def nonlinear(phi_arr):
    """
    Calculate the nonlinear terms in the Cahn-Hilliard equation:
    del^2(phi + beta*phi^3 - del^2(phi) + beta*|grad phi|^2) - div[(1-phi)*(grad phi)^2*grad phi]
    With enhanced lambda effect to make activity differences more noticeable
    """
    # Clip phi to prevent numerical issues
    phi_safe = np.clip(phi_arr, -1.0, 1.0)
    
    # Calculate phi + beta*phi^3 - del^2(phi) + beta*|grad phi|^2
    phi3_term = beta * phi_safe**3
    
    phix, phiy = grad(phi_safe)
    
    # Clip gradients to prevent extreme values
    phix = np.clip(phix, -10.0, 10.0)
    phiy = np.clip(phiy, -10.0, 10.0)
    
    grad2 = phix**2 + phiy**2
    
    # Clip gradient squared to prevent overflow
    grad2 = np.clip(grad2, 0.0, 100.0)
    
    grad2_term = beta * grad2
    
    # Chemical potential: mu = phi + beta*phi^3 - del^2(phi) + beta*|grad phi|^2
    lap_phi = laplacian(phi_safe)
    lap_phi = np.clip(lap_phi, -100.0, 100.0)
    
    chem_pot = phi_safe + phi3_term - lap_phi + grad2_term
    
    # Clip chemical potential to prevent extreme values
    chem_pot = np.clip(chem_pot, -100.0, 100.0)
    
    # Passive part: del^2(mu)
    lap_mu = laplacian(chem_pot)
    lap_mu = np.clip(lap_mu, -1000.0, 1000.0)
    
    # Active part: div(J), where J = lambda0*(1-phi)*(grad phi)^2*grad phi
    # Same enhanced lambda effect as in calculate_active_flux
    _, _, _, divJ = calculate_active_flux(phi_safe)
    
    # Further increase the dynamic effect of lambda on the simulation
    # by applying a nonlinear coefficient to the active term
    lambda_dynamic_factor = 1.0  # For lambda=0, no change
    if lambda0 > 0:
        # Make higher lambda values have increasingly stronger effects
        lambda_dynamic_factor = 1.0 + 0.5 * lambda0  
    
    # Complete right-hand side: del^2(mu) - lambda_dynamic_factor * divJ
    # Enhanced scaling to make different lambda values show distinct behaviors
    return lap_mu - lambda_dynamic_factor * divJ

def step(phi_old):
    """
    Perform one time step using a semi-implicit scheme with numerical stabilization
    """
    # Ensure input is in valid range
    phi_old = np.clip(phi_old, -1.0, 1.0)
    
    # Calculate nonlinear terms
    N1 = nonlinear(phi_old)
    
    # Clip to prevent extreme values
    N1 = np.clip(N1, -1000.0, 1000.0)
    
    # First half step (implicit linear part)
    lap_lap_phi = laplacian(laplacian(phi_old))
    lap_lap_phi = np.clip(lap_lap_phi, -1000.0, 1000.0)
    
    phi_tmp = phi_old + 0.5*dt*lap_lap_phi
    
    # Clip intermediate value
    phi_tmp = np.clip(phi_tmp, -1.0, 1.0)
    
    # Full non-linear step
    phi_tmp = phi_tmp + dt*N1
    
    # Clip intermediate value again
    phi_tmp = np.clip(phi_tmp, -1.0, 1.0)
    
    # Second half step (implicit linear part)
    lap_lap_phi_tmp = laplacian(laplacian(phi_tmp))
    lap_lap_phi_tmp = np.clip(lap_lap_phi_tmp, -1000.0, 1000.0)
    
    phi_new = phi_tmp + 0.5*dt*lap_lap_phi_tmp
    
    # Final clipping to ensure values stay in physical range
    phi_new = np.clip(phi_new, -1.0, 1.0)
    
    # Enforce boundary conditions
    return enforce_BC(phi_new)

def droplet_stats(field):
    """
    Count heterochromatin droplets (phi > threshold)
    Returns the number of droplets, mean size, and droplet type ("heterochromatin").
    """
    mask = field > droplet_threshold
    labeled, num_features = label(mask & mask_core)  # Only count droplets in core
    
    # Get sizes of features
    sizes = []
    if num_features > 0:
        objects = find_objects(labeled)
        for obj in objects:
            size = np.sum(labeled[obj] > 0)
            if size > 5:  # Ignore tiny features (noise)
                sizes.append(size)
    
    # Calculate mean size
    mean_size = np.mean(sizes) if sizes else 0
    
    # Return count, mean size, and type
    return num_features, mean_size, "heterochromatin"

def chemical_potential(phi_arr):
    """
    Calculate the chemical potential mu = phi + beta*phi^3 - del^2(phi) + beta*|grad phi|^2
    """
    phi3_term = beta * phi_arr**3
    
    phix, phiy = grad(phi_arr)
    grad2 = phix**2 + phiy**2
    
    grad2_term = beta * grad2
    
    # Chemical potential: mu = phi + beta*phi^3 - del^2(phi) + beta*|grad phi|^2
    chem_pot = phi_arr + phi3_term - laplacian(phi_arr) + grad2_term
    
    return chem_pot

def calculate_flux(phi_arr, mu):
    """
    Calculate the flux for visualization and analysis
    """
    phix, phiy = grad(phi_arr)
    grad2 = phix**2 + phiy**2
    
    # Active flux components
    Jx = lambda0 * (1 - phi_arr) * grad2 * phix
    Jy = lambda0 * (1 - phi_arr) * grad2 * phiy
    
    # Flux magnitude
    flux_mag = np.sqrt(Jx**2 + Jy**2)
    
    return Jx, Jy, flux_mag

def save_checkpoint(step_id):
    """
    Save a checkpoint of the simulation state
    """
    checkpoint_file = outdir / f"checkpoint_{step_id:08d}.npz"
    try:
        # Calculate flux for saving in checkpoint
        mu = chemical_potential(phi)
        jx, jy, _ = calculate_flux(phi, mu)
        
        np.savez(checkpoint_file,
                 step=step_id, 
                 phi=phi,
                 jx=jx,
                 jy=jy,
                 time_series=np.array(time_series),
                 n_drops=np.array(n_drop_list),
                 mean_sizes=np.array(mean_size_list))
        log(f"Saved checkpoint at step {step_id}")
    except Exception as e:
        log(f"Warning: Failed to save checkpoint at step {step_id}: {e}")

# ------------------  Additional Analysis Functions  -----------------------
def compute_fourier_modes(field):
    """
    Compute the Fourier transform of a field and return the 2D spectrum
    """
    fft_field = np.fft.fft2(field)
    fft_field = np.fft.fftshift(fft_field)  # Shift zero frequency to center
    return fft_field

def compute_power_spectrum(field):
    """
    Compute the 2D power spectrum of a field
    """
    fft_field = compute_fourier_modes(field)
    return np.abs(fft_field)**2

def compute_radial_spectrum(field):
    """
    Compute the radially-averaged power spectrum (1D) from a 2D spectrum
    """
    # Get 2D power spectrum
    power_2d = compute_power_spectrum(field)
    
    # Create coordinate grid
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=1.0/nx)
    ky = np.fft.fftfreq(ny, d=1.0/ny)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kxx, kyy = np.meshgrid(kx, ky)
    
    # Calculate radial distance from center
    k_rad = np.sqrt(kxx**2 + kyy**2)
    
    # Convert to integer indices for binning
    k_int = np.round(k_rad).astype(int)
    
    # Radial average by binning
    k_bins = np.bincount(k_int.flatten(), weights=power_2d.flatten())
    k_counts = np.bincount(k_int.flatten())
    radial_spectrum = k_bins / np.maximum(k_counts, 1)  # Avoid division by zero
    
    # Return wavenumbers and spectrum
    k_values = np.arange(len(radial_spectrum))
    return k_values, radial_spectrum

def compute_spatial_correlation(field):
    """
    Compute the two-point spatial correlation function of a field
    """
    # Subtract mean to get fluctuations
    field_fluct = field - np.mean(field)
    
    # Compute the 2D power spectrum
    power_2d = compute_power_spectrum(field_fluct)
    
    # Inverse FFT to get the correlation function
    corr_2d = np.fft.ifft2(np.fft.ifftshift(power_2d))
    corr_2d = np.real(corr_2d)
    
    # Normalize by the variance (zero-lag correlation)
    corr_2d = corr_2d / corr_2d[0, 0]
    
    # Compute radial average
    ny, nx = field.shape
    x = np.arange(nx) - nx//2
    y = np.arange(ny) - ny//2
    xx, yy = np.meshgrid(x, y)
    r_grid = np.sqrt(xx**2 + yy**2)
    r_int = np.round(r_grid).astype(int)
    
    # Perform radial averaging
    r_bins = np.bincount(r_int.flatten(), weights=corr_2d.flatten())
    r_counts = np.bincount(r_int.flatten())
    radial_corr = r_bins / np.maximum(r_counts, 1)  # Avoid division by zero
    
    # Return distances and correlation
    r_values = np.arange(len(radial_corr))
    return r_values, radial_corr

def compute_correlation_length(r_values, correlation):
    """
    Estimate the correlation length from the two-point correlation function
    """
    # Find the distance where correlation drops to 1/e
    threshold = 1.0/np.e
    
    try:
        # Find the first point where correlation drops below threshold
        idx = np.where(correlation < threshold)[0][0]
        if idx > 0:
            # Linear interpolation for better precision
            r1, r2 = r_values[idx-1], r_values[idx]
            c1, c2 = correlation[idx-1], correlation[idx]
            r_intercept = r1 + (r2 - r1) * (threshold - c1) / (c2 - c1)
            return r_intercept
        else:
            return r_values[0]
    except (IndexError, ValueError):
        # If correlation never drops below threshold
        return r_values[-1]

def compute_structure_factor(field):
    """
    Compute the structure factor S(k) from a field
    """
    # Subtract mean for fluctuations
    field_fluct = field - np.mean(field)
    
    # Get 2D power spectrum and perform radial averaging
    k_values, spectrum = compute_radial_spectrum(field_fluct)
    
    return k_values, spectrum

def velocity_field_from_flux(Jx, Jy):
    """
    Compute an effective velocity field from the flux
    """
    # Normalize by flux magnitude
    flux_mag = np.sqrt(Jx**2 + Jy**2) + 1e-10  # Avoid division by zero
    vx = Jx / flux_mag
    vy = Jy / flux_mag
    return vx, vy, flux_mag

# ----------------------- Visualization Function --------------------------
def create_visualizations(phi_field, step_id, t, n_d, m_sz, time_series, n_drop_list, mean_size_list):
    """
    Create comprehensive visualizations for the current simulation state,
    similar to the figures in the referenced papers
    """
    # Ensure phi is in stable range
    phi_safe = np.clip(phi_field, -1.0, 1.0)
    
    # Calculate flux and related quantities
    Jx, Jy, flux_mag, divJ = calculate_active_flux(phi_safe)
    
    # Scale flux magnitude by lambda0 for better visualization across lambda values
    # Enhanced visualization scaling to make lambda effects clearly visible
    if lambda0 > 0:
        # Stronger nonlinear scaling to emphasize differences between lambda values
        effective_flux_vmax = flux_vmax * (1.0 + 2.0 * lambda0)
        
        # Add lambda0 label explicitly to title with increased font size
        lambda_label = f"λ = {lambda0:.1f}"
    else:
        effective_flux_vmax = flux_vmax
        lambda_label = "λ = 0.0 (passive)"
    
    vx, vy, _ = velocity_field_from_flux(Jx, Jy)
    
    # Calculate correlation functions
    r_values, spatial_correlation = compute_spatial_correlation(phi_safe)
    correlation_length = compute_correlation_length(r_values, spatial_correlation)
    
    # Calculate structure factor
    k_values, structure_factor = compute_structure_factor(phi_safe)
    
    # Create a figure with 6 panels (3x2 grid for better visualization)
    fig = plt.figure(figsize=(16, 12))
    
    # Define a grid for flexible subplot layout
    gs = plt.GridSpec(3, 3, figure=fig)
    
    # Panel 1: Chromatin field phi (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(phi_safe.T, origin='lower', extent=(-1,1,-1,1),
                     cmap=field_cmap, vmin=-1, vmax=1)
    circle1 = plt.Circle((0, 0), R_phys, fill=False, edgecolor='black', linestyle='--')
    ax1.add_patch(circle1)
    ax1.set_title(f"Chromatin field, t = {t:.2f}\n{lambda_label}", fontsize=11)
    plt.colorbar(im1, ax=ax1, fraction=0.046, label="phi (-1:euchromatin, +1:heterochromatin)")
    
    # Panel 2: Active flux magnitude (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if lambda0 > 0:
        im2 = ax2.imshow(flux_mag.T, origin='lower', extent=(-1,1,-1,1),
                       cmap=flux_cmap, vmin=flux_vmin, vmax=effective_flux_vmax)
        circle2 = plt.Circle((0, 0), R_phys, fill=False, edgecolor='black', linestyle='--')
        ax2.add_patch(circle2)
        ax2.set_title(f"Active flux magnitude\n{lambda_label}", fontsize=11)
        max_flux = np.max(flux_mag)
        ax2.text(0.05, -0.95, f"Max flux: {max_flux:.4f}", color='white', 
                backgroundcolor='black', fontsize=8)
        plt.colorbar(im2, ax=ax2, fraction=0.046, label="Flux magnitude")
    else:
        im2 = ax2.imshow(np.zeros_like(phi_field.T), origin='lower', extent=(-1,1,-1,1),
                      cmap=flux_cmap, vmin=flux_vmin, vmax=effective_flux_vmax)
        circle2 = plt.Circle((0, 0), R_phys, fill=False, edgecolor='black', linestyle='--')
        ax2.add_patch(circle2)
        ax2.text(0.5, 0.5, "No active flux\n(λ = 0)", 
                ha='center', va='center', fontsize=14)
        ax2.set_title("Active flux magnitude")
        plt.colorbar(im2, ax=ax2, fraction=0.046, label="Flux magnitude")
    
    # Panel 3: Velocity field with streamlines (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(phi_safe.T, origin='lower', extent=(-1,1,-1,1),
                  cmap=field_cmap, vmin=-1, vmax=1, alpha=0.7)
    
    if lambda0 > 0:
        # Use quiver plot instead of streamplot for robustness
        skip = flux_skip  # Use the same skip factor
        
        # Create grid for quiver plot (simple, regular grid)
        x_grid = np.linspace(-1, 1, Nx)[::skip]
        y_grid = np.linspace(-1, 1, Ny)[::skip]
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Sample flux vectors at grid points
        Jx_grid = Jx[::skip, ::skip].T  # Transpose for correct orientation
        Jy_grid = Jy[::skip, ::skip].T
        
        # Normalize vectors to better see direction
        J_mag = np.sqrt(Jx_grid**2 + Jy_grid**2) + 1e-10
        Jx_norm = Jx_grid / J_mag
        Jy_norm = Jy_grid / J_mag
        
        # Plot quiver
        ax3.quiver(X_grid, Y_grid, Jx_norm, Jy_norm, 
                  scale=30, width=0.003, color='black',
                  pivot='mid', headwidth=4, headlength=4)
    
    circle3 = plt.Circle((0, 0), R_phys, fill=False, edgecolor='black', linestyle='--')
    ax3.add_patch(circle3)
    ax3.set_title(f"Active flux directions")
    
    # Panel 4: Droplet statistics / time series (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    if len(time_series) > 1:
        ax4.plot(time_series, n_drop_list, 'b-o', label='Droplet count')
        if lambda0 == 0 and len(time_series) > 5:
            # Add reference power law for passive case
            tref = np.array(time_series[1:])
            nref = 50*tref**(-1/3)
            ax4.plot(tref, nref, 'k--', label=r'$\sim t^{-1/3}$')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Droplet count')
        ax4.set_title(f'Heterochromatin droplets vs time')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        if len(time_series) > 5:
            ax4.set_xscale('log')
            ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, "Insufficient data\nfor time series", 
                ha='center', va='center', fontsize=14)
        ax4.set_title("Droplet statistics")
    
    # Panel 5: Structure factor S(k) (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    if len(k_values) > 1:
        # Plot structure factor (excluding k=0)
        ax5.plot(k_values[1:], structure_factor[1:], '-o')
        ax5.set_xlabel('Wavenumber k')
        ax5.set_ylabel('Structure factor S(k)')
        ax5.set_title('Structure factor')
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "Error computing\nstructure factor", 
                ha='center', va='center', fontsize=14)
        ax5.set_title("Structure factor")
    
    # Panel 6: Two-point correlation function (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    if len(r_values) > 1:
        # Plot correlation function with correlation length marked
        ax6.plot(r_values[:min(100, len(r_values))], 
               spatial_correlation[:min(100, len(r_values))], '-o')
        ax6.axhline(1/np.e, color='r', linestyle='--', 
                   label=r'$1/e$ threshold')
        ax6.axvline(correlation_length, color='g', linestyle='--', 
                   label=f'ξ = {correlation_length:.1f}')
        ax6.set_xlabel('Distance r')
        ax6.set_ylabel('Correlation C(r)')
        ax6.set_title(f'Two-point correlation, ξ = {correlation_length:.1f}')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, "Error computing\ncorrelation function", 
                ha='center', va='center', fontsize=14)
        ax6.set_title("Two-point correlation")
    
    # Panel 7: Separate euchromatin and heterochromatin regions (bottom left)
    ax7 = fig.add_subplot(gs[2, 0])
    # Create a masked version showing only the high/low regions
    phi_masked = np.copy(phi_safe)
    high_mask = phi_safe > 0.5  # Heterochromatin
    low_mask = phi_safe < -0.5  # Euchromatin
    
    # Use different colors for heterochromatin and euchromatin
    cmap_regions = plt.cm.coolwarm
    im7 = ax7.imshow(phi_masked.T, origin='lower', extent=(-1,1,-1,1),
                    cmap=cmap_regions, vmin=-1, vmax=1)
    
    # Overlay contours to highlight the interfaces
    contour = ax7.contour(X, Y, phi_safe, levels=[0], colors='k', linewidths=0.5)
    
    circle7 = plt.Circle((0, 0), R_phys, fill=False, edgecolor='black', linestyle='--')
    ax7.add_patch(circle7)
    ax7.set_title(f"Chromatin regions")
    plt.colorbar(im7, ax=ax7, fraction=0.046, label="phi")
    
    # Panel 8: Divergence of the flux (bottom middle)
    ax8 = fig.add_subplot(gs[2, 1])
    if lambda0 > 0:
        divJ_max = max(abs(np.min(divJ)), abs(np.max(divJ)))
        divJ_vmin, divJ_vmax = -divJ_max, divJ_max
        im8 = ax8.imshow(divJ.T, origin='lower', extent=(-1,1,-1,1),
                      cmap='RdBu_r', vmin=divJ_vmin, vmax=divJ_vmax)
        circle8 = plt.Circle((0, 0), R_phys, fill=False, edgecolor='black', linestyle='--')
        ax8.add_patch(circle8)
        ax8.set_title(f"Divergence of flux, div(J)")
        plt.colorbar(im8, ax=ax8, fraction=0.046, label="div(J)")
    else:
        ax8.text(0.5, 0.5, "No active flux\n(λ = 0)", 
                ha='center', va='center', fontsize=14)
        ax8.set_title("Divergence of flux")
    
    # Panel 9: Power spectrum of phi (bottom right)
    ax9 = fig.add_subplot(gs[2, 2])
    power_2d = compute_power_spectrum(phi_safe)
    # Take log of power for better visualization
    log_power = np.log10(power_2d + 1e-10)  # Add small constant to avoid log(0)
    im9 = ax9.imshow(log_power.T, origin='lower', 
                   cmap='viridis')
    ax9.set_title(f"Power spectrum (log scale)")
    plt.colorbar(im9, ax=ax9, fraction=0.046, label="log10(Power)")
    
    # Common settings for spatial panels
    for ax in [ax1, ax2, ax3, ax7, ax8]:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add panel labels
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]):
        ax.text(-0.12, 1.05, chr(97+i), transform=ax.transAxes, 
               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    filename = outdir/f"frame_{step_id:08d}.png"
    return fig, filename

# -------------------------- Main simulation loop -------------------------
time_series, n_drop_list, mean_size_list = [], [], []

# Visualization settings
field_cmap = 'coolwarm'  # For phi field (blue: euchromatin, red: heterochromatin)
flux_cmap = 'plasma'     # For active flux magnitude

# Protection wrapper for file operations
def safe_save_plot(fig, filename):
    try:
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        return True
    except Exception as e:
        log(f"Warning: Failed to save plot to {filename}: {e}")
        plt.close(fig)
        return False

try:
    for step_id in range(n_steps+1):
        t = step_id*dt
        
        if step_id % save_every == 0:
            n_d, m_sz, droplet_type = droplet_stats(phi)
            time_series.append(t)
            n_drop_list.append(n_d)
            mean_size_list.append(m_sz)
            
            # Save text data
            if step_id > 0 and step_id % (save_every * 5) == 0:
                try:
                    np.savetxt(outdir/'droplet_count.txt',
                               np.column_stack([time_series, n_drop_list, mean_size_list]),
                               header='t   count   mean_size')
                except Exception as e:
                    log(f"Warning: Failed to save droplet count at step {step_id}: {e}")
            
            # For long runs, only save visualization less frequently
            if step_id % (save_every * 5) == 0:
                # Create comprehensive visualizations
                fig, filename = create_visualizations(
                    phi, step_id, t, n_d, m_sz, 
                    time_series, n_drop_list, mean_size_list
                )
                safe_save_plot(fig, filename)
                
                log(f"Step {step_id}/{n_steps}: t = {t:.2f}, heterochromatin droplets = {n_d}")
        
        # Save checkpoints periodically
        if step_id > 0 and step_id % checkpoint_every == 0:
            save_checkpoint(step_id)
        
        # Advance simulation by one time step
        if step_id < n_steps:
            phi = step(phi)
    
    # Save final state
    save_checkpoint(n_steps)
    
    # Save final droplet count data
    np.savetxt(outdir/'droplet_count.txt',
               np.column_stack([time_series, n_drop_list, mean_size_list]),
               header='t   count   mean_size')
    
    # Create summary plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(time_series, n_drop_list, 'o-')
    if lambda0 == 0 and len(time_series) > 5:
        tref = np.array(time_series[1:])
        nref = 50*tref**(-1/3)
        ax1.plot(tref, nref, 'k--', label=r'$\sim t^{-1/3}$')
        ax1.legend()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Droplet count')
    ax1.set_title(f'{droplet_type.capitalize()} droplets vs time (lambda0={lambda0})')
    ax1.grid(True, alpha=0.3)
    if len(time_series) > 5:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
    
    ax2.plot(time_series, mean_size_list, 'o-')
    if lambda0 == 0 and len(time_series) > 5:
        tref = np.array(time_series[1:])
        sref = 0.1*tref**(1/3)
        ax2.plot(tref, sref, 'k--', label=r'$\sim t^{1/3}$')
        ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean droplet size')
    ax2.set_title(f'Mean {droplet_type} droplet size vs time (lambda0={lambda0})')
    ax2.grid(True, alpha=0.3)
    if len(time_series) > 5:
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    
    plt.tight_layout()
    safe_save_plot(fig, outdir/'summary_plots.png')
    
    log(f"Simulation completed successfully. Results in {outdir}")

except Exception as e:
    log(f"Error during simulation: {e}")
    traceback.print_exc(file=log_file)
    sys.exit(1)

finally:
    log_file.close()
    print(f"Simulation finished. Results in {outdir}")

def save_final_plots(outdir, time_series, n_drop_list, mean_size_list, phi_mean, lambda0):
    """Save a final set of plots summarizing time evolution"""
    # Create a figure to show time evolution data
    fig_ts = plt.figure(figsize=(16, 8))
    ax1 = fig_ts.add_subplot(121)
    ax2 = fig_ts.add_subplot(122)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Droplet count')
    ax1.set_title(f'Heterochromatin droplets vs time (lambda0={lambda0})')
    ax1.grid(True, alpha=0.3)
    if len(time_series) > 5:
        ax1.plot(time_series, n_drop_list, 'b-o', label='Droplet count')
        ax1.legend()
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean droplet size')
    ax2.set_title(f'Mean heterochromatin droplet size vs time (lambda0={lambda0})')
    ax2.grid(True, alpha=0.3)
    if len(time_series) > 5:
        ax2.plot(time_series, mean_size_list, 'r-o', label='Mean size')
        ax2.legend()
    
    fig_ts.tight_layout()
    safe_save_plot(fig_ts, outdir / "time_series.png")
    plt.close(fig_ts) 
