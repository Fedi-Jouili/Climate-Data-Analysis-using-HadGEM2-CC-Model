"""
Robust Climate Data Regridding Pipeline
Author: Senior Climate-Data Scientist

This module provides the correct scientific workflow and utilities to prevent 
regridding artifacts (blurry zones, staircase effects, band distortions) when 
working with climate data fields such as aridity_coldest-quarter.
"""

import xarray as xr
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def standardize_coords(ds, lon_var='lon', lat_var='lat'):
    """
    Standardize the coordinate system of an xarray Dataset to avoid interpolation artifacts.
    1. Ensures longitude is in the [-180, 180] range (prevents prime meridian seams).
    2. Ensures latitude is monotonically increasing (prevents flipped arrays and SciPy NaNs).
    """
    ds = ds.copy()
    
    # 1. Normalize Longitude to [-180, 180]
    if ds[lon_var].max() > 180:
        ds.coords[lon_var] = (ds.coords[lon_var] + 180) % 360 - 180
        ds = ds.sortby(lon_var)
        print(f"[*] Converted {lon_var} from [0, 360] to [-180, 180] and sorted.")

    # 2. Ensure Latitude is monotonically increasing
    if ds[lat_var].values[0] > ds[lat_var].values[-1]:
        ds = ds.isel({lat_var: slice(None, None, -1)})
        print(f"[*] Flipped {lat_var} to be monotonically increasing.")
        
    return ds

def create_regridder(ds_in, ds_out, method='bilinear', periodic=True, extrap_method='nearest_s2d'):
    """
    Create a robust xESMF regridder avoiding boundary and coastal artifacts.
    
    Args:
        ds_in: Source dataset.
        ds_out: Target dataset.
        method: Interpolation method. 
                - Use 'bilinear' for smooth continuous fields (Temperature).
                - Use 'conservative' for fluxes/extensive fields (Precipitation).
                - Use 'nearest_s2d' for categorical masks.
        periodic: Must be True for global grids to prevent a seam at the antimeridian.
        extrap_method: Extrapolate to prevent NaN coastal erosion when blending grids.
    """
    print(f"[*] Creating {method} Regridder (Periodic: {periodic})")
    
    # reuse_weights=False ensures we don't accidentally load corrupted weights
    regridder = xe.Regridder(
        ds_in, 
        ds_out, 
        method, 
        periodic=periodic,
        extrap_method=extrap_method,
        ignore_degenerate=True # helps with slightly irregular bounds
    )
    return regridder

def verify_regridding(ds_in, ds_out, var_name):
    """
    Quantitatively verify the regridding process by comparing global area-weighted means.
    A properly conservative or bilinear regrid should have a negligible difference.
    """
    weights_in = np.cos(np.deg2rad(ds_in.lat))
    weights_out = np.cos(np.deg2rad(ds_out.lat))
    
    mean_in = ds_in[var_name].weighted(weights_in).mean().values
    mean_out = ds_out[var_name].weighted(weights_out).mean().values
    
    print(f"\n--- Verification for '{var_name}' ---")
    print(f"Source Global Mean: {mean_in:.4f}")
    print(f"Target Global Mean: {mean_out:.4f}")
    print(f"Difference:         {abs(mean_in - mean_out):.4e}")
    if abs(mean_in - mean_out) > 0.05 * abs(mean_in):
        print("[!] WARNING: Significant energy/mass loss detected during regridding!")

def plot_sanity_check(ds_in, ds_out, var_name):
    """
    Plot source and target datasets side-by-side using Cartopy to visually spot artifacts.
    Always uses pcolormesh to avoid imshow distortions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300, subplot_kw={'projection': ccrs.Robinson()})
    
    # Source
    ax = axes[0]
    ax.coastlines()
    ax.set_title(f"Source Data: {var_name}")
    # Note: explicitly passing transform=ccrs.PlateCarree() is critical for Cartopy mapping
    ds_in[var_name].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', add_colorbar=True
    )
    
    # Target
    ax = axes[1]
    ax.coastlines()
    ax.set_title(f"Regridded Data: {var_name}")
    ds_out[var_name].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', add_colorbar=True
    )
    
    plt.tight_layout()
    plt.show()

def run_anomaly_pipeline(ds_raw, ds_target, var_name, method='bilinear'):
    """
    Correct scientific workflow: 
    1. Regrid RAW DATA first.
    2. Compute anomalies on the TARGET grid.
    
    This prevents coastal smearing where land-anomalies and ocean-anomalies 
    are physically mixed by the spatial filter.
    """
    print("[1] Standardizing Source Coordinates...")
    ds_raw = standardize_coords(ds_raw)
    
    print("[2] Standardizing Target Coordinates...")
    ds_target = standardize_coords(ds_target)
    
    print(f"[3] Regridding RAW field '{var_name}'...")
    regridder = create_regridder(ds_raw, ds_target, method=method)
    ds_raw_regridded = regridder(ds_raw, keep_attrs=True)
    
    print("[4] Computing Climatology on TARGET grid...")
    # Standard 30-year or full-period mean calculation
    climatology = ds_raw_regridded.mean('time')
    
    print("[5] Computing Anomalies on TARGET grid...")
    anomalies = ds_raw_regridded - climatology
    
    print("[6] Verification (First Time Step)...")
    # Verify using the first time slice for speed
    t_idx = 0 if 'time' in ds_raw.dims else None
    ds_check_in = ds_raw.isel(time=t_idx) if t_idx is not None else ds_raw
    ds_check_out = ds_raw_regridded.isel(time=t_idx) if t_idx is not None else ds_raw_regridded
    verify_regridding(ds_check_in, ds_check_out, var_name)
    
    return anomalies, ds_raw_regridded

if __name__ == '__main__':
    print("Climate Data Regridding utilities successfully loaded.")
    # Example usage:
    # ds_anom, ds_raw_regrid = run_anomaly_pipeline(ds_cmip, ds_obs_target, 'aridity_coldest-quarter')
    # plot_sanity_check(ds_cmip.isel(time=0), ds_raw_regrid.isel(time=0), 'aridity_coldest-quarter')
