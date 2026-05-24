# Surface Energy Balance (SEB) Climate Analysis

A comprehensive climate data analysis workflow and interactive visualization dashboard focusing on the Surface Energy Balance over African sub-regions.

## Overview
This project processes, analyzes, and visualizes climate data (such as temperature, precipitation, heat fluxes, and aridity indices) to better understand climate trends, seasonal climatology, and energy partitioning. It leverages data from the Copernicus Climate Data Store (CDS) and climate models like HadGEM2-CC under the RCP8.5 scenario.

## Project Structure
- `climate_seb_updated.ipynb`: The core Jupyter Notebook containing the full data engineering pipeline—from data loading, regridding, and masking, to statistical analysis (Mann-Kendall trend tests, EOF/PCA), and exporting artifacts.
- `outputs/`: 
  - Contains the exported statistical results in CSV format, spatial mapping data, and NetCDF anomaly files.
  - Hosts the modular **Dash web application** (`app.py`, `layout.py`, `callbacks.py`, `data_loader.py`) for an interactive exploration of the computed climate metrics.
- `data/`: Directory designated for the raw `.nc` (NetCDF) model data files.
- `qualite_donnees.md`: Documentation on data quality, processing steps (like interpolation and handling of missing values), and data provenance.

## Key Features
- **Spatial Regridding & Climatology**: Analyzes spatial data at 0.5-degree grid resolution.
- **Regional Analysis**: Focuses on specific zones such as North Africa, the Sahel, and West Africa.
- **Statistical Trend Testing**: Uses the Mann-Kendall test to identify significant trends in energy fluxes and climate variables over the 1960–2099 timeframe.
- **Variability Modes (EOF/PCA)**: Evaluates principal components of spatial anomalies to explain major variance patterns.
- **Interactive Dashboard**: A multi-tab Dash application to browse the exported maps, decadal changes, time series, and extreme event frequencies.

## Prerequisites
To run the analysis pipeline or the dashboard, ensure you have a Python environment set up with the following primary dependencies:
- `xarray`
- `dask`
- `pandas`
- `numpy`
- `dash`
- `plotly`
- `dash-bootstrap-components`

## Usage
1. **Data Pipeline**: Run the cells in `climate_seb_updated.ipynb` sequentially to process the raw NetCDF data and export the findings to the `outputs/` folder.
2. **Dashboard**: 
   - You can launch the interactive dashboard directly from the notebook.
   - Alternatively, navigate to the `outputs/` directory and run:  
     `python app.py`
   - Access the dashboard at `http://127.0.0.1:8050`.