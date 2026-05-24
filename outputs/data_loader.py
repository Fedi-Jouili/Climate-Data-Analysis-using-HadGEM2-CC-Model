import os
import json
import pandas as pd
import xarray as xr

# ==========================================
# 1. LOAD DATA & INVENTORY
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
inventory = []

def load_csv(filename, index_col=None):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=index_col)
            # Check rows/shape
            inventory.append((filename, f"{df.shape[0]} rows", "v"))
            return df
        except Exception as e:
            inventory.append((filename, f"ERROR: {str(e)}", "x ERROR"))
            return pd.DataFrame()
    else:
        inventory.append((filename, "MISSING", "x MISSING"))
        return pd.DataFrame()

def load_nc(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        try:
            ds = xr.open_dataset(path)
            shape_str = f"({','.join(map(str, ds.dims.values()))}) x{len(ds.data_vars)}v"
            inventory.append((filename, shape_str, "v"))
            return ds
        except Exception as e:
            inventory.append((filename, f"ERROR: {str(e)}", "x ERROR"))
            return None
    else:
        inventory.append((filename, "MISSING", "x MISSING"))
        return None

def load_json(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            inventory.append((filename, "JSON", "v"))
            return data
        except Exception as e:
            inventory.append((filename, f"ERROR: {str(e)}", "x ERROR"))
            return {}
    else:
        inventory.append((filename, "MISSING", "x MISSING"))
        return {}

print(f"{'FILE':<30} | {'ROWS/SHAPE':<15} | {'STATUS'}")
print("-" * 30 + "|" + "-" * 17 + "|" + "-" * 7)

# Load config
config = load_json("config.json")

spatial_period = load_csv("spatial_period_maps.csv")
if not spatial_period.empty:
    try:
        df_sp = spatial_period.pivot_table(index=['period_year', 'lat', 'lon'], columns='variable', values='value')
        ds_anomaly = xr.Dataset.from_dataframe(df_sp).rename({'period_year': 'time'})
    except:
        ds_anomaly = None
else:
    ds_anomaly = None

eof_loadings = load_csv("eof_spatial_loadings.csv")
if not eof_loadings.empty and 'eof' in eof_loadings.columns:
    try:
        eof_loadings = eof_loadings.pivot(index=['lat', 'lon'], columns='eof', values='loading').reset_index()
    except:
        pass

eof_scores = load_csv("eof_pc_scores.csv")
if not eof_scores.empty and 'pc' in eof_scores.columns:
    try:
        eof_scores = eof_scores.pivot(index='year', columns='pc', values='score').reset_index()
        eof_scores.rename(columns={'PC1': 'EOF1', 'PC2': 'EOF2', 'PC3': 'EOF3', 'PC4': 'EOF4'}, inplace=True)
    except:
        pass

var_stats = load_csv("variable_statistics.csv")
mk_results = load_csv("mann_kendall_results.csv")

decadal = load_csv("decadal_changes.csv")
if not decadal.empty and 'Unnamed: 0' in decadal.columns:
    decadal.rename(columns={'Unnamed: 0': 'variable'}, inplace=True)

reg_na = load_csv("regional_ts_north_africa.csv")
if not reg_na.empty and 'variable' in reg_na.columns:
    try:
        reg_na = reg_na.pivot(index='year', columns='variable', values='value').reset_index()
    except:
        pass

reg_sa = load_csv("regional_ts_sahel.csv")
if not reg_sa.empty and 'variable' in reg_sa.columns:
    try:
        reg_sa = reg_sa.pivot(index='year', columns='variable', values='value').reset_index()
    except:
        pass

reg_wa = load_csv("regional_ts_west_africa.csv")
if not reg_wa.empty and 'variable' in reg_wa.columns:
    try:
        reg_wa = reg_wa.pivot(index='year', columns='variable', values='value').reset_index()
    except:
        pass

extremes = load_csv("extreme_events.csv")
corr_matrix = load_csv("correlation_matrix.csv", index_col=0)

seasonal = load_csv("seasonal_climatology.csv")
if not seasonal.empty and 'month_name' in seasonal.columns:
    try:
        seasonal = seasonal.pivot(index=['variable', 'period_label'], columns='month_name', values='value').reset_index()
        labels = seasonal['period_label'].unique()
        label_to_idx = {l: i for i, l in enumerate(labels)}
        seasonal['period_index'] = seasonal['period_label'].map(label_to_idx)
    except:
        pass

energy = load_csv("energy_balance.csv")
if not energy.empty and 'period_year' in energy.columns:
    energy.rename(columns={'period_year': 'year'}, inplace=True)

# Secondary files
spatial_mean = load_csv("spatial_mean_maps.csv")
period_spatial = load_csv("period_spatial_means.csv")
trend_slope = load_csv("trend_slope_maps.csv")

for item in inventory:
    status_icon = "V" if item[2] == "v" else "X MISSING"
    print(f"{item[0]:<30} | {item[1]:<15} | {status_icon}")

regional_ts = {
    "North Africa": reg_na,
    "Sahel": reg_sa,
    "West Africa": reg_wa
}

# Determine default variables
variables = []
if ds_anomaly is not None:
    variables = list(ds_anomaly.data_vars)
elif not decadal.empty and 'variable' in decadal.columns:
    variables = decadal['variable'].unique().tolist()
else:
    variables = ['temperature_mean', 'precipitation_flux'] # Fallback
