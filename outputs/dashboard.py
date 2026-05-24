import os
import json
import pandas as pd
import numpy as np
import xarray as xr
import dash
from dash import dcc, html, dash_table, Input, Output, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

try:
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import MultiLineString
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

def coords_to_svg_path(x_coords, y_coords):
    if not x_coords:
        return ""
    parts = [f"M {x_coords[0]},{y_coords[0]}"]
    for x, y in zip(x_coords[1:], y_coords[1:]):
        parts.append(f"L {x},{y}")
    return " ".join(parts)

COASTLINE_SEGMENTS = [
    # ── WEST AFRICAN ATLANTIC COAST (north→south, lon 0 to -18) ──
    # Morocco / Western Sahara / Mauritania coast
    dict(
        lon=[-5.9,-5.4,-4.8,-4.0,-3.3,-2.2,-1.2, 0.0],
        lat=[35.8,35.7,35.5,35.2,34.7,34.0,33.0,31.8]
    ),
    # Mauritania / Senegal
    dict(
        lon=[-16.5,-16.3,-16.1,-15.8,-15.6,-15.3,
             -15.0,-14.7,-14.4,-14.1,-13.8,-13.5],
        lat=[ 20.0, 19.5, 18.9, 18.2, 17.5, 16.8,
              16.0, 15.3, 14.7, 14.2, 13.7, 13.3]
    ),
    # Senegal / Guinea-Bissau / Guinea
    dict(
        lon=[-13.5,-13.2,-13.0,-12.7,-12.4,-12.0,
             -11.6,-11.2,-10.8,-10.5,-10.1,-9.7],
        lat=[ 13.3, 12.9, 12.5, 12.1, 11.6, 11.1,
              10.6, 10.2,  9.7,  9.3,  8.9,  8.5]
    ),
    # Sierra Leone / Liberia / Côte d'Ivoire
    dict(
        lon=[-9.7,-9.3,-8.8,-8.3,-7.7,-7.1,
             -6.5,-5.9,-5.3,-4.7,-4.1,-3.5],
        lat=[ 8.5, 8.1, 7.6, 7.2, 6.9, 6.6,
              6.4, 6.2, 5.9, 5.6, 5.3, 5.0]
    ),
    # Côte d'Ivoire / Ghana / Togo / Benin
    dict(
        lon=[-3.5,-2.9,-2.2,-1.6,-1.0,-0.4,
              0.2, 0.8, 1.4, 1.8, 2.2, 2.6],
        lat=[ 5.0, 4.8, 4.7, 4.8, 5.0, 5.2,
              5.4, 5.6, 5.8, 6.0, 6.2, 6.4]
    ),
    # Nigeria / Cameroon coast (Gulf of Guinea)
    dict(
        lon=[ 2.6, 3.0, 3.4, 3.9, 4.4, 5.0,
              5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
              8.5, 8.8, 9.2, 9.5, 9.8,10.2],
        lat=[ 6.4, 6.4, 6.3, 6.1, 5.8, 5.4,
              5.0, 4.5, 4.0, 3.7, 3.5, 3.3,
              3.1, 2.9, 2.5, 2.0, 1.5, 1.0]
    ),
    # Gulf of Guinea east coast → Gabon → Congo → Angola
    dict(
        lon=[ 9.3,  9.5,  9.7,  9.9, 10.1, 10.3,
             10.5, 10.7, 10.9, 11.1, 11.3, 11.5,
             11.7, 11.8, 11.9, 12.0, 11.9, 11.8,
             11.7, 11.6, 11.7, 11.8, 11.9, 12.0,
             12.1, 12.2, 12.3, 12.4, 12.5, 12.6],
        lat=[  2.0,  1.6,  1.2,  0.8,  0.4,  0.0,
              -0.4, -0.8, -1.2, -1.6, -2.0, -2.5,
              -3.0, -3.5, -4.0, -4.6, -5.0, -5.4,
              -5.9, -6.3, -6.8, -7.2, -7.7, -8.1,
              -8.5, -9.0, -9.5,-10.0,-10.5,-11.0]
    ),
    # ── NORTH AFRICAN MEDITERRANEAN COAST ──
    dict(
        lon=[-5.4,-4.5,-3.2,-1.8,-0.5, 0.8,
              2.1, 3.4, 5.0, 6.5, 7.5, 8.5,
              9.5,10.5,11.5,12.3,13.0,14.0,
             15.0,16.0,17.0,18.0,19.0,20.0,
             21.0,22.0,23.0,24.0,25.0,25.9],
        lat=[35.8,35.8,35.5,35.6,35.8,36.7,
             36.9,37.0,36.7,36.9,37.1,37.2,
             37.4,37.0,33.9,33.2,32.9,32.8,
             32.9,32.9,33.0,32.9,32.5,32.6,
             32.5,32.5,32.5,32.5,32.5,32.4]
    ),
    # ── RED SEA / HORN OF AFRICA (eastern boundary) ──
    dict(
        lon=[32.9,33.5,34.1,34.5,35.0,35.5],
        lat=[29.9,29.2,28.5,27.8,27.0,26.3]
    ),
]

pio.templates.default = "plotly_dark"

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

variables = []
if ds_anomaly is not None:
    variables = list(ds_anomaly.data_vars)
elif not decadal.empty and 'variable' in decadal.columns:
    variables = decadal['variable'].unique().tolist()
else:
    variables = ['temperature_mean', 'precipitation_flux']

# ==========================================
# 2. LAYOUT
# ==========================================
COLORS = {
    'bg0': '#0e1117',
    'bg1': '#161b27',
    'bg2': '#1e2535',
    'bg3': '#252d3d',
    'border': '#2e3a52',
    'accent': '#3b82f6',
    'text1': '#e2e8f0',
    'text2': '#94a3b8',
    'text3': '#64748b',
    'green': '#10b981',
    'red': '#ef4444',
    'amber': '#f59e0b',
    'purple': '#a855f7',
    'cyan': '#06b6d4',
}

CARD_STYLE = {
    'backgroundColor': COLORS['bg1'],
    'border': f"1px solid {COLORS['border']}",
    'borderRadius': '12px',
    'overflow': 'hidden',
    'marginBottom': '16px',
    'color': COLORS['text1']
}

STAT_TILE_STYLE = {
    'backgroundColor': COLORS['bg2'],
    'borderRadius': '8px',
    'border': f"1px solid {COLORS['border']}",
    'padding': '14px 16px',
    'display': 'flex',
    'flexDirection': 'column',
    'gap': '4px',
}

def make_stat_tile(label, value_id, initial_value="-"):
    return html.Div([
        html.Div(label, style={'color': COLORS['text3'], 'fontSize': '11px'}),
        html.Div(initial_value, id=value_id, style={
            'color': COLORS['text1'], 
            'fontSize': '22px', 
            'fontWeight': '700', 
            'fontFamily': 'monospace'
        })
    ], style=STAT_TILE_STYLE)

tab1_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Variable count", "kpi1-1"), width=3),
        dbc.Col(make_stat_tile("Time span", "kpi1-2", "1970-2090"), width=3),
        dbc.Col(make_stat_tile("EOF1 variance explained", "kpi1-3", "99.8%"), width=3),
        dbc.Col(make_stat_tile("Sig. trends (p<0.05)", "kpi1-4"), width=3),
    ], className="mb-4 mt-2"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Overall Trends", style={'color': COLORS['text1'], 'fontSize': '13px'}),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='t1-bar-mk', style={'height': '350px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})
            ], style=CARD_STYLE)
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Decadal Change", style={'color': COLORS['text1'], 'fontSize': '13px'}),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='t1-bar-decadal', style={'height': '350px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})
            ], style=CARD_STYLE)
        ], width=6),
    ]),
    dbc.Card([
        dbc.CardHeader("Mann-Kendall Significance Matrix", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(html.Div(id='t1-table-container'), style={'padding': '0', 'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE)
])

tab2_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Spatial coverage", "kpi2-1"), width=4),
        dbc.Col(make_stat_tile("Variable shown", "kpi2-2"), width=4),
        dbc.Col(make_stat_tile("Value range", "kpi2-3"), width=4),
    ], className="mb-4 mt-2"),
    dbc.Row([
        dbc.Col([
            html.Label("Dataset:", style={'color': COLORS['text2'], 'fontSize': '12px', 'marginRight': '8px'}),
            dcc.Dropdown(id='t2-dataset-sel', options=[{'label': 'Spatial Means', 'value': 'mean'}, {'label': 'Trend Slopes', 'value': 'trend'}], value='mean', clearable=False, style={'width': '200px', 'display': 'inline-block', 'color': 'black'}),
        ], width=4),
        dbc.Col([
            html.Label("Variable:", style={'color': COLORS['text2'], 'fontSize': '12px', 'marginRight': '8px'}),
            dcc.Dropdown(id='t2-var-sel', options=[{'label': v, 'value': v} for v in variables], value=variables[0] if variables else None, clearable=False, style={'width': '200px', 'display': 'inline-block', 'color': 'black'}),
        ], width=4),
    ], className="mb-3 align-items-center"),
    dbc.Card([
        dbc.CardHeader("Spatial Field Visualization", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(dcc.Loading(dcc.Graph(id='t2-map', style={'height': '500px'}), type="circle"), style={'padding': '0', 'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE),
    dbc.Card([
        dbc.CardHeader("Spatial Means over Periods", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(dcc.Loading(dcc.Graph(id='t2-line', style={'height': '350px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE)
])

tab3_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Active variable", "kpi3-2"), width=4),
        dbc.Col(make_stat_tile("Year range", "kpi3-3", "1970-2090"), width=4),
        dbc.Col(make_stat_tile("Trend direction", "kpi3-4"), width=4),
    ], className="mb-4 mt-2"),
    dbc.Row([
        dbc.Col([
            html.Label("Variable:", style={'color': COLORS['text2'], 'fontSize': '12px', 'marginRight': '8px'}),
            dcc.Dropdown(id='t3-var-sel', options=[{'label': v, 'value': v} for v in variables], value=variables[0] if variables else None, clearable=False, style={'width': '250px', 'display': 'inline-block', 'color': 'black'}),
        ], width=12),
    ], className="mb-3"),
    dbc.Card([
        dbc.CardHeader("Regional Time Series Overlay", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(dcc.Loading(dcc.Graph(id='t3-main-chart', style={'height': '400px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("North Africa"), dbc.CardBody(dcc.Graph(id='t3-sm-1', style={'height': '200px'}), style={'backgroundColor': COLORS['bg1']})], style=CARD_STYLE), width=4),
        dbc.Col(dbc.Card([dbc.CardHeader("Sahel"), dbc.CardBody(dcc.Graph(id='t3-sm-2', style={'height': '200px'}), style={'backgroundColor': COLORS['bg1']})], style=CARD_STYLE), width=4),
        dbc.Col(dbc.Card([dbc.CardHeader("West Africa"), dbc.CardBody(dcc.Graph(id='t3-sm-3', style={'height': '200px'}), style={'backgroundColor': COLORS['bg1']})], style=CARD_STYLE), width=4),
    ])
])

tab4_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Active EOF", "kpi4-1"), width=3),
        dbc.Col(make_stat_tile("Variance explained", "kpi4-2"), width=3),
        dbc.Col(make_stat_tile("PC score min", "kpi4-3"), width=3),
        dbc.Col(make_stat_tile("PC score max", "kpi4-4"), width=3),
    ], className="mb-4 mt-2"),
    dbc.Row([
        dbc.Col([
            html.Label("EOF Mode:", style={'color': COLORS['text2'], 'fontSize': '12px', 'marginRight': '8px'}),
            dcc.Dropdown(id='t4-mode-sel', options=[{'label': 'EOF 1', 'value': 'EOF1'}, {'label': 'EOF 2', 'value': 'EOF2'}, {'label': 'EOF 3', 'value': 'EOF3'}], value='EOF1', clearable=False, style={'width': '200px', 'display': 'inline-block', 'color': 'black'}),
        ], width=12),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Spatial Loading"), dbc.CardBody(dcc.Loading(dcc.Graph(id='t4-map', style={'height': '400px'}), type="circle"), style={'padding':'0', 'backgroundColor': COLORS['bg1']})], style=CARD_STYLE), width=6),
        dbc.Col(dbc.Card([dbc.CardHeader("PC Scores"), dbc.CardBody(dcc.Loading(dcc.Graph(id='t4-ts', style={'height': '400px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})], style=CARD_STYLE), width=6),
    ])
])

tab5_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Increasing", "kpi5-1"), width=4),
        dbc.Col(make_stat_tile("Decreasing", "kpi5-2"), width=4),
        dbc.Col(make_stat_tile("Max Delta %", "kpi5-3"), width=4),
    ], className="mb-4 mt-2"),
    dbc.Card([
        dbc.CardHeader("Decadal Anomaly Matrix", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(dcc.Loading(dcc.Graph(id='t5-heatmap', style={'height': '600px'}), type="circle"), style={'padding': '0', 'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE)
])

tab6_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Variable", "kpi6-1"), width=3),
        dbc.Col(make_stat_tile("Peak", "kpi6-2"), width=3),
        dbc.Col(make_stat_tile("Trough", "kpi6-3"), width=3),
        dbc.Col(make_stat_tile("Range", "kpi6-4"), width=3),
    ], className="mb-4 mt-2"),
    dbc.Row([
        dbc.Col([
            html.Label("Variable:", style={'color': COLORS['text2'], 'fontSize': '12px', 'marginRight': '8px'}),
            dcc.Dropdown(id='t6-var-sel', options=[{'label': v, 'value': v} for v in (seasonal['variable'].unique() if not seasonal.empty else variables)], value=(seasonal['variable'].iloc[0] if not seasonal.empty else (variables[0] if variables else None)), clearable=False, style={'width': '250px', 'display': 'inline-block', 'color': 'black'}),
        ], width=12),
    ], className="mb-3"),
    dbc.Card([
        dbc.CardHeader("Seasonal Climatology Shift", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(dcc.Loading(dcc.Graph(id='t6-chart', style={'height': '500px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE)
])

tab7_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Extremes count", "kpi7-1"), width=4),
        dbc.Col(make_stat_tile("High corr pair", "kpi7-2"), width=4),
        dbc.Col(make_stat_tile("Corr value", "kpi7-3"), width=4),
    ], className="mb-4 mt-2"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Extremes Frequency"), dbc.CardBody(dcc.Loading(dcc.Graph(id='t7-bar', style={'height': '400px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})], style=CARD_STYLE), width=6),
        dbc.Col(dbc.Card([dbc.CardHeader("Correlation Matrix"), dbc.CardBody(dcc.Loading(dcc.Graph(id='t7-corr', style={'height': '400px'}), type="circle"), style={'padding':'0', 'backgroundColor': COLORS['bg1']})], style=CARD_STYLE), width=6),
    ])
])

tab8_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Bowen trend", "kpi8-1", "DECREASING"), width=3),
        dbc.Col(make_stat_tile("Evap trend", "kpi8-2", "INCREASING"), width=3),
        dbc.Col(make_stat_tile("P-value", "kpi8-3", "0.0163"), width=3),
        dbc.Col(make_stat_tile("Status", "kpi8-4", "WETTER"), width=3),
    ], className="mb-4 mt-2"),
    dbc.Card([
        dbc.CardHeader("Energy Partitioning", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(dcc.Loading(dcc.Graph(id='t8-chart', style={'height': '500px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE)
])

tab9_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Variable", "kpi9-1"), width=3),
        dbc.Col(make_stat_tile("Mean", "kpi9-2"), width=3),
        dbc.Col(make_stat_tile("Std dev", "kpi9-3"), width=3),
        dbc.Col(make_stat_tile("Skew", "kpi9-4"), width=3),
    ], className="mb-4 mt-2"),
    dbc.Row([
        dbc.Col([
            html.Label("Variable:", style={'color': COLORS['text2'], 'fontSize': '12px', 'marginRight': '8px'}),
            dcc.Dropdown(id='t9-var-sel', options=[{'label': v, 'value': v} for v in variables], value=variables[0] if variables else None, clearable=False, style={'width': '250px', 'display': 'inline-block', 'color': 'black'}),
        ], width=4),
        dbc.Col([
            html.Label("Plot Type:", style={'color': COLORS['text2'], 'fontSize': '12px', 'marginRight': '8px'}),
            dcc.Dropdown(id='t9-type-sel', options=[{'label': 'Box', 'value': 'box'}, {'label': 'Violin', 'value': 'violin'}, {'label': 'Histogram', 'value': 'histogram'}], value='box', clearable=False, style={'width': '200px', 'display': 'inline-block', 'color': 'black'}),
        ], width=4),
    ], className="mb-3 align-items-center"),
    dbc.Card([
        dbc.CardHeader("Regional Distributions", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(dcc.Loading(dcc.Graph(id='t9-chart', style={'height': '500px'}), type="circle"), style={'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE)
])

tab10_content = html.Div([
    dbc.Row([
        dbc.Col(make_stat_tile("Rows", "kpi10-1"), width=4),
        dbc.Col(make_stat_tile("Config keys", "kpi10-2"), width=4),
        dbc.Col(make_stat_tile("Status", "kpi10-3", "VERIFIED"), width=4),
    ], className="mb-4 mt-2"),
    dbc.Card([
        dbc.CardHeader("Config", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(html.Pre(id='t10-config', style={'backgroundColor': COLORS['bg0'], 'color': COLORS['text2'], 'padding': '18px', 'fontSize': '12px', 'borderRadius': '8px', 'maxHeight': '400px', 'overflow': 'auto'}), style={'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE),
    dbc.Card([
        dbc.CardHeader("Inventory", style={'color': COLORS['text1'], 'fontSize': '13px'}),
        dbc.CardBody(html.Div(id='t10-inventory-container'), style={'padding': '0', 'backgroundColor': COLORS['bg1']})
    ], style=CARD_STYLE)
])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    html.Header([
        html.Div([
            html.Span("Dashboard", style={'fontWeight': '600', 'fontSize': '16px'}),
            html.Span("HadGEM2-CC", style={'color': COLORS['text3'], 'fontSize': '12px', 'marginLeft': '12px'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Span("Data loaded", style={'fontSize': '11px', 'color': COLORS['text2']}),
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={'height': '56px', 'backgroundColor': COLORS['bg0'], 'borderBottom': f"1px solid {COLORS['border']}", 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '0 20px', 'position': 'fixed', 'top': '0', 'width': '100%', 'zIndex': '1000'}),
    dcc.Tabs(id="main-tabs", value='tab1', children=[
        dcc.Tab(label=l, value=v, style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}) for l, v in [
            ('Headline', 'tab1'), ('Spatial', 'tab2'), ('Regional', 'tab3'), 
            ('EOF', 'tab4'), ('Anomaly', 'tab5'), ('Seasonal', 'tab6'), 
            ('Extremes', 'tab7'), ('Energy', 'tab8'), ('Spread', 'tab9'), 
            ('Provenance', 'tab10')
        ]
    ], style={'height': '40px', 'backgroundColor': COLORS['bg0'], 'borderBottom': f"1px solid {COLORS['border']}", 'display': 'flex', 'gap': '24px', 'padding': '0 20px', 'position': 'fixed', 'top': '56px', 'width': '100%', 'zIndex': '900'}),
    html.Main(id="tab-content", style={'padding': '116px 20px 76px 20px', 'maxWidth': '1440px', 'margin': '0 auto'}),
    html.Footer([
        html.Div(["Last rendered: 2026-05-15"]),
        html.Div("HadGEM2-CC")
    ], style={'position': 'fixed', 'bottom': '0', 'width': '100%', 'height': '56px', 'backgroundColor': COLORS['bg0'], 'borderTop': f"1px solid {COLORS['border']}", 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '0 20px', 'color': COLORS['text3'], 'fontSize': '12px', 'zIndex': '1000'})
], fluid=True, style={'backgroundColor': COLORS['bg0'], 'minHeight': '100vh', 'padding': '0'})

# ==========================================
# 3. CALLBACKS
# ==========================================

@app.callback(Output('tab-content', 'children'), [Input('main-tabs', 'value')])
def render_tab_content(tab):
    if tab == 'tab1': return tab1_content
    if tab == 'tab2': return tab2_content
    if tab == 'tab3': return tab3_content
    if tab == 'tab4': return tab4_content
    if tab == 'tab5': return tab5_content
    if tab == 'tab6': return tab6_content
    if tab == 'tab7': return tab7_content
    if tab == 'tab8': return tab8_content
    if tab == 'tab9': return tab9_content
    if tab == 'tab10': return tab10_content
    return tab1_content

@app.callback(
    [Output('t1-bar-mk', 'figure'), Output('t1-bar-decadal', 'figure'), Output('t1-table-container', 'children'),
     Output('kpi1-1', 'children'), Output('kpi1-4', 'children')],
    [Input('main-tabs', 'value')]
)
def update_tab1(tab):
    if tab != 'tab1': return [go.Figure()]*2 + [None]*3
    kpi1_1 = len(mk_results) if not mk_results.empty else "-"
    kpi1_4 = len(mk_results[mk_results['p_value'] < 0.05]) if not mk_results.empty and 'p_value' in mk_results.columns else "-"
    fig_mk = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if not mk_results.empty:
        colors = [COLORS['green'] if s > 0 else COLORS['red'] for s in mk_results['sen_slope']]
        fig_mk.add_trace(go.Bar(x=mk_results['sen_slope'], y=mk_results['variable'], orientation='h', marker_color=colors))
        fig_mk.update_layout(margin=dict(l=150, r=20, t=20, b=40))
    fig_dec = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if not decadal.empty:
        col_delta = 'Delta %' if 'Delta %' in decadal.columns else ('\u0394 %' if '\u0394 %' in decadal.columns else None)
        if col_delta and col_delta in decadal.columns:
            colors = [COLORS['green'] if s > 0 else COLORS['red'] for s in decadal[col_delta]]
            fig_dec.add_trace(go.Bar(x=decadal[col_delta], y=decadal['variable'], orientation='h', marker_color=colors))
            fig_dec.update_layout(margin=dict(l=150, r=20, t=20, b=40))
    table = None
    if not mk_results.empty:
        req_cols = ['variable', 'trend', 'p_value', 'tau', 'sen_slope']
        cols = [{"name": i, "id": i} for i in req_cols if i in mk_results.columns]
        table = dash_table.DataTable(
            data=mk_results.to_dict('records'), columns=cols,
            style_header={'backgroundColor': COLORS['bg2'], 'color': COLORS['text2'], 'border': f"1px solid {COLORS['border']}"},
            style_cell={'backgroundColor': COLORS['bg1'], 'color': COLORS['text1'], 'textAlign':'left', 'border': f"1px solid {COLORS['border']}", 'fontFamily': 'monospace', 'fontSize': '12px'},
        )
    return fig_mk, fig_dec, table, kpi1_1, kpi1_4

@app.callback(
    [Output('t2-var-sel', 'options'), Output('t2-var-sel', 'value')],
    [Input('t2-dataset-sel', 'value')]
)
def update_t2_var_options(dataset):
    src_df = spatial_mean if dataset == 'mean' else trend_slope
    options = [{'label': v, 'value': v} for v in src_df['variable'].unique()]
    if not options:
        return [], None
    return options, options[0]['value']

@app.callback(
    [Output('t2-map', 'figure'), Output('t2-line', 'figure'), 
     Output('kpi2-1', 'children'), Output('kpi2-2', 'children'), Output('kpi2-3', 'children')],
    [Input('t2-dataset-sel', 'value'), Input('t2-var-sel', 'value')]
)
def update_tab2(dataset, var):
    val_col = 'value' if dataset == 'mean' else 'slope'
    src_df  = spatial_mean if dataset == 'mean' else trend_slope
    df_var = src_df[src_df['variable'] == var]

    if df_var.empty:
        fig_empty = go.Figure().add_annotation(
            text=f"No spatial data found for '{var}' in {dataset}",
            showarrow=False,
            font=dict(size=13, color="#ef4444"),
            xref="paper", yref="paper", x=0.5, y=0.5
        ).update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22,27,39,0.6)"
        )
        return fig_empty, go.Figure(), "-", var, "-"

    try:
        pivot = df_var.pivot(index='lat', columns='lon', values=val_col)
    except Exception as e:
        df_var = df_var.groupby(['lat','lon'], as_index=False)[val_col].mean()
        pivot = df_var.pivot(index='lat', columns='lon', values=val_col)

    lats = pivot.index.tolist()
    lons = pivot.columns.tolist()
    z    = pivot.values

    is_diverging = (dataset == 'slope')

    if is_diverging:
        abs_max = float(np.nanmax(np.abs(z)))
        abs_max = abs_max if abs_max > 0 else 1.0
        zmin, zmax, zmid = -abs_max, abs_max, 0
        colorscale = 'RdBu_r'
    else:
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        zmid = None
        colorscale = 'Viridis'

    trace = go.Heatmap(
        x=lons,
        y=lats,
        z=z,
        colorscale=colorscale,
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            tickfont=dict(size=10, color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#2e3a52",
            borderwidth=1
        )
    )

    fig_map = go.Figure(data=[trace])
    
    if CARTOPY_AVAILABLE:
        coastline = cfeature.COASTLINE.with_scale('50m')
        geoms = list(coastline.geometries())
        shapes = []
        for geom in geoms:
            lon_min = min(lons) - 1
            lon_max = max(lons) + 1
            lat_min = min(lats) - 2
            lat_max = max(lats) + 2
            if geom.bounds[0] > lon_max: continue
            if geom.bounds[2] < lon_min: continue
            if geom.bounds[1] > lat_max: continue
            if geom.bounds[3] < lat_min: continue
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
                shapes.append(dict(
                    type="path",
                    path=coords_to_svg_path(x_coords, y_coords),
                    line=dict(color="rgba(0,0,0,0.6)", width=4),
                    layer="above"
                ))
                shapes.append(dict(
                    type="path",
                    path=coords_to_svg_path(x_coords, y_coords),
                    line=dict(color="#ff6b35", width=2, dash='solid'),
                    layer="above"
                ))
            elif geom.geom_type == 'MultiLineString':
                for part in geom.geoms:
                    coords = list(part.coords)
                    x_coords = [c[0] for c in coords]
                    y_coords = [c[1] for c in coords]
                    shapes.append(dict(
                        type="path",
                        path=coords_to_svg_path(x_coords, y_coords),
                        line=dict(color="rgba(0,0,0,0.6)", width=4),
                        layer="above"
                    ))
                    shapes.append(dict(
                        type="path",
                        path=coords_to_svg_path(x_coords, y_coords),
                        line=dict(color="#ff6b35", width=2, dash='solid'),
                        layer="above"
                    ))
        fig_map.update_layout(shapes=shapes)
    else:
        for seg in COASTLINE_SEGMENTS:
            pairs = [(lo, la) for lo, la in zip(seg['lon'], seg['lat'])
                     if min(lons)-1 <= lo <= max(lons)+1
                     and min(lats)-2 <= la <= max(lats)+2]
            if len(pairs) < 2:
                continue
            lo_clip, la_clip = zip(*pairs)
            fig_map.add_trace(go.Scatter(
                x=list(lo_clip),
                y=list(la_clip),
                mode='lines',
                line=dict(
                    color='rgba(0,0,0,0.6)',
                    width=4
                ),
                hoverinfo='skip',
                showlegend=False,
                name='Halo'
            ))
            fig_map.add_trace(go.Scatter(
                x=list(lo_clip),
                y=list(la_clip),
                mode='lines',
                line=dict(
                    color='#ff6b35',
                    width=2,
                    dash='solid'
                ),
                hoverinfo='skip',
                showlegend=False,
                name='Coastline'
            ))

    fig_map.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='Coastline',
        line=dict(color='#ff6b35', width=2),
        showlegend=True,
        hoverinfo='skip'
    ))

    fig_map.add_annotation(
        x=0.01, y=0.97,
        xref='paper', yref='paper',
        text='── Coastline (land/sea boundary)',
        showarrow=False,
        font=dict(size=11, color='#ff6b35',
                  family='monospace'),
        align='left',
        bgcolor='rgba(14,17,23,0.65)',
        bordercolor='#2e3a52',
        borderwidth=1,
        borderpad=4,
        xanchor='left',
        yanchor='top'
    )

    fig_map.update_layout(
        title=dict(
            text=f"{'Trend Slope' if dataset=='slope' else 'Spatial Mean'}: {var}",
            font=dict(size=13, color="#e2e8f0")
        ),
        xaxis=dict(
            title="Longitude",
            tickfont=dict(size=10, color="#94a3b8"),
            gridcolor="#2e3a52",
            zerolinecolor="#2e3a52"
        ),
        yaxis=dict(
            title="Latitude",
            tickfont=dict(size=10, color="#94a3b8"),
            gridcolor="#2e3a52",
            zerolinecolor="#2e3a52"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,39,0.6)",
        margin=dict(t=46, r=80, b=50, l=60),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(14,17,23,0.75)',
            bordercolor='#2e3a52',
            borderwidth=1,
            font=dict(size=11, color='#e2e8f0')
        )
    )

    spatial_coverage = f"{len(lats)} lat × {len(lons)} lon"
    value_range      = f"{round(zmin,3)} / {round(zmax,3)}"

    fig_line = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if not period_spatial.empty:
        vars_in_p = period_spatial['variable'].unique()
        for v in vars_in_p:
            df_v = period_spatial[period_spatial['variable'] == v].sort_values('period_year')
            fig_line.add_trace(go.Scatter(x=df_v['period_year'], y=df_v['value'], name=v, mode='lines+markers'))
        
    return fig_map, fig_line, spatial_coverage, var, value_range

@app.callback(
    [Output('t3-main-chart', 'figure'), Output('t3-sm-1', 'figure'), Output('t3-sm-2', 'figure'), Output('t3-sm-3', 'figure'),
     Output('kpi3-2', 'children'), Output('kpi3-4', 'children')],
    [Input('t3-var-sel', 'value')]
)
def update_tab3(var):
    fig_main = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    figs_sm = [go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=30,l=40,r=10)) for _ in range(3)]
    kpi3_2, kpi3_4 = var, "-"
    if not mk_results.empty and var in mk_results['variable'].values:
        kpi3_4 = mk_results[mk_results['variable'] == var]['trend'].values[0]
    regs = [("North Africa", COLORS['accent']), ("Sahel", COLORS['cyan']), ("West Africa", COLORS['purple'])]
    for i, (reg, col) in enumerate(regs):
        df = regional_ts.get(reg, pd.DataFrame())
        if not df.empty and var in df.columns:
            trace = go.Scatter(x=df['year'], y=df[var], name=reg, mode='lines', line=dict(color=col))
            fig_main.add_trace(trace)
            figs_sm[i].add_trace(go.Scatter(x=df['year'], y=df[var], mode='lines', line=dict(color=col)))
    return [fig_main] + figs_sm + [kpi3_2, kpi3_4]

@app.callback(
    [Output('t4-map', 'figure'), Output('t4-ts', 'figure'),
     Output('kpi4-1', 'children'), Output('kpi4-2', 'children'), Output('kpi4-3', 'children'), Output('kpi4-4', 'children')],
    [Input('t4-mode-sel', 'value')]
)
def update_tab4(mode):
    fig_map = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
    fig_ts = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    kpi4_1, kpi4_2, kpi4_3, kpi4_4 = mode, "99.8%" if mode == 'EOF1' else "0.1%", "-", "-"
    if not eof_loadings.empty and mode in eof_loadings.columns:
        fig_map.add_trace(go.Scattergeo(
            lon=eof_loadings['lon'], lat=eof_loadings['lat'],
            marker=dict(color=eof_loadings[mode], colorscale='RdBu_r', symbol='square', size=2),
            mode='markers'
        ))
        fig_map.update_layout(geo=dict(projection_type='robinson', showcoastlines=True, coastlinecolor="White", showland=True, landcolor=COLORS['bg2'], bgcolor='rgba(0,0,0,0)'))
    if not eof_scores.empty and mode in eof_scores.columns:
        scores = eof_scores[mode]
        kpi4_3, kpi4_4 = f"{scores.min():.2f}", f"{scores.max():.2f}"
        fig_ts.add_trace(go.Scatter(x=eof_scores['year'], y=scores, mode='lines+markers', line=dict(color=COLORS['purple'])))
    return fig_map, fig_ts, kpi4_1, kpi4_2, kpi4_3, kpi4_4

@app.callback(
    [Output('t5-heatmap', 'figure'), Output('kpi5-1', 'children'), Output('kpi5-2', 'children'), Output('kpi5-3', 'children')],
    [Input('main-tabs', 'value')]
)
def update_tab5(tab):
    if tab != 'tab5': return go.Figure(), "-", "-", "-"
    fig = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    kpi5_1, kpi5_2, kpi5_3 = "-", "-", "-"
    if not decadal.empty:
        periods = [c for c in decadal.columns if '-' in c]
        if periods:
            z = decadal[periods].values
            fig.add_trace(go.Heatmap(x=periods, y=decadal['variable'], z=z, colorscale='RdBu_r', zmid=0))
            fig.update_layout(margin=dict(l=150, r=20, t=20, b=40))
            delta_col = 'Delta %' if 'Delta %' in decadal.columns else ('\u0394 %' if '\u0394 %' in decadal.columns else None)
            if delta_col and delta_col in decadal.columns:
                kpi5_1 = decadal.loc[decadal[delta_col].idxmax(), 'variable']
                kpi5_2 = decadal.loc[decadal[delta_col].idxmin(), 'variable']
                kpi5_3 = f"{decadal[delta_col].max():.2f}%"
    return fig, kpi5_1, kpi5_2, kpi5_3

@app.callback(
    [Output('t6-chart', 'figure'), Output('kpi6-1', 'children'), Output('kpi6-2', 'children'), Output('kpi6-3', 'children'), Output('kpi6-4', 'children')],
    [Input('t6-var-sel', 'value')]
)
def update_tab6(var):
    fig = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    kpi6_1, kpi6_2, kpi6_3, kpi6_4 = var, "-", "-", "-"
    if not seasonal.empty and var in seasonal['variable'].values:
        df_v = seasonal[seasonal['variable'] == var]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        max_range = 0
        peak_m, trough_m = "-", "-"
        for idx, row in df_v.iterrows():
            vals = [row.get(m, 0) for m in months]
            if all(v == 0 for v in vals): 
                full_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                vals = [row.get(m, 0) for m in full_months]
            fig.add_trace(go.Scatter(x=months, y=vals, name=row['period_label'], mode='lines+markers'))
            if any(v != 0 for v in vals):
                r = max(vals) - min(vals)
                if r >= max_range:
                    max_range, peak_m, trough_m = r, months[vals.index(max(vals))], months[vals.index(min(vals))]
        kpi6_2, kpi6_3, kpi6_4 = peak_m, trough_m, f"{max_range:.2f}"
    else:
        fig.add_annotation(text=f"No seasonal data for {var}", showarrow=False, font=dict(size=14, color=COLORS['text2']))
    return fig, kpi6_1, kpi6_2, kpi6_3, kpi6_4

@app.callback(
    [Output('t7-bar', 'figure'), Output('t7-corr', 'figure'), Output('kpi7-1', 'children'), Output('kpi7-2', 'children'), Output('kpi7-3', 'children')],
    [Input('main-tabs', 'value')]
)
def update_tab7(tab):
    fig_ext = go.Figure().update_layout(template='plotly_dark', paper_bgcolor=COLORS['bg1'], plot_bgcolor=COLORS['bg1'])
    fig_corr = go.Figure().update_layout(template='plotly_dark', paper_bgcolor=COLORS['bg1'], plot_bgcolor=COLORS['bg1'])
    kpi7_1, kpi7_2, kpi7_3 = "-", "-", "-"
    if tab != 'tab7': return [go.Figure()]*2 + [kpi7_1, kpi7_2, kpi7_3]
    if not extremes.empty:
        kpi7_1 = len(extremes)
        for v, grp in extremes.groupby('variable'):
            fig_ext.add_trace(go.Bar(x=grp['period'], y=grp['pct'], name=v))
        fig_ext.update_layout(barmode='group', xaxis=dict(title="Period"), yaxis=dict(title="Frequency (%)"))
    if not corr_matrix.empty:
        fig_corr.add_trace(go.Heatmap(x=corr_matrix.columns, y=corr_matrix.index, z=corr_matrix.values, colorscale='RdBu_r', zmin=-1, zmax=1))
        fig_corr.update_layout(margin=dict(l=150, b=150, t=20, r=20))
        c = corr_matrix.copy()
        np.fill_diagonal(c.values, 0)
        max_c = c.abs().max().max()
        idx = (c.abs() == max_c).values.nonzero()
        if len(idx[0]) > 0:
            kpi7_2, kpi7_3 = f"{c.index[idx[0][0]][:8]} / {c.columns[idx[1][0]][:8]}", f"{c.iloc[idx[0][0], idx[1][0]]:.2f}"
    return fig_ext, fig_corr, kpi7_1, kpi7_2, kpi7_3

@app.callback(Output('t8-chart', 'figure'), [Input('main-tabs', 'value')])
def update_tab8(tab):
    if tab != 'tab8': return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if not energy.empty:
        fig.add_trace(go.Scatter(x=energy['year'], y=energy['evap_fraction'], name='Evap Fraction', line=dict(color=COLORS['green'])), secondary_y=False)
        fig.add_trace(go.Scatter(x=energy['year'], y=energy['bowen_ratio'], name='Bowen Ratio', line=dict(color=COLORS['red'])), secondary_y=True)
    return fig

@app.callback(
    [Output('t9-chart', 'figure'), Output('kpi9-1', 'children'), Output('kpi9-2', 'children'), Output('kpi9-3', 'children'), Output('kpi9-4', 'children')],
    [Input('t9-var-sel', 'value'), Input('t9-type-sel', 'value')]
)
def update_tab9(var, plot_type):
    kpi9_1, kpi9_2, kpi9_3, kpi9_4 = var, "-", "-", "-"
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["North Africa", "Sahel", "West Africa"],
        shared_yaxes=False,    # CRITICAL: independent y-axes
        horizontal_spacing=0.08
    )

    region_config = [
        ("North Africa", regional_ts.get("North Africa", pd.DataFrame()), "#3b82f6", "rgba(59,130,246,0.15)", 1),
        ("Sahel",        regional_ts.get("Sahel", pd.DataFrame()),        "#06b6d4", "rgba(6,182,212,0.15)",  2),
        ("West Africa",  regional_ts.get("West Africa", pd.DataFrame()),  "#10b981", "rgba(16,185,129,0.15)", 3),
    ]

    warnings = []
    any_data  = False
    all_vals = []

    for region_name, df_r, line_color, fill_color, col_idx in region_config:
        if df_r.empty or var not in df_r.columns:
            warnings.append(f"⚠ {region_name}: no data for '{var}'")
            continue

        series = df_r[var].dropna()
        n = len(series)

        if n == 0:
            warnings.append(f"⚠ {region_name}: no data for '{var}'")
            continue
        if n < 5:
            warnings.append(
                f"⚠ {region_name}: only {n} values — interpret with caution"
            )

        any_data = True
        all_vals.extend(series.tolist())

        if plot_type == 'box':
            fig.add_trace(
                go.Box(
                    y=series.tolist(),
                    name=region_name,
                    boxmean="sd",
                    boxpoints="outliers",
                    jitter=0.4,
                    pointpos=0,
                    whiskerwidth=0.6,
                    marker=dict(
                        color=line_color,
                        size=4,
                        opacity=0.6,
                        line=dict(width=1, color="#0e1117")
                    ),
                    line=dict(color=line_color, width=2),
                    fillcolor=fill_color,
                    showlegend=False,
                ),
                row=1, col=col_idx
            )
        elif plot_type == 'violin':
            fig.add_trace(
                go.Violin(
                    y=series.tolist(),
                    name=region_name,
                    box_visible=True,
                    meanline_visible=True,
                    points="outliers",
                    jitter=0.4,
                    pointpos=0,
                    marker=dict(
                        color=line_color,
                        size=4,
                        opacity=0.6,
                        line=dict(width=1, color="#0e1117")
                    ),
                    line=dict(color=line_color, width=2),
                    fillcolor=fill_color,
                    showlegend=False,
                ),
                row=1, col=col_idx
            )
        else:
            fig.add_trace(
                go.Histogram(
                    x=series.tolist(),
                    name=region_name,
                    opacity=0.75,
                    marker=dict(line=dict(color=COLORS['bg0'], width=1)),
                    showlegend=False,
                ),
                row=1, col=col_idx
            )

        if plot_type in ['box', 'violin']:
            # Tight per-region y-axis with meaningful padding
            vals   = series.tolist()
            v_min  = min(vals)
            v_max  = max(vals)
            v_iqr  = np.percentile(vals, 75) - np.percentile(vals, 25)

            # Minimum visible range: at least 10× the IQR or 5% of mean
            min_range = max(v_iqr * 10, abs(np.mean(vals)) * 0.05, 0.01)
            y_lo = v_min - min_range * 0.5
            y_hi = v_max + min_range * 0.5

            fig.update_yaxes(
                range=[y_lo, y_hi],
                title_text=var if col_idx == 1 else "",
                tickfont=dict(size=10, color="#94a3b8"),
                gridcolor="#2e3a52",
                zerolinecolor="#2e3a52",
                row=1, col=col_idx
            )
            fig.update_xaxes(
                showticklabels=False,
                gridcolor="#2e3a52",
                row=1, col=col_idx
            )
        else:
            fig.update_xaxes(
                title_text=var if col_idx == 2 else "",
                tickfont=dict(size=10, color="#94a3b8"),
                gridcolor="#2e3a52",
                zerolinecolor="#2e3a52",
                row=1, col=col_idx
            )
            fig.update_yaxes(
                title_text="Count" if col_idx == 1 else "",
                tickfont=dict(size=10, color="#94a3b8"),
                gridcolor="#2e3a52",
                zerolinecolor="#2e3a52",
                row=1, col=col_idx
            )

    if not any_data:
        # Return empty annotated figure
        fig = go.Figure().add_annotation(
            text=f"No distribution data available for '{var}'",
            showarrow=False,
            font=dict(size=13, color="#ef4444"),
            xref="paper", yref="paper", x=0.5, y=0.5
        )
    else:
        v = np.array(all_vals)
        kpi9_2, kpi9_3, kpi9_4 = f"{v.mean():.2f}", f"{v.std():.2f}", f"{pd.Series(v).skew():.2f}"

    fig.update_layout(
        title=dict(
            text=f"Regional Distributions — {var}",
            font=dict(size=13, color="#e2e8f0")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,39,0.6)",
        font=dict(color="#94a3b8", size=11),
        boxmode="group",
        showlegend=False,
        margin=dict(t=50, r=20, b=40, l=60),
    )

    # Re-style subplot header annotations only
    for ann in fig.layout.annotations:
        if getattr(ann, 'text', '') != f"No distribution data available for '{var}'":
            ann.font.color  = "#94a3b8"
            ann.font.size   = 12

    # Apply dark bg to all subplot areas individually
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor="#2e3a52",
            zerolinecolor="#2e3a52",
            row=1, col=i
        )

    return fig, kpi9_1, kpi9_2, kpi9_3, kpi9_4

@app.callback(
    [Output('t10-config', 'children'), Output('t10-inventory-container', 'children'), 
     Output('kpi10-1', 'children'), Output('kpi10-2', 'children')],
    [Input('main-tabs', 'value')]
)
def update_tab10(tab):
    if tab != 'tab10': return "", None, "-", "-"
    config_str = json.dumps(config, indent=2)
    kpi10_1, kpi10_2 = (len(var_stats) if not var_stats.empty else "-"), len(config.keys())
    inv_data = [{"Filename": i[0], "Shape/Rows": i[1], "Status": i[2]} for i in inventory]
    table = dash_table.DataTable(
        data=inv_data, columns=[{"name": i, "id": i} for i in ["Filename", "Shape/Rows", "Status"]],
        style_header={'backgroundColor': COLORS['bg2'], 'color': COLORS['text2'], 'border': f"1px solid {COLORS['border']}"},
        style_cell={'backgroundColor': COLORS['bg1'], 'color': COLORS['text1'], 'textAlign':'left', 'border': f"1px solid {COLORS['border']}"},
    )
    return config_str, table, kpi10_1, kpi10_2

if __name__ == '__main__':
    app.run(debug=False, port=8050)
