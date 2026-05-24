import os
import json
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import xarray as xr
import numpy as np

# ==========================================
# 1. LOAD DATA & INVENTORY (MODULE LEVEL)
# ==========================================
inventory = []

def load_json(filename):
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        inventory.append((filename, f"{len(data)} keys", "v"))
        return data
    else:
        inventory.append((filename, "MISSING", "x MISSING"))
        return {}


def load_csv(filename, index_col=None):
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=index_col)
        # Check rows/shape
        inventory.append((filename, f"{df.shape[0]} rows", "v"))
        return df
    else:
        inventory.append((filename, "MISSING", "x MISSING"))
        return pd.DataFrame()

def load_nc(filename):
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        ds = xr.open_dataset(path)
        shape_str = f"({','.join(map(str, ds.dims.values()))}) x{len(ds.data_vars)}v"
        inventory.append((filename, shape_str, "v"))
        return ds
    else:
        inventory.append((filename, "MISSING", "x MISSING"))
        return None

print(f"{'FILE':<30} | {'ROWS/SHAPE':<15} | {'STATUS'}")
print("-" * 30 + "|" + "-" * 17 + "|" + "-" * 7)

ds_anomaly = load_nc("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\ds_anomaly.nc")
eof_loadings = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\eof_spatial_loadings.csv")
eof_scores = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\eof_pc_scores.csv")
var_stats = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\variable_statistics.csv")
mk_results = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\mann_kendall_results.csv")
decadal = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\decadal_changes.csv")
reg_na = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\regional_ts_north_africa.csv")
reg_sa = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\regional_ts_sahel.csv")
reg_wa = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\regional_ts_west_africa.csv")
extremes = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\extreme_events.csv")
corr_matrix = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\correlation_matrix.csv", index_col=0)
seasonal = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\seasonal_climatology.csv")
energy = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\energy_balance.csv")

# New explicit loads to fulfill all requests:
spatial_mean_maps = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\spatial_mean_maps.csv")
period_spatial = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\period_spatial_means.csv")
trend_slope_maps = load_csv("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\trend_slope_maps.csv")
config_data = load_json("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\config.json")
repro_data = load_json("C:\\Users\\MSI\\Desktop\\projet tej\\outputs\\reproducibility_report.json")

for item in inventory:
    status_icon = "✓" if item[2] == "v" else "✗ MISSING"
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

# ==========================================
# 2. APP INITIALIZATION & LAYOUT
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

tab1_content = html.Div([
    html.H3("🗺️ Regridded Maps Workspace"),
    dbc.Row([
        dbc.Col([
            html.Label("Variable:"),
            dcc.Dropdown(id='t1-var', options=[{'label': v, 'value': v} for v in variables], value=variables[0] if variables else None, clearable=False, style={'color':'black'}),
        ], width=4),
        dbc.Col([
            html.Label("Time Index (from ds_anomaly.nc):"),
            dcc.Slider(
                id='t1-time', 
                min=0, 
                max=(len(ds_anomaly.time)-1) if ds_anomaly is not None and 'time' in ds_anomaly.dims else 10, 
                step=1, 
                value=0,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=8)
    ], className="mb-4 mt-2"),
    dcc.Graph(id='t1-map', style={'height': '70vh'})
])

tab2_content = html.Div([
    html.H3("📈 Regional Time Series"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='t2-reg', options=[{'label': r, 'value': r} for r in regional_ts.keys()], value='Sahel', clearable=False, style={'color':'black'}),
        ], width=4),
        dbc.Col([
            dcc.Dropdown(id='t2-var', options=[{'label': v, 'value': v} for v in variables], value=variables[0] if variables else None, clearable=False, style={'color':'black'}),
        ], width=4)
    ]),
    dcc.Graph(id='t2-ts-plot')
])

tab3_content = html.Div([
    html.H3("🌊 EOF / PCA Analysis"),
    dbc.Card(dbc.CardBody(
        html.H5("Variance Explained: EOF1=99.8%, EOF2=0.1%, EOF3=0.1%", className="text-center m-0")
    ), className="mb-4"),
    dcc.RadioItems(id='t3-mode', options=[
        {'label': ' EOF1  ', 'value': 'EOF1'},
        {'label': ' EOF2  ', 'value': 'EOF2'},
        {'label': ' EOF3  ', 'value': 'EOF3'}
    ], value='EOF1', inline=True, style={'fontSize':'1.2rem', 'marginBottom':'20px'}),
    dbc.Row([
        dbc.Col(dcc.Graph(id='t3-spatial'), width=6),
        dbc.Col(dcc.Graph(id='t3-time'), width=6)
    ])
])

tab4_content = html.Div([
    html.H3("📊 Statistics & Trends"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='t4-bar'), width=5),
        dbc.Col(dcc.Graph(id='t4-heatmap'), width=7)
    ]),
    html.Hr(),
    html.H4("Mann-Kendall Results", style={'marginTop': '20px', 'marginBottom': '10px'}),
    html.Div(id='t4-table-container')
])

# Tab 5
monthly_vars = [
    'cloud-cover_monthly-mean',
    'temperature_monthly-mean',
    'water-vapor-pressure_monthly-mean',
    'wind-speed_monthly-mean'
]
tab5_content = html.Div([
    html.H3("🌡️ Seasonal Climatology"),
    dbc.Row(dbc.Col(
        dcc.Dropdown(id='t5-var', options=[{'label': v, 'value': v} for v in monthly_vars], value=monthly_vars[0], clearable=False, style={'color':'black'}, className="mb-4"),
        width=4
    )),
    dcc.Graph(id='t5-plot')
])

tab6_content = html.Div([
    html.H3("⚡ Extremes & Correlation"),
    dbc.Row([
        dbc.Col([
            html.Label("Filter Variables (Extremes):"),
            dcc.Dropdown(id='t6-var-filter', options=[{'label': v, 'value': v} for v in variables], multi=True, placeholder="All Variables", style={'color':'black'}, className="mb-4"),
            dcc.Graph(id='t6-extremes')
        ], width=6),
        dbc.Col([
            html.Label("Inter-variable Pearson Correlation"),
            dcc.Graph(id='t6-corr')
        ], width=6)
    ])
])

tab7_content = html.Div([
    html.H3("⚖️ Energy Balance"),
    html.H5("bowen_ratio: DECREASING, p=0.0163, slope=-0.0057/period | evap_fraction: INCREASING, p=0.0163, slope=+0.0025/period", style={'color': '#bbb', 'marginBottom': '20px'}),
    dcc.Graph(id='t7-eb-plot', style={'height': '70vh'})
])

# New Tabs for all CSVs and JSONs (Boxplots, Histograms, Regrid, Courbes)
tab8_content = html.Div([
    html.H3("📦 Distributions (Boxplots & Histograms)"),
    dbc.Row([
        dbc.Col([
            html.Label("Variable:"),
            dcc.Dropdown(id='t8-var', options=[{'label': v, 'value': v} for v in variables], value=variables[0] if variables else None, clearable=False, style={'color':'black'}),
        ], width=4),
        dbc.Col([
            html.Label("Graph Type:"),
            dcc.RadioItems(id='t8-type', options=[
                {'label': ' Histogram  ', 'value': 'hist'},
                {'label': ' Boxplot  ', 'value': 'box'}
            ], value='hist', inline=True, style={'fontSize':'1.2rem', 'marginLeft':'10px'}),
        ], width=4)
    ], className="mb-4 mt-2"),
    dcc.Graph(id='t8-plot', style={'height': '70vh'})
])

tab9_content = html.Div([
    html.H3("🗺️ Static Regrid & Trends"),
    dbc.Row([
        dbc.Col([
            html.Label("Dataset:"),
            dcc.Dropdown(id='t9-dataset', options=[
                {'label': 'Spatial Mean Maps', 'value': 'mean'},
                {'label': 'Trend Slope Maps', 'value': 'slope'}
            ], value='mean', clearable=False, style={'color':'black'})
        ], width=4),
        dbc.Col([
            html.Label("Variable:"),
            dcc.Dropdown(id='t9-var', options=[], value=None, clearable=False, style={'color':'black'})
        ], width=4)
    ], className="mb-4 mt-2"),
    dcc.Graph(id='t9-plot', style={'height': '70vh'})
])

tab10_content = html.Div([
    html.H3("📈 Variables Stats & JSON Metadata"),
    html.H4("Courbes: Period Spatial Means"),
    dcc.Dropdown(id='t10-var', options=[], value=None, clearable=False, style={'color':'black', 'width':'40%', 'marginBottom':'15px'}),
    dcc.Graph(id='t10-plot'),
    html.Hr(),
    html.H4("Variable Statistics"),
    html.Div(id='t10-stats-table'),
    html.Hr(),
    html.H4("Config & Reproducibility JSONs"),
    dbc.Row([
        dbc.Col(html.Pre(json.dumps(config_data, indent=2), style={'backgroundColor': '#222', 'color': '#0f0', 'padding': '10px', 'maxHeight': '400px', 'overflowY': 'auto'})),
        dbc.Col(html.Pre(json.dumps(repro_data, indent=2), style={'backgroundColor': '#222', 'color': '#0ff', 'padding': '10px', 'maxHeight': '400px', 'overflowY': 'auto'}))
    ])
])

app.layout = dbc.Container([
    html.H2("Surface Energy Balance Dashboard / HadGEM2-CC", className="text-center my-4"),
    dcc.Tabs([
        dcc.Tab(label='🗺️ Regridded Maps', children=tab1_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='📈 Regional Time Series', children=tab2_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='🌊 EOF / PCA Analysis', children=tab3_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='📊 Statistics & Trends', children=tab4_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='🌡️ Seasonal Climatology', children=tab5_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='⚡ Extremes & Correlation', children=tab6_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='⚖️ Energy Balance', children=tab7_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='📦 Distributions', children=tab8_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='🗺️ Static Regrid & Trends', children=tab9_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
        dcc.Tab(label='📈 Stats & Meta', children=tab10_content, style={'backgroundColor':'#333'}, selected_style={'backgroundColor':'#119DFF','color':'white'}),
    ], className="mb-4")
], fluid=True, style={'backgroundColor': '#222', 'color': 'white', 'minHeight': '100vh', 'paddingBottom': '50px'})

# ==========================================
# 3. CALLBACKS
# ==========================================

@app.callback(Output('t1-map', 'figure'), [Input('t1-var', 'value'), Input('t1-time', 'value')])
def update_t1(var, t_idx):
    if ds_anomaly is None or var not in ds_anomaly.data_vars:
        return go.Figure().add_annotation(text="Data Missing ds_anomaly.nc", showarrow=False, font=dict(size=20, color="white")).update_layout(template='plotly_dark')
    try:
        data = ds_anomaly[var].isel(time=int(t_idx))
        fig = go.Figure(go.Heatmap(z=data.values, x=ds_anomaly.get('lon', ds_anomaly.get('longitude', [])), y=ds_anomaly.get('lat', ds_anomaly.get('latitude', [])), colorscale='RdBu_r'))
        fig.update_layout(title=f"Anomaly: {var} (Time Index {t_idx})", template='plotly_dark')
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=str(e), showarrow=False, font=dict(size=14, color="white")).update_layout(template='plotly_dark')

@app.callback(Output('t2-ts-plot', 'figure'), [Input('t2-reg', 'value'), Input('t2-var', 'value')])
def update_t2(reg, var):
    df = regional_ts.get(reg, pd.DataFrame())
    if df.empty or var not in df.columns:
        return go.Figure().add_annotation(text=f"regional_ts_{reg.lower().replace(' ', '_')}.csv Missing or empty", showarrow=False, font=dict(size=16, color="white")).update_layout(template='plotly_dark')
    
    x_col = 'year' if 'year' in df.columns else df.columns[0]
    fig = go.Figure(go.Scatter(x=df[x_col], y=df[var], mode='lines+markers', name=var, line=dict(color='#00ffcc', width=2)))
    fig.update_layout(title=f"Regional Time Series - {var} ({reg})", template='plotly_dark', xaxis_title=x_col.capitalize(), yaxis_title=var)
    return fig

@app.callback(
    [Output('t3-spatial', 'figure'), Output('t3-time', 'figure')],
    [Input('t3-mode', 'value')]
)
def update_t3(mode):
    fig_sp = go.Figure().update_layout(template='plotly_dark')
    fig_ts = go.Figure().update_layout(template='plotly_dark')
    
    if eof_loadings.empty:
        fig_sp.add_annotation(text="eof_spatial_loadings.csv MISSING", showarrow=False, font=dict(size=16, color="red"))
    elif mode in eof_loadings.columns and 'lat' in eof_loadings.columns and 'lon' in eof_loadings.columns:
        df_pivot = eof_loadings.pivot(index='lat', columns='lon', values=mode)
        zmax = np.nanmax(np.abs(df_pivot.values)) if not np.isnan(df_pivot.values).all() else 1
        fig_sp.add_trace(go.Heatmap(
            x=df_pivot.columns, y=df_pivot.index, z=df_pivot.values, 
            colorscale='RdBu_r', zmid=0, zmin=-zmax, zmax=zmax
        ))
        fig_sp.update_layout(title=f"Spatial Loading - {mode}", xaxis_title="Lon", yaxis_title="Lat")
        
    if eof_scores.empty:
        fig_ts.add_annotation(text="eof_pc_scores.csv MISSING", showarrow=False, font=dict(size=16, color="red"))
    elif mode in eof_scores.columns and 'year' in eof_scores.columns:
        fig_ts.add_trace(go.Scatter(x=eof_scores['year'], y=eof_scores[mode], mode='lines+markers', name=mode, line=dict(color='#ff9900')))
        fig_ts.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_ts.update_layout(title=f"PC Scores - {mode}", xaxis_title="Year", yaxis_title="Score")
        
    return fig_sp, fig_ts

@app.callback(
    [Output('t4-bar', 'figure'), Output('t4-heatmap', 'figure'), Output('t4-table-container', 'children')]
)
def update_t4():
    fig_bar = go.Figure().update_layout(template='plotly_dark')
    fig_hm = go.Figure().update_layout(template='plotly_dark')
    table = html.Div([html.P("mann_kendall_results.csv MISSING", className="text-danger")])
    
    if not decadal.empty and 'Δ %' in decadal.columns and 'variable' in decadal.columns:
        df_b = decadal.dropna(subset=['Δ %'])
        colors = ['red' if val < 0 else 'green' for val in df_b['Δ %']]
        fig_bar.add_trace(go.Bar(y=df_b['variable'], x=df_b['Δ %'], orientation='h', marker_color=colors))
        fig_bar.update_layout(title="Δ% change (first→last period)", yaxis={'categoryorder':'total ascending'})
        
        # Identify period columns for heatmap
        period_cols = [c for c in decadal.columns if ('-' in c or '19' in c or '20' in c) and c not in ['variable', 'Δ %']]
        if period_cols:
            z = decadal[period_cols].values
            zmax = np.nanmax(np.abs(z)) if np.nanmax(np.abs(z)) != 0 else 1
            fig_hm.add_trace(go.Heatmap(x=period_cols, y=decadal['variable'], z=z, colorscale='RdBu_r', zmid=0, zmin=-zmax, zmax=zmax))
            fig_hm.update_layout(title="Period Anomaly vs First Period", margin=dict(l=150))
    else:
        fig_bar.add_annotation(text="decadal_changes.csv MISSING", showarrow=False, font=dict(size=16, color="red"))
        fig_hm.add_annotation(text="decadal_changes.csv MISSING", showarrow=False, font=dict(size=16, color="red"))

    if not mk_results.empty:
        df_mk = mk_results.copy()
        conditions = [
            (df_mk.get('p_value', 1) < 0.05) & (df_mk.get('trend', '').str.lower() == 'increasing'),
            (df_mk.get('p_value', 1) < 0.05) & (df_mk.get('trend', '').str.lower() == 'decreasing')
        ]
        choices = ['#1E5631', '#8B0000']
        df_mk['bg_color'] = np.select(conditions, choices, default='#B58E00')

        data = df_mk.to_dict('records')
        req_cols = ['variable', 'trend', 'p_value', 'tau', 'sen_slope']
        cols = [{"name": i, "id": i} for i in req_cols if i in df_mk.columns]
        
        if cols:
            table = dash_table.DataTable(
                data=data, columns=cols,
                style_data_conditional=[
                    {
                        'if': {'filter_query': f'{{variable}} eq "{row["variable"]}"'},
                        'backgroundColor': row['bg_color'],
                        'color': 'white'
                    } for row in data
                ],
                style_header={'backgroundColor': '#222', 'color': '#eee', 'border': '1px solid #444'},
                style_cell={'backgroundColor': '#333', 'color': 'white', 'textAlign':'left', 'border': '1px solid #444'},
                style_as_list_view=True,
            )

    return fig_bar, fig_hm, table

@app.callback(Output('t5-plot', 'figure'), [Input('t5-var', 'value')])
def update_t5(var):
    fig = go.Figure().update_layout(template='plotly_dark')
    if seasonal.empty or 'variable' not in seasonal.columns:
        return fig.add_annotation(text="seasonal_climatology.csv MISSING", showarrow=False, font=dict(size=16, color="red"))
    
    df = seasonal[seasonal['variable'] == var]
    if df.empty:
        return fig.add_annotation(text=f"No data for {var}", showarrow=False)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    hist_colors = ['#1f77b4', '#4292c6', '#9ecae1'] 
    fut_colors = ['#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']
    
    for idx, row in df.iterrows():
        try:
            pi = int(row.get('period_index', 0))
            col = hist_colors[pi] if pi < 3 else (fut_colors[pi-3] if (pi-3) < len(fut_colors) else '#ffffff')
            
            y_vals = [row.get(m, np.nan) for m in months]
            fig.add_trace(go.Scatter(x=months, y=y_vals, mode='lines+markers', name=f"Period {pi}", line=dict(color=col, width=2)))
        except:
             continue
        
    fig.update_layout(title=f"Seasonal Climatology - {var}", xaxis_title="Month", yaxis_title="Mean Value")
    return fig

@app.callback(
    [Output('t6-extremes', 'figure'), Output('t6-corr', 'figure')],
    [Input('t6-var-filter', 'value')]
)
def update_t6(var_filters):
    fig_ext = go.Figure().update_layout(template='plotly_dark')
    fig_cor = go.Figure().update_layout(template='plotly_dark')
    
    if not extremes.empty and 'period' in extremes.columns and 'pct' in extremes.columns and 'variable' in extremes.columns:
        df_e = extremes.copy()
        if var_filters and len(var_filters) > 0:
            df_e = df_e[df_e['variable'].isin(var_filters)]
            
        for v, grp in df_e.groupby('variable'):
            fig_ext.add_trace(go.Bar(x=grp['period'], y=grp['pct'], name=v))
        fig_ext.update_layout(barmode='group', title="Extreme Events (%) > Threshold", xaxis_title="Period", yaxis_title="% of grid cells")
    else:
        fig_ext.add_annotation(text="extreme_events.csv MISSING or invalid", showarrow=False, font=dict(size=16, color="red"))
        
    if not corr_matrix.empty:
        z = corr_matrix.values
        fig_cor.add_trace(go.Heatmap(
            x=corr_matrix.columns, y=corr_matrix.index, z=z,
            text=np.round(z, 2), texttemplate="%{text}",
            colorscale='RdBu_r', zmin=-1, zmax=1,
            textfont={"size": 10}
        ))
        fig_cor.update_layout(title="Inter-variable Pearson Correlation", margin=dict(l=150, b=150))
    else:
        fig_cor.add_annotation(text="correlation_matrix.csv MISSING", showarrow=False, font=dict(size=16, color="red"))
        
    return fig_ext, fig_cor

@app.callback(Output('t7-eb-plot', 'figure'), [Input('t5-var', 'value')])
def update_t7(_):
    fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(template='plotly_dark')
    if energy.empty or 'year' not in energy.columns:
        return fig.add_annotation(text="energy_balance.csv MISSING", showarrow=False, font=dict(size=16, color="red"))
    
    if 'evap_fraction' in energy.columns:
        fig.add_trace(go.Scatter(x=energy['year'], y=energy['evap_fraction'], mode='lines', name='Evaporative Fraction', line=dict(color='#00ff00')), secondary_y=False)
    
    if 'bowen_ratio' in energy.columns:
        fig.add_trace(go.Scatter(x=energy['year'], y=energy['bowen_ratio'], mode='lines', name='Bowen Ratio', line=dict(color='#ff9900')), secondary_y=True)
    
    fig.add_hline(y=1.0, line_dash="dash", line_color="white", secondary_y=True, opacity=0.7)
    
    fig.update_layout(title="Energy Balance Assessment", xaxis_title="Year")
    fig.update_yaxes(title_text="Evaporative Fraction", secondary_y=False, color='#00ff00')
    fig.update_yaxes(title_text="Bowen Ratio", secondary_y=True, color='#ff9900')
    return fig

# --- New Callbacks (Tab 8, Tab 9, Tab 10) ---

@app.callback(Output('t8-plot', 'figure'), [Input('t8-var', 'value'), Input('t8-type', 'value')])
def update_t8(var, plot_type):
    fig = go.Figure().update_layout(template='plotly_dark')
    # Combine regional series into one for distributions
    df_list = []
    for reg, df_reg in regional_ts.items():
        if not df_reg.empty and var in df_reg.columns:
            temp = df_reg.copy()
            temp['Region'] = reg
            df_list.append(temp)
    if not df_list:
        return fig.add_annotation(text=f"No data for {var}", showarrow=False, font=dict(size=16, color="red"))
    
    df_all = pd.concat(df_list, ignore_ignore=True) if pd.__version__ >= '1.3' else pd.concat(df_list, ignore_index=True)
    
    if plot_type == 'hist':
        fig = px.histogram(df_all, x=var, color="Region", barmode="overlay", marginal="box", template="plotly_dark")
        fig.update_layout(title=f"Histogram of {var}")
    else:
        fig = px.box(df_all, x="Region", y=var, color="Region", template="plotly_dark")
        fig.update_layout(title=f"Boxplot of {var}")
    return fig

def get_unique_vars_for_dataset(ds_name):
    if ds_name == 'mean' and not spatial_mean_maps.empty and 'variable' in spatial_mean_maps.columns:
        return [{'label': v, 'value': v} for v in spatial_mean_maps['variable'].unique()]
    elif ds_name == 'slope' and not trend_slope_maps.empty and 'variable' in trend_slope_maps.columns:
        return [{'label': v, 'value': v} for v in trend_slope_maps['variable'].unique()]
    return []

@app.callback(
    [Output('t9-var', 'options'), Output('t9-var', 'value')],
    [Input('t9-dataset', 'value')]
)
def update_t9_options(ds_name):
    opts = get_unique_vars_for_dataset(ds_name)
    val = opts[0]['value'] if opts else None
    return opts, val

@app.callback(Output('t9-plot', 'figure'), [Input('t9-dataset', 'value'), Input('t9-var', 'value')])
def update_t9_plot(ds_name, var):
    fig = go.Figure().update_layout(template='plotly_dark')
    df = spatial_mean_maps if ds_name == 'mean' else trend_slope_maps
    val_col = 'value' if ds_name == 'mean' else 'slope'
    
    if df.empty or var is None or val_col not in df.columns:
        return fig.add_annotation(text="Data missing", showarrow=False, font=dict(color="red"))
        
    df_var = df[df['variable'] == var]
    if df_var.empty or 'lat' not in df_var.columns or 'lon' not in df_var.columns:
        return fig.add_annotation(text=f"No spatial data for {var}", showarrow=False)
        
    pivot = df_var.pivot(index='lat', columns='lon', values=val_col)
    zmax = np.nanmax(np.abs(pivot.values)) if not np.isnan(pivot.values).all() else 1
    
    fig.add_trace(go.Heatmap(x=pivot.columns, y=pivot.index, z=pivot.values, colorscale='RdBu_r', 
                             zmid=0 if ds_name == 'slope' else None, 
                             zmin=-zmax if ds_name == 'slope' else None, 
                             zmax=zmax if ds_name == 'slope' else None))
    title_prefix = "Spatial Mean" if ds_name == 'mean' else "Trend Slope"
    fig.update_layout(title=f"{title_prefix} - {var}", xaxis_title="Longitude", yaxis_title="Latitude")
    return fig

@app.callback(
    [Output('t10-var', 'options'), Output('t10-var', 'value'), Output('t10-stats-table', 'children')],
    [Input('t5-var', 'value')] # dummy input just to trigger once
)
def update_t10_setup(_):
    # Setting up the period spatial dropdown
    opts = []
    val = None
    if not period_spatial.empty and 'variable' in period_spatial.columns:
        opts = [{'label': v, 'value': v} for v in period_spatial['variable'].unique()]
        if opts: val = opts[0]['value']
        
    # Stats table
    table = html.Div("variable_statistics.csv MISSING", className="text-danger")
    if not var_stats.empty:
        data = var_stats.to_dict('records')
        cols = [{"name": i, "id": i} for i in var_stats.columns]
        table = dash_table.DataTable(
            data=data, columns=cols,
            style_header={'backgroundColor': '#222', 'color': '#eee', 'border': '1px solid #444'},
            style_cell={'backgroundColor': '#333', 'color': 'white', 'textAlign':'left', 'border': '1px solid #444'},
            style_as_list_view=True,
            page_size=10
        )
    return opts, val, table

@app.callback(Output('t10-plot', 'figure'), [Input('t10-var', 'value')])
def update_t10_plot(var):
    fig = go.Figure().update_layout(template='plotly_dark')
    if period_spatial.empty or var is None or 'variable' not in period_spatial.columns:
        return fig
        
    df_var = period_spatial[period_spatial['variable'] == var]
    if df_var.empty or 'period_year' not in df_var.columns or 'value' not in df_var.columns:
        return fig
        
    fig.add_trace(go.Scatter(x=df_var['period_year'], y=df_var['value'], mode='lines+markers', line=dict(color='#ff00ff', width=3)))
    fig.update_layout(title=f"Evolution (Courbes) Spatial Mean: {var}", xaxis_title="Start Year of Period", yaxis_title="Value")
    return fig

if __name__ == '__main__':
    app.run(debug=False, port=8050)

