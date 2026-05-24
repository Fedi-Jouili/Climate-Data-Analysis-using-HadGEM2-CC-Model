from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import json
import os

# Import data variables populated by data_loader
from data_loader import variables, ds_anomaly, regional_ts, config, inventory, seasonal

# Styling constants to match index.html
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

# --- COMPONENTS ---

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

# --- TAB CONTENTS ---

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

# --- MAIN LAYOUT ---

layout = dbc.Container([
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
        dcc.Tab(label='Headline', value='tab1', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='Spatial', value='tab2', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='Regional', value='tab3', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='EOF', value='tab4', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='Anomaly', value='tab5', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='Seasonal', value='tab6', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='Extremes', value='tab7', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='Energy', value='tab8', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='Spread', value='tab9', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
        dcc.Tab(label='Provenance', value='tab10', style={'backgroundColor': COLORS['bg0'], 'border': 'none', 'color': COLORS['text3'], 'fontSize': '13px', 'padding': '10px'}, selected_style={'backgroundColor': COLORS['bg0'], 'borderBottom': f"2px solid {COLORS['accent']}", 'color': COLORS['text1'], 'fontSize': '13px', 'padding': '10px'}),
    ], style={'height': '40px', 'backgroundColor': COLORS['bg0'], 'borderBottom': f"1px solid {COLORS['border']}", 'display': 'flex', 'gap': '24px', 'padding': '0 20px', 'position': 'fixed', 'top': '56px', 'width': '100%', 'zIndex': '900'}),
    html.Main(children=tab1_content, id="tab-content", style={'padding': '116px 20px 76px 20px', 'maxWidth': '1440px', 'margin': '0 auto'}),
    html.Footer([
        html.Div(["Last rendered: 2026-05-15"]),
        html.Div("HadGEM2-CC")
    ], style={'position': 'fixed', 'bottom': '0', 'width': '100%', 'height': '56px', 'backgroundColor': COLORS['bg0'], 'borderTop': f"1px solid {COLORS['border']}", 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '0 20px', 'color': COLORS['text3'], 'fontSize': '12px', 'zIndex': '1000'})
], fluid=True, style={'backgroundColor': COLORS['bg0'], 'minHeight': '100vh', 'padding': '0'})
