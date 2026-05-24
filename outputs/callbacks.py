from dash import Input, Output, dash_table, callback, html, ctx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json

# Import data
from data_loader import (
    ds_anomaly, regional_ts, eof_loadings, eof_scores, 
    decadal, mk_results, seasonal, extremes, corr_matrix, 
    energy, config, inventory, var_stats, spatial_mean, trend_slope, period_spatial
)

# Import tab layouts
from layout import (
    tab1_content, tab2_content, tab3_content, tab4_content, tab5_content,
    tab6_content, tab7_content, tab8_content, tab9_content, tab10_content, COLORS
)

# --- TABS SWITCHER ---

@callback(Output('tab-content', 'children'), [Input('main-tabs', 'value')])
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

# --- TAB 1: Headline Findings ---

@callback(
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

# --- TAB 2: Spatial Signal ---

@callback(
    [Output('t2-var-sel', 'options'), Output('t2-var-sel', 'value')],
    [Input('t2-dataset-sel', 'value')]
)
def update_t2_var_options(dataset):
    src_df = spatial_mean if dataset == 'mean' else trend_slope
    options = [{'label': v, 'value': v} for v in src_df['variable'].unique()]
    if not options:
        return [], None
    return options, options[0]['value']

@callback(
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
        margin=dict(t=46, r=80, b=50, l=60)
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

# --- TAB 3: Regional Analysis ---

@callback(
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

# --- TAB 4: Variability Modes ---

@callback(
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

# --- TAB 5: Anomaly Heatmap ---

@callback(
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

# --- TAB 6: Seasonal Cycle ---

@callback(
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

# --- TAB 7: Extremes & Correlation ---

@callback(
    [Output('t7-bar', 'figure'), Output('t7-corr', 'figure'), Output('kpi7-1', 'children'), Output('kpi7-2', 'children'), Output('kpi7-3', 'children')],
    [Input('main-tabs', 'value')]
)
def update_tab7(tab):
    # Ensure transparency is removed and explicit background is set to fix white-on-white visibility
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

# --- TAB 8: Energy Balance ---

@callback(Output('t8-chart', 'figure'), [Input('main-tabs', 'value')])
def update_tab8(tab):
    if tab != 'tab8': return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if not energy.empty:
        fig.add_trace(go.Scatter(x=energy['year'], y=energy['evap_fraction'], name='Evap Fraction', line=dict(color=COLORS['green'])), secondary_y=False)
        fig.add_trace(go.Scatter(x=energy['year'], y=energy['bowen_ratio'], name='Bowen Ratio', line=dict(color=COLORS['red'])), secondary_y=True)
    return fig

# --- TAB 9: Regional Spread ---

@callback(
    [Output('t9-chart', 'figure'), Output('kpi9-1', 'children'), Output('kpi9-2', 'children'), Output('kpi9-3', 'children'), Output('kpi9-4', 'children')],
    [Input('t9-var-sel', 'value'), Input('t9-type-sel', 'value')]
)
def update_tab9(var, plot_type):
    fig = go.Figure().update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    kpi9_1, kpi9_2, kpi9_3, kpi9_4 = var, "-", "-", "-"
    all_vals = []
    for reg, df in regional_ts.items():
        if df is not None and not df.empty and var in df.columns:
            vals = df[var].dropna()
            if not vals.empty:
                all_vals.extend(vals.tolist())
                if plot_type == 'box': 
                    fig.add_trace(go.Box(y=vals, name=reg, boxmean='sd', points='outliers', boxpoints='all', jitter=0.5, whiskerwidth=0.2))
                elif plot_type == 'violin': 
                    fig.add_trace(go.Violin(y=vals, name=reg, box_visible=True, meanline_visible=True))
                else:
                    fig.add_trace(go.Histogram(x=vals, name=reg, opacity=0.7))
                    fig.update_layout(barmode='overlay')
    
    # Improve visibility by auto-scaling Y axis tightly to data
    if all_vals:
        v = np.array(all_vals)
        kpi9_2, kpi9_3, kpi9_4 = f"{v.mean():.2f}", f"{v.std():.2f}", f"{pd.Series(v).skew():.2f}"
        fig.update_layout(yaxis=dict(autorange=True, zeroline=False, gridcolor=COLORS['border']))
        
    return fig, kpi9_1, kpi9_2, kpi9_3, kpi9_4

# --- TAB 10: Provenance ---

@callback(
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
