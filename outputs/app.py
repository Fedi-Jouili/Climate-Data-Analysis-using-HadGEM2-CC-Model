import dash
import dash_bootstrap_components as dbc

# Create Dash app instance
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

# Import layout
from layout import layout

app.layout = layout

# Import callbacks to register them with the app
import callbacks

if __name__ == '__main__':
    app.run(debug=False, port=8050)
