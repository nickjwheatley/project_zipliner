from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
from app_layout import serve_layout
# import callbacks  # Ensure callbacks are imported to register them

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], use_pages=True)  # Use the Cyborg theme for dark mode
app.layout = serve_layout
app.layout = html.Div([
    html.H1('Multi-page app with Dash Pages'),
    html.Div([
        html.Div(
            dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        ) for page in dash.page_registry.values()
    ]),
    dash.page_container
])
 
if __name__ == "__main__":
    app.run_server(debug=True)
