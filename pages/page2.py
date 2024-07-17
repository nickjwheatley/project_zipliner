from dash import html
import dash

dash.register_page(__name__)

def layout():
    return html.Div([
        html.H1("Page 2"),
        html.P("This is Page 2 content.")
        # Add more page-specific code here
    ])
