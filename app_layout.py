from dash import dcc, html
import dash_bootstrap_components as dbc

def serve_layout():
    return dbc.Container([
        dcc.Location(id='url', refresh=True),  # Track URL changes
        dbc.Row([
            dbc.Col(
                dbc.Nav(
                    [
                        dbc.NavLink("Home", href="/", active="exact"),
                        dbc.NavLink("Page 1", href="/page1", active="exact"),
                        dbc.NavLink("Page 2", href="/page2", active="exact"),
                    ],
                    vertical=True,
                    pills=True,
                ),
                width=2,
            ),
            dbc.Col(
                html.Div(id='page-content'),  # Content will be updated here
                width=10,
            ),
        ])
    ], fluid=True)
