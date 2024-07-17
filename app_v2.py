import dash
from dash import html, dcc, callback, Output, Input, Dash, DiskcacheManager, CeleryManager
import pandas as pd
from visualization import render_choropleth_map, make_progress_graph
import dash_bootstrap_components as dbc
import os

if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

# br = 4
df = pd.read_csv('data/processed/prelim_merged_pivoted_data.csv')
# df1 = df.loc[df.bedrooms == br]

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], background_callback_manager=background_callback_manager)
app.layout = html.Div([
        html.H1("Project Zipliner"),
        html.P("Welcome to the Home Page"),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label(
                        'Selected Metro Area',
                        style={'font-weight': 'bold', "text-align": "center"}),
                    dcc.Dropdown(
                        id='metro-dropdown',
                        options=[{'label': x, 'value': x} for x in sorted(df.metro.dropna().unique())],
                        searchable=True,
                        value='San Jose-Sunnyvale-Santa Clara, CA'
                    )
                ], width=6),
                dbc.Col([
                    html.Label(
                        'Selected Number of Bedrooms',
                        style={'font-weight': 'bold', "text-align": "center"}),
                    dcc.Dropdown(
                        id='bedrooms-dropdown',
                        options=[{'label': x, 'value': x} for x in sorted(df.bedrooms.dropna().unique())],
                        searchable=True,
                        value=4
                    )
                ], width=6)
            ]),
            html.Hr(),  # This creates the divider
        ]),
        # dcc.Graph(id="progress_bar_graph", figure=make_progress_graph(0, 10)),
        html.H6('Map rendering. Please wait...',id='progress_status'),
        dcc.Graph(id='choropleth-graph')
    ])

# Define the callback to update the graph
@callback(
    Output('choropleth-graph', 'figure'),
    Input('metro-dropdown', 'value'),
    Input('bedrooms-dropdown','value'),
    background=True,
    running = [
        (
            Output('choropleth-graph','style'),
            {"visibility": "hidden"},
            {"visibility": "visible"}
        ),
        (
            Output("progress_status", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
    ]
)
def update_graph(selected_metro_area, selected_num_bedrooms):
    fig = render_choropleth_map(df.loc[df.bedrooms == selected_num_bedrooms], selected_metro_area, 'zhvi')
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
