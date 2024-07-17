import dash
from dash import html, dcc, callback, Output, Input
import pandas as pd
from visualization import render_choropleth_map

dash.register_page(__name__)

br = 4
# desired_metro_area = 'Phoenix-Mesa-Chandler, AZ'
df = pd.read_csv('data/processed/zillow_current_snapshot.csv')
df1 = df.loc[df.bedrooms == br]

def layout():
    return html.Div([
        html.H1("Home"),
        html.P("Welcome to the Home Page"),
        html.Div([
            html.Label(
                ['Selected Metro Area'],
                style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(
                id='metro-dropdown',
                options={x: x for x in sorted(list(df.metro.dropna().unique()))},
                searchable=True,
                value='San Jose-Sunnyvale-Santa Clara, CA'
            )
        ]),
        dcc.Graph(id='choropleth-graph')
    ])

# Define the callback to update the graph
@callback(
    Output('choropleth-graph', 'figure'),
    Input('metro-dropdown', 'value')
)
def update_graph(selected_metro_area):
    fig = render_choropleth_map(df1, selected_metro_area, 'zhvi')
    return fig
