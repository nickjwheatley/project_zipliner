import dash
from dash import html, dcc, callback, Output, Input, Dash, DiskcacheManager, CeleryManager, dash_table
import pandas as pd
from visualization import render_choropleth_map, make_progress_graph, render_time_series_plot, render_choropleth_mapbox
import dash_bootstrap_components as dbc
import os
import numpy as np
from all_labels import get_metric_labels
import plotly.express as px

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

df = pd.read_csv('data/processed/prelim_merged_pivoted_data.csv')
df_zillow_ts = pd.read_parquet('data/processed/zillow_all_data.parquet')
df_gs = pd.read_csv('data/processed/great_schools_mean_ratings.csv')

metric_labels = get_metric_labels()

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
                ], width=4),
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
                ], width=4),
                dbc.Col([
                    html.Label(
                        'Selected Metric',
                        style={'font-weight': 'bold', "text-align": "center"}),
                    dcc.Dropdown(
                        id='metrics-dropdown',
                        options=[{'label': lab[1], 'value': lab[0]} for lab in get_metric_labels().items()],
                        searchable=True,
                        value='zhvi'
                    )
                ], width=4)
            ]),
            html.Hr(),  # This creates the divider
        ]),
        # dcc.Graph(id="progress_bar_graph", figure=make_progress_graph(0, 10)),
        html.H6('Map rendering. Please wait...',id='progress_status'),
        dbc.Collapse(
            html.Div(id='card-container'),
            is_open=True
        ),
        # html.Br(),
        html.P([
            html.I([
                html.Sup('c'),
                'Data is only available at the county level'
            ])
        ],id='card-caveats'),
        html.Div([
            dcc.Graph(id='choropleth-graph')
        ], style={'width':'64%','display':'inline-block'}),
        html.Div(style={'display':'inline-block','width':'1%'}),
        html.Div([
            dcc.Graph(id='time-series-graph')
        ], style={'width':'35%','display':'inline-block'})
    ])

# Define the callback to update the graph
@callback(
    Output('choropleth-graph', 'figure'),
    Input('metro-dropdown', 'value'),
    Input('bedrooms-dropdown','value'),
    Input('metrics-dropdown','value'),
    background=True,
    running = [
        (
            Output('choropleth-graph','style'),
            {"visibility": "hidden"},
            {"visibility": "visible"}
        ),
        (
            Output('card-container','style'),
            {"visibility": "hidden"},
            {"visibility": "visible"}
        ),
        (
            Output('time-series-graph','style'),
            {"visibility": "hidden"},
            {"visibility": "visible"}
        ),
        (
            Output('card-caveats','style'),
            {"visibility": "hidden"},
            {"visibility": "visible", 'font-size':'12px'}
        ),
        (
            Output("progress_status", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden", 'font-size':'0px'},
        )
    ]
)
def update_choropleth_graph(selected_metro_area, selected_num_bedrooms, selected_metric='zhvi'):
    fig = render_choropleth_mapbox(df, selected_metro_area, selected_metric, selected_num_bedrooms)
    return fig


@callback(
    Output('time-series-graph', 'figure'),
    [
        Input('choropleth-graph', 'hoverData'),
        Input('bedrooms-dropdown',  'value')
    ]
)
def update_time_series_graph(hoverData, bedrooms):
    if hoverData is None:
        return px.line(title='Hover over map regions to see trended home values')
    zc = hoverData['points'][0]['location']
    fig = render_time_series_plot(df_zillow_ts, zc, bedrooms)
    return fig


# Update metric cards by hovering on choropleth graph regions
@callback(
    Output('card-container', 'children'),
    [
        Input('choropleth-graph','hoverData'),
        Input('bedrooms-dropdown','value')
    ]
)
def update_cards(hoverData, bedrooms):
    label_style = {
        'font-weight': 'bold',
        'text-align': 'center',
        'font-size': '20px',
        'color': '#161616'
    }
    paragraph_style = {
        'color': '#676767'
    }
    card_style = {
        'border': '2px solid #4d4d4d',
        'padding': '5',
        'paddingLeft': '20px',
        'borderRadius': '5px',
        'backgroundColor': 'white'
    }
    if hoverData is None:
        return [
            dbc.Row([
                dbc.Col([
                    html.Label(
                        'Economic',
                        style=label_style),
                    html.P('Hover over zip code to see details...', id='economic-metrics')
                ], width=3, style=card_style),
                dbc.Col([
                    html.Label(
                        'Cost of Living',
                        style=label_style),
                    html.P('Hover over zip code to see details...', id='col-metrics')
                ], width=3, style=card_style),
                dbc.Col([
                    html.Label(
                        'Demographic',
                        style=label_style),
                    html.P('Hover over zip code to see details...', id='demographic-metrics')
                ], width=3, style=card_style),
                dbc.Col([
                    html.Label(
                        'Education',
                        style=label_style),
                    html.P('Hover over zip code to see details...', id='education-metrics')
                ], width=3, style=card_style)
            ])
        ]
    zc = hoverData['points'][0]['location']
    tmp_df = df.loc[(df.zip_code == zc) & (df.bedrooms == bedrooms)]
    tmp_gs = df_gs.loc[df_gs.zip_code == zc]
    tmp_gs['rating_distance'] = tmp_gs.apply(lambda x: f'GS: {x.rating:.1f}/10  Avg Dist: {x.distance:.1f} Miles',
                                             axis=1)
    tmp_gs1 = tmp_gs[['type', 'level_family', 'rating_distance']].pivot(columns=['type'], index='level_family',
                                                                        values='rating_distance').fillna('No Data')
    return [
        dbc.Row([
            dbc.Col([
                html.Label(
                    'Economic',
                    style=label_style),
                html.P([
                    html.Span([
                        html.B([
                            metric_labels['economic_diversity_index'],
                            html.Sup('c'),
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["economic_diversity_index"].iloc[0]) \
                            else f'{tmp_df["economic_diversity_index"].iloc[0]:.2f}'],
                    title='data definition here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['est_number_of_jobs'],
                            html.Sup('c'),
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["est_number_of_jobs"].iloc[0]) \
                            else f'{tmp_df["est_number_of_jobs"].iloc[0]:,.0f}'
                    ], title='data definiton here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['median_income_families'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["median_income_families"].iloc[0]) \
                            else f'${tmp_df["median_income_families"].iloc[0]:,.0f}/yr'
                    ], title='data definiton here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['median_income'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["median_income"].iloc[0]) \
                            else f'${tmp_df["median_income"].iloc[0]:,.0f}/yr'
                    ],title='data definition here')],
                style=paragraph_style)
            ], width=3, style=card_style),
            dbc.Col([
                html.Label(
                    'Cost of Living',
                    style=label_style),
                html.P('Hover over zip code to see details...', id='col-metrics')
            ], width=3, style=card_style),
            dbc.Col([
                html.Label(
                    'Demographic',
                    style=label_style),
                html.P('Hover over zip code to see details...', id='demographic-metrics')
            ], width=3, style=card_style),
            dbc.Col([
                html.Label(
                    'Education',
                    style=label_style),
                html.P('Hover over zip code to see details...', id='education-metrics')
                # dash_table.DataTable(tmp_gs1)
            ], width=3, style=card_style)
        ])
    ]

if __name__ == "__main__":
    app.run_server(debug=True)
