import dash
from dash import html, dcc, callback, Output, Input, Dash, DiskcacheManager, CeleryManager, dash_table
import pandas as pd
from visualization import render_choropleth_map, make_progress_graph, render_time_series_plot, render_choropleth_mapbox
import dash_bootstrap_components as dbc
import os
import numpy as np
from all_labels import get_metric_labels
import plotly.express as px
from data.data_extract import query_rds

ON_HEROKU = os.getenv('ON_HEROKU')
if (ON_HEROKU == True) | (ON_HEROKU == 'TRUE'):
    host = '0.0.0.0'
else:
    host = '127.0.0.1'

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

# Get RDS Instance table names schema
# rds_schema = get_rds_schema()
df_query = "SELECT * FROM all_data_current_snapshot_v1;"
# df_zillow_query = "SELECT * FROM prelim_zillow_time_series;"
df_gs_query = "SELECT * FROM great_schools_mean_ratings LIMIT 10;"

# Read data from AWS RDS
df = query_rds(df_query, config_filepath='SECRETS.ini')
# df_zillow_ts = query_rds(df_zillow_query, config_filepath='SECRETS.ini')
df_gs = query_rds(df_gs_query, config_filepath='SECRETS.ini')

# Create field for county working age population numbers (should move to data_extract eventually
county_population = df[['county_name','total_working_age_population']].groupby('county_name').sum().reset_index()
county_population.columns = ['county_name','county_working_age_population']
df = df.merge(county_population, on='county_name', how='left')


# df = pd.read_csv('data/processed/prelim_merged_pivoted_data.csv')
# df_zillow_ts = pd.read_parquet('data/processed/zillow_all_data.parquet')
# df_gs = pd.read_csv('data/processed/great_schools_mean_ratings.csv')

metric_labels = get_metric_labels()
# all_metro_areas = sorted(df.metro.dropna().unique())
prelim_metro_areas = query_rds('SELECT DISTINCT metro FROM prelim_zillow_time_series ORDER BY metro',
                               config_filepath='SECRETS.ini').metro.tolist()

# Ensure proper data types
df['mean_travel_time_to_work'] = df['mean_travel_time_to_work'].replace('N',np.nan)
df['median_age'] = df['median_age'].replace('-',np.nan)
for label in metric_labels:
    if df[label].dtypes == 'O':
        df[label] = df[label].astype(float)

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], background_callback_manager=background_callback_manager)
app.layout = html.Div([
        # html.H4("Project Zipliner"),
        html.P(
            ["Welcome to Project Zipliner, our preliminary web app intended to assist perspective home buyers know which zip codes to target for their home.",
            html.Br(),
            'Please provide your feedback ',
            dcc.Link('HERE', href='https://forms.gle/kMcM4EqzEZXsRgDDA')
        ]),
        html.P(""),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label(
                        'Selected Metro Area',
                        style={'font-weight': 'bold', "text-align": "center"}),
                    dcc.Dropdown(
                        id='metro-dropdown',
                        options=[{'label': x, 'value': x} for x in prelim_metro_areas],
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
        Input('choropleth-graph', 'clickData'),
        Input('bedrooms-dropdown',  'value')
    ]
)
def update_time_series_graph(clickData, bedrooms):
    if clickData is None:
        return px.line(title='Click on map regions to see trended home values')
    zc = clickData['points'][0]['location']
    df_zillow_ts = query_rds(f"SELECT * FROM prelim_zillow_time_series WHERE zip_code = {int(zc)}",
                             config_filepath='SECRETS.ini')
    fig = render_time_series_plot(df_zillow_ts, zc, bedrooms)
    return fig


# Update metric cards by hovering on choropleth graph regions
@callback(
    Output('card-container', 'children'),
    [
        Input('choropleth-graph','clickData'),
        Input('bedrooms-dropdown','value')
    ]
)
def update_cards(clickData, bedrooms):
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
    if clickData is None:
        return [
            dbc.Row([
                dbc.Col([
                    html.Label(
                        'Economic',
                        style=label_style),
                    html.P('Click zip code to see details...', id='economic-metrics')
                ], width=3, style=card_style),
                dbc.Col([
                    html.Label(
                        'Cost of Living',
                        style=label_style),
                    html.P('Click zip code to see details...', id='col-metrics')
                ], width=3, style=card_style),
                dbc.Col([
                    html.Label(
                        'Quality of Life',
                        style=label_style),
                    html.P('Click zip code to see details...', id='qol-metrics')
                ], width=3, style=card_style),
                dbc.Col([
                    html.Label(
                        'Demographic',
                        style=label_style),
                    html.P('Click zip code to see details...', id='demographic-metrics')
                ], width=3, style=card_style)
            ])
        ]
    zc = clickData['points'][0]['location']
    tmp_df = df.loc[(df.zip_code == zc) & (df.bedrooms == bedrooms)].copy()
    tmp_gs = df_gs.loc[df_gs.zip_code == zc].copy()
    tmp_gs['rating_distance'] = tmp_gs.apply(lambda x: f'GS: {x.rating:.1f}/10  Avg Dist: {x.distance:.1f} Miles',
                                             axis=1)
    tmp_gs1 = tmp_gs[['type', 'level_family', 'rating_distance']].pivot(
        columns=['type'],
        index='level_family',
        values='rating_distance').fillna('No Data').copy()
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
                            'Estimated Number of Jobs per Person',
                            html.Sup('c'),
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["est_number_of_jobs"].iloc[0]) \
                            else f'{(tmp_df["est_number_of_jobs"].iloc[0]/tmp_df['county_working_age_population'].iloc[0]):,.2f}'
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
                html.P([
                    html.Span([
                        html.B([
                            metric_labels['zhvi'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["zhvi"].iloc[0]) \
                            else f'${tmp_df["zhvi"].iloc[0]:,.0f}'],
                        title='data definition here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['median_real_estate_taxes'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["median_real_estate_taxes"].iloc[0]) \
                            else f'${tmp_df["median_real_estate_taxes"].iloc[0]:,.0f}'
                    ], title='data definiton here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['affordability_ratio'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["affordability_ratio"].iloc[0]) \
                            else f'{tmp_df["affordability_ratio"].iloc[0]:.2f}'
                    ], title='data definiton here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            "Home Prices Over/Under Valued",
                            ': TBD'
                        ]),
                        # 'No Data' if np.isnan(tmp_df["median_income"].iloc[0]) \
                        #     else f'${tmp_df["median_income"].iloc[0]:,.0f}/yr'
                    ], title='data definition here')],
                    style=paragraph_style)
            ], width=3, style=card_style),
            dbc.Col([
                html.Label(
                    'Quality of Life',
                    style=label_style),
                html.P([
                    html.Span([
                        html.B([
                            "Appeal Index: TBD",
                            # html.Sup('c'),
                            # ': '
                        ])],
                        # 'No Data' if np.isnan(tmp_df["zhvi"].iloc[0]) \
                        #     else f'${tmp_df["zhvi"].iloc[0]:,.0f}'],
                        title='data definition here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['mean_travel_time_to_work'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["mean_travel_time_to_work"].iloc[0]) \
                            else f'{tmp_df["mean_travel_time_to_work"].iloc[0]}'
                    ], title='data definiton here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['total_crime_rate'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["total_crime_rate"].iloc[0]) \
                            else f'{tmp_df["total_crime_rate"].iloc[0]:.3f}'
                    ], title='data definiton here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['mean_education_rating'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["mean_education_rating"].iloc[0]) \
                            else f'{tmp_df["mean_education_rating"].iloc[0]:,.1f}'
                    ], title='data definition here')],
                    style=paragraph_style)
            ], width=3, style=card_style),
            dbc.Col([
                html.Label(
                    'Demographic',
                    style=label_style),
                html.P([
                    html.Span([
                        html.B([
                            metric_labels["total_working_age_population"],
                            ": ",
                            # html.Sup('c'),
                            # ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["total_working_age_population"].iloc[0]) \
                            else f'{tmp_df["total_working_age_population"].iloc[0]:,.0f}'],
                        title='data definition here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['median_age'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["median_age"].iloc[0]) \
                            else f'{tmp_df["median_age"].iloc[0]}'
                    ], title='data definiton here'),
                    html.Br(),
                    html.Span([
                        html.B([
                            metric_labels['higher_education'],
                            ': '
                        ]),
                        'No Data' if np.isnan(tmp_df["higher_education"].iloc[0]) \
                            else f'{tmp_df["higher_education"].iloc[0]:.1f}%'
                    ], title='data definiton here')],
                    style=paragraph_style)
            ], width=3, style=card_style)
        ])
    ]

if __name__ == "__main__":
    app.run_server(debug=True, port=int(os.environ.get('PORT', 8050)), host=host)
