from dash import html, dcc, callback, Output, Input, Dash, DiskcacheManager, CeleryManager
from visualization import render_time_series_plot, render_choropleth_mapbox
import dash_bootstrap_components as dbc
import os
import numpy as np
from all_labels import get_metric_labels
import plotly.express as px
from data.data_extract import query_rds
import json
from dash_objects.cards import generate_populated_cards
from dash_objects.chatgpt_object import get_chatgpt_response

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

# Get all metric definitions
data_dictionary = json.load(open('data/data_dict_v2.json'))

# Read list of all available metro areas from AWS RDS
all_metro_areas = query_rds('SELECT DISTINCT metro FROM all_data_current_snapshot_v1 ORDER BY metro',
                            config_filepath='SECRETS.ini')['metro'].tolist()

metric_labels = get_metric_labels()

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], background_callback_manager=background_callback_manager)
server = app.server

app.title = 'Zipliner'
app.layout = html.Div([
        html.P(
            ["Welcome to Zipliner! This application is designed to help prospective home buyers learn which regions to target for their future home. ",
            html.A('See site documentation.',href='https://github.com/nickjwheatley/project_zipliner/tree/main')
        ]),
        # Create Dropdowns
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label(
                        'Selected Metro Area',
                        style={'font-weight': 'bold', "text-align": "center"}),
                    dcc.Dropdown(
                        id='metro-dropdown',
                        options=[{'label': x, 'value': x} for x in all_metro_areas],
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
                        options=[{'label': x, 'value': x} if x != 5 else {'label': '5+', 'value': 5} for x in list(range(1,6))],
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
            html.Hr(),
        ]),

        # Create link to zillow homes
        html.A('See homes from selected region', href='http://www.zillow.com', id='zillow-link', target='_blank'),
        dbc.Collapse(
            dcc.Loading(html.Div(id='card-container')),
            is_open=True
        ),

        # Placeholder for superscript caveats
        html.P([
            html.I([
                html.Sup('c'),
                'Data at the county level'
            ]),
            html.Br(),
            html.I([
                html.Sup('h'),
                'Hover for details'
            ]),
        ],id='card-caveats'),

        # Placeholder for choropleth map
        html.Div([
            dcc.Loading(dcc.Graph(id='choropleth-graph'))
        ], style={'width':'64%','display':'inline-block'}),

        # Small divider between visuals
        html.Div(style={'display':'inline-block','width':'1%'}),

        # Placeholder for time series plot
        html.Div([
            dcc.Loading(dcc.Graph(id='time-series-graph'))
        ], style={'width':'35%','display':'inline-block'}),

        # Placeholder for ChatGPT Summary box
        dcc.Loading(html.Div(id='chatgpt-container'))
    ])

# Define callbacks for the choropleth graph and hide all other visuals prior to its initial load
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
            Output('zillow-link','style'),
            {"visibility": "hidden"},
            {"visibility": "visible"}
        ),
        (
            Output('chatgpt-container','style'),
            {"visibility": "hidden"},
            {"visibility": "visible"}
        )
    ]
)
def update_choropleth_graph(selected_metro_area, selected_num_bedrooms, selected_metric='zhvi'):
    # Pull choropleth data from RD
    df_query = f"SELECT * FROM all_data_current_snapshot_v1 WHERE metro = '{selected_metro_area}' AND bedrooms = {selected_num_bedrooms};"
    df = query_rds(df_query, config_filepath='SECRETS.ini')

    # Get county population
    county_population = df[['county_name','total_working_age_population']].groupby('county_name').sum().reset_index()
    county_population.columns = ['county_name','county_working_age_population']
    df = df.merge(county_population, on='county_name', how='left')

    # Ensure proper data types
    df['mean_travel_time_to_work'] = df['mean_travel_time_to_work'].replace('N',np.nan)
    df['median_age'] = df['median_age'].replace('-',np.nan)
    for label in metric_labels:
        if (df[label].dtypes == 'O') & (label != 'home_valuation_status'):
            df[label] = df[label].astype(float)

    # Render choropleth visual
    fig = render_choropleth_mapbox(df, selected_metro_area, selected_metric, selected_num_bedrooms)
    return fig


# Create function and callbacks to update the Zillow Time Series Line Plot
@callback(
    Output('time-series-graph', 'figure'),
    [
        Input('choropleth-graph', 'clickData'),
        Input('bedrooms-dropdown',  'value')
    ]
)
def update_time_series_graph(clickData, bedrooms):
    # Display empty plot with guiding title if no choropleth region has been selected
    if clickData is None:
        return px.line(title='Hover over map regions to see trended home values')

    # Get selected zip_code
    zc = clickData['points'][0]['location']

    # Get Zillow time series data for the filtered zip_code
    df_zillow_ts = query_rds(f"SELECT * FROM zillow_time_series_optimized WHERE zip_code = {int(zc)}",
                             config_filepath='SECRETS.ini')

    # Render time-series line plot
    fig = render_time_series_plot(df_zillow_ts, zc, bedrooms)
    return fig


# Update zillow home link
@callback(
    Output('zillow-link', 'href'),
    [
        Input('choropleth-graph', 'clickData')
    ]
)
def update_zillow_home_link(clickData):
    # Take user to zillow.com root if no zip code is selected
    if clickData is None:
        return 'http://www.zillow.com'

    # get filtered zip_code for choropleth selection
    zc = clickData['points'][0]['location']

    # get all zip_codes with their cities and states
    df_query = f"SELECT DISTINCT zip_code, city, state FROM all_data_current_snapshot_v1;"
    df = query_rds(df_query, config_filepath='SECRETS.ini')

    # filter on the selected zip code
    tmp_df = df.loc[(df.zip_code == zc)].head(1)
    city = tmp_df['city'].item()
    state = tmp_df['state'].item()

    # Return the specific zillow link routing the user to that zip code region
    return f'https://www.zillow.com/{city.lower()}-{state.lower()}-{zc}/'


# Create function that queries ChatGPT for a summary of the selected choropleth region
@callback(
    Output('chatgpt-container', 'children'),
    [
        Input('choropleth-graph','clickData')
    ]
)
def update_chatgpt(clickData):
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

    # Return an empty box with a helpful message if no zip_code is selected
    if clickData is None:
        return dbc.Row([
                dbc.Col([
                    html.Label(
                        'Zip Code Summary',
                        style=label_style),
                    html.P('Click zip code to see details...', style=paragraph_style)
                ], style=card_style)])
    else:
        # Get selected choropleth zip code
        zc = clickData['points'][0]['location']

        # Get ChatGPT response for selected zip code
        chatgpt_response = get_chatgpt_response(zc)

        # Return dash object showing ChatGPT response
        return dbc.Row([
                dbc.Col([
                    html.Label(
                        f'Summary for {zc}',
                        style=label_style),
                    html.P(chatgpt_response, style=paragraph_style),
                    html.I('Powered by ChatGPT')
                ], style=card_style)])



# Update metric cards by clicking on choropleth graph regions
@callback(
    Output('card-container', 'children'),
    [
        Input('choropleth-graph','clickData'),
        Input('bedrooms-dropdown','value'),
        Input('metro-dropdown','value')
    ]
)
def update_cards(clickData, bedrooms, metro):
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

    # Return empty cards if no region is selected
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

    # Get zip code of selected choropleth region
    zc = clickData['points'][0]['location']

    # Get data for zip_code
    df_query = f"SELECT * FROM all_data_current_snapshot_v1 WHERE metro = '{metro}' AND bedrooms = {bedrooms};"
    df = query_rds(df_query, config_filepath='SECRETS.ini')

    # Get county metrics
    county_population = df[['county_name', 'total_working_age_population']].groupby('county_name').sum().reset_index()
    county_population.columns = ['county_name', 'county_working_age_population']
    df = df.merge(county_population, on='county_name', how='left')

    # Ensure proper data types
    df['mean_travel_time_to_work'] = df['mean_travel_time_to_work'].replace('N', np.nan)
    df['median_age'] = df['median_age'].replace('-', np.nan)
    for label in metric_labels:
        if (df[label].dtypes == 'O') & (label != 'home_valuation_status'):
            df[label] = df[label].astype(float)
    tmp_df = df.loc[(df.zip_code == zc) & (df.bedrooms == bedrooms)].copy()

    # Get GreatSchools.org data and filter on selected zip code
    df_gs_query = f"SELECT * FROM great_schools_mean_ratings WHERE zip_code = '{str(int(zc))}'"
    df_gs = query_rds(df_gs_query, config_filepath='SECRETS.ini')
    tmp_gs = df_gs.loc[
        df_gs.zip_code == int(zc),
        ['type','level_family','distance','rating']].rename(
        columns= {
            'type':'School Type',
            'level_family':'Level',
            'distance':'Distance (M)',
            'rating':'Rating'}).copy()[['School Type', 'Level', 'Rating', 'Distance (M)']]

    # Return populated metric cards for the selected zip code
    return generate_populated_cards(tmp_df, tmp_gs, data_dictionary)

if __name__ == "__main__":
    app.run_server(debug=False, port=int(os.environ.get('PORT', 8050)), host=host)
