from data.data_extract import query_rds
from sklearn.linear_model import LinearRegression
import pandas as pd
from visualization import render_choropleth_mapbox
import dash
from dash import html, dcc, callback, Output, Input, Dash, DiskcacheManager, CeleryManager, dash_table
import dash_bootstrap_components as dbc
import json
from all_labels import get_metric_labels
import numpy as np
from dash_objects.cards import generate_populated_cards

df_query = "SELECT * FROM all_data_current_snapshot_v2 WHERE zip_code = 85286 AND bedrooms = 5"
# df_zillow_query = f"SELECT * FROM prelim_zillow_time_series;"
df_gs_query = "SELECT * FROM great_schools_mean_ratings WHERE zip_code = 85286"

metric_labels = get_metric_labels()


# Read data from AWS RDS
df = query_rds(df_query, config_filepath='../SECRETS.ini')
df_gs = query_rds(df_gs_query, config_filepath='../SECRETS.ini')

tmp_gs = df_gs.loc[
        df_gs.zip_code == 85286,
        ['type','level_family','distance','rating']].rename(
        columns= {
            'type':'School Type',
            'level_family':'Level',
            'distance':'Distance (M)',
            'rating':'Rating'}).copy()[['School Type', 'Level', 'Rating', 'Distance (M)']]

# Ensure proper data types
df['mean_travel_time_to_work'] = df['mean_travel_time_to_work'].replace('N',np.nan)
df['median_age'] = df['median_age'].replace('-',np.nan)
for label in metric_labels:
    if df[label].dtypes == 'O':
        df[label] = df[label].astype(float)

data_dictionary = json.load(open('../data/data_dict_v2.json'))
metric_labels = get_metric_labels()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(generate_populated_cards(df, tmp_gs, data_dictionary))

if __name__ == '__main__':
    app.run_server(debug=True)
# df_zillow_ts = query_rds(df_zillow_query, config_filepath='../SECRETS.ini')
# df_gs = query_rds(df_gs_query, config_filepath='../SECRETS.ini')

# render_choropleth_mapbox(df,'Atlanta-Sandy Springs-Alpharetta, GA', 'zhvi', 4)

