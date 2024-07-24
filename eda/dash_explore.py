from data.data_extract import query_rds
import pandas as pd
from visualization import render_choropleth_mapbox

df_query = "SELECT * FROM all_data_current_snapshot_v1;"
# df_zillow_query = f"SELECT * FROM prelim_zillow_time_series;"
# df_gs_query = "SELECT * FROM great_schools_mean_ratings LIMIT 10;"


# Read data from AWS RDS
df = query_rds(df_query, config_filepath='../SECRETS.ini')
# df_zillow_ts = query_rds(df_zillow_query, config_filepath='../SECRETS.ini')
# df_gs = query_rds(df_gs_query, config_filepath='../SECRETS.ini')

render_choropleth_mapbox(df,'Atlanta-Sandy Springs-Alpharetta, GA', 'zhvi', 4)

