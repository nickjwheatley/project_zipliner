from sklearn.linear_model import LinearRegression
from data.data_extract import query_rds
from all_labels import get_metric_labels
import numpy as np

df_query = "SELECT * FROM all_data_current_snapshot_v1;"
df = query_rds(df_query, config_filepath='../SECRETS.ini')

metric_labels = get_metric_labels()

df['mean_travel_time_to_work'] = df['mean_travel_time_to_work'].replace('N',np.nan)
df['median_age'] = df['median_age'].replace('-',np.nan)
for label in metric_labels:
    if df[label].dtypes == 'O':
        df[label] = df[label].astype(float)

value_cols = []
for col in df.columns:
    if (df[col].dtypes != 'O') & (col not in ['zip_code', 'bedrooms', 'county_fips']):
        value_cols.append(col)

df_county = df[['county_name','bedrooms']+value_cols].groupby(['county_name','bedrooms']).mean().reset_index()
df_county.columns = ['county_name','bedrooms'] + [col+'_county_mean' for col in df_county.columns if col not in ['county_name','bedrooms']]

df_state = df[['state_name','bedrooms']+value_cols].groupby(['state_name','bedrooms']).mean().reset_index()
df_state.columns = ['state_name','bedrooms'] + [col+'_state_mean' for col in df_state.columns if col not in ['state_name','bedrooms']]

df_country = df[['bedrooms']+value_cols].groupby(['bedrooms']).mean().reset_index()
df_country.columns = ['bedrooms'] + [col+'_country_mean' for col in df_country.columns if col not in ['bedrooms']]

df1 = df.merge(df_county, on=['county_name','bedrooms'], how='left')
df2 = df1.merge(df_state, on=['state_name','bedrooms'], how='left')
df3 = df2.merge(df_country, on=['bedrooms'], how='left')

# for col in value_cols:
#     df3[col] = df3[col].apply(lambda x: )
print('test')