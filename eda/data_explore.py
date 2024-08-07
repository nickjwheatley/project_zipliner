import pandas as pd
import re
import numpy as np
import json
from data.data_dict import *
from data.data_extract import query_rds
from visualization import get_geo_json_codes, render_choropleth_map

def name_standardizer(x):
    return '_'.join(x.split(' ')).lower()

def extract_highest_number(text):
    # if np.isnan(text) | (text in ['NA',np.nan,'<NA>']):
    if type(text) != type(''):
        return np.nan
    # Find all numbers in the text
    text = text.replace(',','')
    numbers = re.findall(r'\d+', text)
    # Convert the numbers to integers
    numbers = [int(num) for num in numbers]
    # Return the highest number
    return max(numbers) if numbers else None

# df_zillow = query_rds(f"SELECT * FROM zillow_time_series_optimized WHERE idx_zip_code_optimized = 95121", config_filepath='../SECRETS.ini')
# df_zillow = query_rds(f"SELECT * FROM zillow_time_series_optimized LIMIT 10", config_filepath='../SECRETS.ini')
# df_zillow = query_rds(f"SELECT * FROM prelim_zillow_time_series WHERE zip_code = 95121", config_filepath='../SECRETS.ini')

# df_zillow_ts = pd.read_parquet('../data/processed/zillow_all_data.parquet')

# df_merged = pd.read_parquet('../data/processed/merged_dataset_before_zillow.parquet')
# df_zillow = pd.read_csv('../data/processed/zillow_current_snapshot.csv')
# df_gs = pd.read_csv('../data/processed/great_schools_mean_ratings.csv')

df_query = "SELECT * FROM all_data_current_snapshot_v1 WHERE metro = 'New York-Newark-Jersey City, NY-NJ-PA';"
df = query_rds(df_query, config_filepath='../SECRETS.ini')

metro = 'New York-Newark-Jersey City, NY-NJ-PA'

df_temp = df.loc[df.metro == metro]

# states = sorted(df_temp.state_name.unique())
# zip_codes = list(df_temp.zip_code.unique())

render_choropleth_map(df_temp, metro, 'zhvi',3)

# geo_json = get_geo_json_codes(states, zip_codes)
# print('test')

# data_dictionary_v2 = {}
# for key in data_dictionary.keys():
#     if key == 'zip':
#         revised_key = 'zip_code'
#     else:
#         revised_key = name_standardizer(key)
#     data_dictionary_v2[revised_key] = data_dictionary[key]
#
# with open('../data/data_dict_v2.json', 'w') as file:
#     json.dump(data_dictionary_v2, file, indent=2)


# df_tmp = df_gs.loc[df_gs.zip_code == 95121]
# df_tmp['rating_distance'] = df_tmp.apply(lambda x: f'GS: {x.rating:.1f}/10  Avg Dist: {x.distance:.1f} Miles', axis=1)
# df_tmp1 = df_tmp[['type','level_family','rating_distance']].pivot(columns=['type'], index='level_family', values='rating_distance').fillna('No Data')
#
# df_zillow.zip_code = df_zillow.zip_code.apply(lambda x: f'{x:05}')
# df_gs.zip_code = df_gs.zip_code.apply(lambda x: f'{x:05}')
# df_gs = df_gs.loc[
#     (df_gs.type == 'public')].drop(['level_family','type'],axis=1)\
#     .groupby('zip_code').mean().reset_index().rename(
#     columns=
#     {'distance':'mean_education_distance',
#      'rating':'mean_education_rating'})
#
# df_establishments = df_merged.loc[
#     (df_merged.Year == 2022),['zip','Number of establishments', 'Establishment Size']]#\
#     # .groupby('zip').sum().reset_index().rename(columns={'zip':'zip_code'})
#
# df_establishments['high_number_of_employees'] = df_establishments['Establishment Size'].apply(extract_highest_number)
# df_establishments['est_number_of_jobs'] = df_establishments.high_number_of_employees * df_establishments['Number of establishments']
# df_est = df_establishments[['zip','est_number_of_jobs']].rename(columns={'zip':'zip_code'})
# df_est = df_est.groupby('zip_code').sum().reset_index()
#
# df_merged1 = df_merged.loc[df_merged.Year == 2022].drop(
#     ['Year', 'Establishment Size', 'Number of establishments', 'Establishment_YoY_Change'], axis=1)\
#     .rename(columns={'zip':'zip_code'}).drop_duplicates()
#
# df = df_zillow.merge(df_merged1, on='zip_code', how='left')
# df1 = df.merge(df_gs, on='zip_code', how='left')
# df2 = df1.merge(df_est, on='zip_code', how='left')
#
# # df2.fillna('No Available Data', inplace=True)
# df2.columns = [name_standardizer(col) for col in df2.columns]
# df2.to_csv('../data/processed/prelim_merged_pivoted_data.csv', index=False)
# print('test')