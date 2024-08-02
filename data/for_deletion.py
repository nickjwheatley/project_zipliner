import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#import data
file_path = ''
df = pd.read_csv(file_path)

#generating additional features
df_merged4['prop_tax_zhvi_ratio'] = df_merged4['median_real_estate_taxes'] / df_merged4['zhvi']
df_merged4['job_opportunity_ratio'] = df_merged4['est_number_of_jobs'] / df_merged4['total_working_age_population']

#cleaning and preparing the data for normalization
df_clean = df_merged4.copy()
df_clean['mean_travel_time_to_work'] = pd.to_numeric(df_clean['mean_travel_time_to_work'], errors='coerce')
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
normalize_columns = ['mean_education_rating', 'affordability_ratio', 'income_growth_rate', 'total_crime_rate','prop_tax_zhvi_ratio', 'mean_travel_time_to_work', 'job_opportunity_ratio', 'mean_education_distance']
for col in normalize_columns:
    df_clean[f'normalized_{col}'] = df_clean[col]
normalize_columns = ['normalized_mean_education_rating', 'normalized_affordability_ratio', 'normalized_income_growth_rate', 'normalized_total_crime_rate','normalized_prop_tax_zhvi_ratio', 'normalized_mean_travel_time_to_work', 'normalized_job_opportunity_ratio', 'normalized_mean_education_distance']

#normalziation group function
def normalize_group(group):
    scaler = MinMaxScaler()
    group[normalize_columns] = scaler.fit_transform(group[normalize_columns])
    return group

#filling in null values with the mean for the other normalilzed values
df_filled = df_clean.fillna(df_clean[normalize_columns].mean())

#applying normlaization group function by metro area
df_filled = df_filled.groupby('metro').apply(normalize_group).reset_index(drop=True)

#adjusting normalized values for specific features to ensure higher values is a more appealing score
df_filled['normalized_affordability_ratio'] = 1-df_filled['normalized_affordability_ratio']
df_filled['normalized_total_crime_rate'] = 1-df_filled['normalized_total_crime_rate']
df_filled['normalized_prop_tax_zhvi_ratio'] = 1-df_filled['normalized_prop_tax_zhvi_ratio']
df_filled['normalized_job_opportunity_ratio'] = 1-df_filled['normalized_job_opportunity_ratio']
df_filled['normalized_mean_travel_time_to_work'] = 1-df_filled['normalized_mean_travel_time_to_work']
df_filled['normalized_mean_education_distance'] = 1-df_filled['normalized_mean_education_distance']

#creation of the appeal index
df_filled['appeal_index'] = df_filled[normalize_columns].sum(axis=1) / df_filled[normalize_columns].notna().sum(axis=1)