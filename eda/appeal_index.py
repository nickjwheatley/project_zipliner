import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#data import
file_path = '/Users/christopherorem/all_data_current_snapshot_v1.csv'
df = pd.read_csv(file_path)

#creation of additional features
df['prop_tax_zhvi_ratio'] = df['median_real_estate_taxes'] / df['zhvi']
df['job_opportunity_ratio'] = df['est_number_of_jobs'] / df['total_working_age_population']

#features to use in appeal index
normalize_columns = ['mean_education_rating', 'zhvi', 'affordability_ratio', 'income_growth_rate', 'total_crime_rate','prop_tax_zhvi_ratio', 'mean_travel_time_to_work', 'job_opportunity_ratio']

#preparing data for normalization
df_clean = df[normalize_columns]
df_clean['mean_travel_time_to_work'] = pd.to_numeric(df_clean['mean_travel_time_to_work'], errors='coerce')
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_filled = df_clean[normalize_columns].fillna(df_clean[normalize_columns].mean())

#normalize the data
scaler = MinMaxScaler()
df_filled[normalize_columns] = scaler.fit_transform(df_filled[normalize_columns])

#generate the appeal index and move back to main dataframe
df_filled['appeal_index'] = ((1-df_filled['affordability_ratio']) + df_filled['mean_education_rating'] + df_filled['zhvi'] + df_filled['income_growth_rate'] + (1-df_filled['total_crime_rate']) + (1-df_filled['prop_tax_zhvi_ratio']) + (1-df_filled['job_opportunity_ratio'])+ (1-df_filled['mean_travel_time_to_work'])) / len(normalize_columns)
df['appeal_index'] = df_filled['appeal_index']

#write to csv
df.to_csv('all_data_current_snapshot_v2.csv', index=False)
