from sklearn.linear_model import LinearRegression
from data.data_extract import query_rds
from all_labels import get_metric_labels
import numpy as np
import pandas as pd

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

def assign_non_nan_value(ser):
    """Helper function in solving for missing values in dataframe. Returns first non-nan value in regional averages"""
    val, county_val, state_val, country_val = ser.values

    if not np.isnan(val):
        return val
    elif not np.isnan(county_val):
        return county_val
    elif not np.isnan(state_val):
        return state_val
    else:
        return country_val

print('SOLVING FOR MISSING VALUES USING COUNTY, STATE, AND COUNTRY AVERAGES')
total_rows = df3.shape[0]
for col in value_cols:
    # Only fill for NA in columns with missing data
    pre_row_count = df3[col].count()
    if df3[col].count() == total_rows:
        continue

    df3[col] = df3[[col,col+'_county_mean',col+'_state_mean',col+'_country_mean',]].apply(assign_non_nan_value, axis=1)

# Drop averaged columns for missing values
df3.drop([col for col in df3.columns if ('_county_mean' in col) | ('_state_mean' in col) | ('_country_mean' in col)], axis=1, inplace=True)

# Calculate regression prediction (country)
print('PREDICTING HOME VALUE')

# Remove fields that cause overfitting
# regression_cols = list(set(value_cols) - set(['appeal_index','prop_tax_zhvi_ratio','median_real_estate_taxes', 'affordability_ratio']))
regression_cols = [
    'zhvi', 'mean_travel_time_to_work', 'median_age', 'no_of_housing_units_that_cost_less_$1000',
    'no_of_housing_units_that_cost_$1000_to_$1999', 'no_of_housing_units_that_cost_$2000_to_$2999',
    'no_of_housing_units_that_cost_$3000_plus', 'median_income', 'median_income_25_44', 'median_income_45_64',
    'median_income_65_plus', 'median_income_families','income_growth_rate', 'economic_diversity_index',
    'higher_education', 'owner_renter_ratio', 'pct_young_adults', 'pct_middle_aged_adults', 'pct_higher_education',
    'crimes_against_persons_rate', 'crimes_against_property_rate', 'crimes_against_society_rate', 'total_crime_rate',
    'total_working_age_population', 'mean_education_distance', 'mean_education_rating', 'est_number_of_jobs','job_opportunity_ratio']
X_cols = [col for col in regression_cols if col != 'zhvi']

dfs = []
for state in sorted(df3.state_name.unique()):
    for br in sorted(df3.bedrooms.unique()):
        tmp_df = df3.loc[(df3.bedrooms == br) & (df3.state_name == state)].copy()
        if tmp_df.shape[0] < 10:
            tmp_df['predicted_home_value_state'] = np.nan
        else:
            X = tmp_df[X_cols]
            y = tmp_df['zhvi']
            reg = LinearRegression().fit(X,y)
            tmp_df['predicted_home_value_state'] = reg.predict(tmp_df[X_cols])
        dfs.append(tmp_df)

df4 = pd.concat(dfs)

df4['county_state'] = df4.apply(lambda x:f'{x.county_name}, {x.state}', axis=1)
dfs = []
# for state in list(df4.state_name.unique()):
for cs in sorted(df4.county_state.unique()):
    for br in sorted(df4.bedrooms.unique()):
        tmp_df = df4.loc[(df4.bedrooms == br) & (df4.county_state == cs)].copy()
        if tmp_df.shape[0] < 20:
            tmp_df['predicted_home_value'] = tmp_df['predicted_home_value_state']
        else:
            X = tmp_df[X_cols]
            y = tmp_df['zhvi']
            reg = LinearRegression().fit(X,y)
            tmp_df['predicted_home_value'] = reg.predict(tmp_df[X_cols])
        dfs.append(tmp_df)

df5 = pd.concat(dfs)
df5.drop(['county_state', 'predicted_home_value_state'],axis=1, inplace=True)
# Calculate regression prediction (metro)

df5['home_price_difference'] = df5['zhvi'] - df5['predicted_home_value']
df5['home_price_difference_perc'] = df5['home_price_difference'] / df5['zhvi']
df5['home_valuation_status'] = 'Fairly Valued'
df5.loc[df5.home_price_difference_perc < .02, 'home_valuation_status'] = 'Undervalued'
df5.loc[df5.home_price_difference_perc > .02, 'home_valuation_status'] = 'Overvalued'

print('test')