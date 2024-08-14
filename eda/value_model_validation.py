import pandas as pd
import pandas as pd
import datetime as dt
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#data import
file_path = '/Users/christopherorem/all_data_current_snapshot_v1.csv'

df3 = pd.read_csv(file_path)

#setting up model inputs
regression_cols = [
    'zhvi', 'mean_travel_time_to_work', 'median_age', 'no_of_housing_units_that_cost_less_$1000',
    'no_of_housing_units_that_cost_$1000_to_$1999', 'no_of_housing_units_that_cost_$2000_to_$2999',
    'no_of_housing_units_that_cost_$3000_plus', 'median_income', 'median_income_25_44', 'median_income_45_64',
    'median_income_65_plus', 'median_income_families', 'income_growth_rate', 'economic_diversity_index',
    'higher_education', 'owner_renter_ratio', 'pct_young_adults', 'pct_middle_aged_adults', 'pct_higher_education',
    'crimes_against_persons_rate', 'crimes_against_property_rate', 'crimes_against_society_rate',
    'total_crime_rate', 'total_working_age_population', 'mean_education_distance',
    'mean_education_rating', 'est_number_of_jobs', 'job_opportunity_ratio']
X_cols = [col for col in regression_cols if col != 'zhvi']

#dropping previosuly calculated columns
df3 = df3.drop(['predicted_home_value', 'home_price_difference', 'home_price_difference_perc', 'home_valuation_status'], axis=1)

#performing regression modeling at the state level
dfs = []
mape_lst = []
for state in sorted(df3.state_name.unique()):
    for br in sorted(df3.bedrooms.unique()):
        tmp_df = df3.loc[(df3.bedrooms == br) & (df3.state_name == state)].copy()
        if tmp_df.shape[0] < 10:
            tmp_df['predicted_home_value_state'] = np.nan
        else:
            X = tmp_df[X_cols]
            y = tmp_df['zhvi']
            reg = LinearRegression().fit(X, y)
            tmp_df['predicted_home_value_state'] = reg.predict(tmp_df[X_cols])
        dfs.append(tmp_df)
df4 = pd.concat(dfs)

#testing model performance at the county level
df4['county_state'] = df4.apply(lambda x: f'{x.county_name}, {x.state}', axis=1)
dfs = []
mape_lst = []
# for state in list(df4.state_name.unique()):
for cs in sorted(df4.county_state.unique()):
    for br in sorted(df4.bedrooms.unique()):
        tmp_df = df4.loc[(df4.bedrooms == br) & (df4.county_state == cs)].copy()
        if tmp_df.shape[0] < 20:
            tmp_df['predicted_home_value'] = tmp_df['predicted_home_value_state']
        else:
            X = tmp_df[X_cols]
            y = tmp_df['zhvi']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            mape_lst.append(mape)
            tmp_df['predicted_home_value'] = reg.predict(tmp_df[X_cols])
        dfs.append(tmp_df)
print(sum(mape_lst)/len(mape_lst))


