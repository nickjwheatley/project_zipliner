import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from config import config
import os
import pandas as pd
import datetime as dt
import numpy as np
import re
from data.zillow_data import get_all_zillow_data
from data.great_schools_data import extract_mean_great_schools_ratings
from sklearn.preprocessing import MinMaxScaler
from all_labels import get_metric_labels
from sklearn.linear_model import LinearRegression
from data.merge import merge_datasets


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


def name_standardizer(x):
    """Function standardizes column name formats"""
    return '_'.join(x.split(' ')).lower()

def extract_highest_number(text):
    """
    Function is used to extract the highest number in a passed string. Used to help other function
    :param text: string containing multiple values
    :return: int showing largest number
    """
    if type(text) != type(''):
        return np.nan
    # Find all numbers in the text
    text = text.replace(',','')
    numbers = re.findall(r'\d+', text) #Credit to ChatGPT
    # Convert the numbers to integers
    numbers = [int(num) for num in numbers]
    # Return the highest number
    return max(numbers) if numbers else None

def get_rds_schema():
    """Reads zipliner database scehma from AWS RDS instance returning it as a pandas dataframe"""
    conn = setup_aws_connection('read','../SECRETS.ini')
    query = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('information_schema', 'pg_catalog');
    """
    return pd.read_sql(query, conn)


def query_rds(query, chunksize=None, config_filepath='../SECRETS.ini'):
    """
    Reads data in AWS RDS
    :param query: string postgres sql query
    :param chunksize: integer indicating chunksize if desired
    :return: Pandas Dataframe containing results of the query
    """
    conn = setup_aws_connection('write',config_filepath)
    if chunksize is None:
        return pd.read_sql(query, conn)
    else:
        return pd.read_sql(query, conn, chunksize=chunksize)


def setup_aws_connection(purpose='read', config_filepath='SECRETS.ini'):
    """
    Reads AWS Credentials and sets up connection depending on if the user wants to read or write data
    :param purpose: string indicating 'read' or 'write'
    :return:sqlalchemy engine (write) or psycopg2 connection (read)
    """
    if os.path.exists(config_filepath):
        aws_credentials = config(config_filepath)
    else:
        try:
            aws_credentials = {
                'database':os.getenv('DB_NAME'),
                'user':os.getenv('DB_USER'),
                'password':os.getenv('DB_PASSWORD'),
                'host':os.getenv('DB_HOST'),
                'port':os.getenv('DB_PORT')
            }
        except Exception as e:
            print(e)
            raise Exception(f'No file "SECRETS.ini" found in directory containing AWS RDS credentials or Heroku environment variables failing')

    db = aws_credentials['database']
    user = aws_credentials['user']
    password = aws_credentials['password']
    host = aws_credentials['host']
    port = aws_credentials['port']

    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db}'
    return create_engine(connection_string)


def write_table(table_name, include_index=False, df=None, path=None, chunk = None):
    """
    Write Pandas Dataframe to AWS RDS table
    :param table_name: string indicating desired name of the table
    :param include_index: bool indicating whether to include the pandas DataFrame index or not
    :param df: pandas Dataframe to be written (if not a filepath to be read)
    :param path: string filepath to data to be written (if not loading an initialized DataFrame)
    :return: None
    """
    conn = setup_aws_connection('write', '../SECRETS.ini')
    if path is not None:
        if '.csv' in path:
            df = pd.read_csv(path)
        elif '.parquet' in path:
            df = pd.read_parquet(path)

    if chunk is None:
        df.to_sql(table_name, con=conn, if_exists='replace', index=include_index)
    else:
        # Load in chunks to prevent memory errors
        print(f'LOADING IN {chunk} CHUNKS')
        num_rows = df.shape[0]
        df.iloc[:chunk].to_sql(table_name, con=conn, if_exists='replace', index=include_index)
        # print(df.iloc[:chunk].index.max())
        for i in range(1,num_rows // chunk):
            print(f'{i} CHUNKS LOADED')
            df.iloc[(i)*chunk:(i+1)*chunk].to_sql(table_name, con=conn, if_exists='append', index=include_index)
            # print(df.iloc[(i)*chunk:(i+1)*chunk].index.max())
        df.iloc[(i+1)*chunk:].to_sql(table_name, con=conn, if_exists='append', index=include_index)
        # print(df.iloc[(i+1)*chunk:].index.max())
    return


def refresh_all_data_in_rds(non_gs_force=False, gs_force=False, time_series_force=False, cache=False):
    """
    Refresh data from all data sources and load to AWS RDS
    :param non_gs_force: bool indicating whether to force a refresh from all non-GreatSchools.org data sources
    :param gs_force: bool indicating whether to force a refresh from GreatSchools.org (long and expensive)
    :param cache: bool indicating whether to cache final outputs locally
    :return: None
    """
    # Refresh Zillow Data
    print('REFRESHING ZILLOW DATA')
    zillow = get_all_zillow_data(non_gs_force, cache=True)
    zillow_current_snapshot = zillow.groupby(['zip_code','bedrooms']).last().reset_index()

    # Refresh GreatSchools.org Data
    print('REFRESHING GREAT SCHOOLS DATA')
    great_schools = extract_mean_great_schools_ratings(force=gs_force) # This takes a long time and can be expensive

    # Refresh all other data
    print('REFRESHING OTHER DATA')
    # merged_sans_zillow = merge_datasets() # Run for a live refresh of non-zillow data
    merged_sans_zillow = pd.read_parquet('raw/merged_dataset_before_zillow_V2.parquet')

    # Merge Zillow and non-Zillow current snapshots
    print('CLEANING DATA')
    zillow_current_snapshot.zip_code = zillow_current_snapshot.zip_code.apply(lambda x: f'{x:05}')
    great_schools.zip_code = great_schools.zip_code.apply(lambda x: f'{x:05}') #format zip code to five digits
    df_gs = great_schools.loc[
        (great_schools.type == 'public')].drop(['level_family', 'type'], axis=1) \
        .groupby('zip_code').mean().reset_index().rename(
        columns=
        {'distance': 'mean_education_distance',
         'rating': 'mean_education_rating'}).copy()

    last_year_in_merged = merged_sans_zillow.Year.max()
    df_establishments = merged_sans_zillow.loc[
        (merged_sans_zillow.Year == last_year_in_merged), ['zip', 'Number of establishments', 'Establishment Size']]

    df_establishments['high_number_of_employees'] = df_establishments['Establishment Size'].apply(
        extract_highest_number)
    df_establishments['est_number_of_jobs'] = df_establishments.high_number_of_employees * df_establishments[
        'Number of establishments']
    df_est = df_establishments[['zip', 'est_number_of_jobs']].rename(columns={'zip': 'zip_code'})
    df_est = df_est.groupby('zip_code').sum().reset_index()

    df_merged1 = merged_sans_zillow.loc[merged_sans_zillow.Year == last_year_in_merged].drop(
        ['Year', 'Establishment Size', 'Number of establishments', 'Establishment_YoY_Change'], axis=1) \
        .rename(columns={'zip': 'zip_code'}).drop_duplicates()

    df_merged2 = zillow_current_snapshot.merge(df_merged1, on='zip_code', how='left')
    df_merged3 = df_merged2.merge(df_gs, on='zip_code', how='left')
    df_merged4 = df_merged3.merge(df_est, on='zip_code', how='left')

    df_merged4.columns = [name_standardizer(col) for col in df_merged4.columns]

    # Solve for missing metro areas
    df_merged4['metro'] = df_merged4.apply(lambda x: f'{x.city}, {x.state}' if x.metro is None else x.metro, axis=1)

    # Solve for missing data
    raw_metric_labels = list(get_metric_labels().keys())
    metric_labels = []
    for col in raw_metric_labels:
        if col in df_merged4.columns.tolist():
            metric_labels.append(col)

    # Ensure correct data types
    df_merged4['mean_travel_time_to_work'] = df_merged4['mean_travel_time_to_work'].replace('N', np.nan)
    df_merged4['mean_travel_time_to_work'] = df_merged4['mean_travel_time_to_work'].replace('<NA>', np.nan)
    df_merged4['median_age'] = df_merged4['median_age'].replace('-', np.nan)
    for label in metric_labels:
        if df_merged4[label].dtypes in ['O','string']:
            df_merged4[label] = df_merged4[label].astype(float)

    value_cols = []
    for col in df_merged4.columns:
        if (df_merged4[col].dtypes not in ['O','string']) & (col not in ['zip_code', 'bedrooms', 'county_fips']):
            value_cols.append(col)

    df_county = df_merged4[['county_name', 'bedrooms'] + value_cols].groupby(['county_name', 'bedrooms']).mean().reset_index()
    df_county.columns = ['county_name', 'bedrooms'] + [col + '_county_mean' for col in df_county.columns if
                                                       col not in ['county_name', 'bedrooms']]

    df_state = df_merged4[['state_name', 'bedrooms'] + value_cols].groupby(['state_name', 'bedrooms']).mean().reset_index()
    df_state.columns = ['state_name', 'bedrooms'] + [col + '_state_mean' for col in df_state.columns if
                                                     col not in ['state_name', 'bedrooms']]

    df_country = df_merged4[['bedrooms'] + value_cols].groupby(['bedrooms']).mean().reset_index()
    df_country.columns = ['bedrooms'] + [col + '_country_mean' for col in df_country.columns if col not in ['bedrooms']]

    df1 = df_merged4.merge(df_county, on=['county_name', 'bedrooms'], how='left')
    df2 = df1.merge(df_state, on=['state_name', 'bedrooms'], how='left')
    df3 = df2.merge(df_country, on=['bedrooms'], how='left')

    print('SOLVING FOR MISSING VALUES USING COUNTY, STATE, AND COUNTRY AVERAGES')
    total_rows = df3.shape[0]
    for col in value_cols:
        # Only fill for NA in columns with missing data
        pre_row_count = df3[col].count()
        if df3[col].count() == total_rows:
            continue

        df3[col] = df3[[col, col + '_county_mean', col + '_state_mean', col + '_country_mean', ]].apply(
            assign_non_nan_value, axis=1)

    # Drop averaged columns for missing values
    df3.drop(
        [col for col in df3.columns if ('_county_mean' in col) | ('_state_mean' in col) | ('_country_mean' in col)],
        axis=1, inplace=True)

    # generating additional features
    df3['prop_tax_zhvi_ratio'] = df3['median_real_estate_taxes'] / df3['zhvi']
    df3['job_opportunity_ratio'] = df3['est_number_of_jobs'] / df3['total_working_age_population']

    # Clean infinite values
    df3 = df3.replace((-np.inf, np.inf), 0)

    # Create linear regression predicting home value
    # Calculate regression prediction (country)
    print('PREDICTING HOME VALUE')

    # Remove fields that cause overfitting
    # regression_cols = list(set(value_cols) - set(['appeal_index','prop_tax_zhvi_ratio','median_real_estate_taxes', 'affordability_ratio']))
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

    # Predict home values at the state level
    dfs = []
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

    # predict home value at the county level
    df4['county_state'] = df4.apply(lambda x: f'{x.county_name}, {x.state}', axis=1)
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
                reg = LinearRegression().fit(X, y)
                tmp_df['predicted_home_value'] = reg.predict(tmp_df[X_cols])
            dfs.append(tmp_df)

    df5 = pd.concat(dfs)
    df5.drop(['county_state', 'predicted_home_value_state'], axis=1, inplace=True)

    df5['home_price_difference'] = df5['zhvi'] - df5['predicted_home_value']
    df5['home_price_difference_perc'] = df5['home_price_difference'] / df5['zhvi']
    df5['home_valuation_status'] = 'Fairly Valued'
    df5.loc[df5.home_price_difference_perc < -.05, 'home_valuation_status'] = 'Undervalued'
    df5.loc[df5.home_price_difference_perc > .05, 'home_valuation_status'] = 'Overvalued'

    # Add Appeal Index

    # cleaning and preparing the data for normalization
    df_clean = df5.copy()
    df_clean['mean_travel_time_to_work'] = pd.to_numeric(df_clean['mean_travel_time_to_work'], errors='coerce')
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    normalize_columns = ['mean_education_rating', 'affordability_ratio', 'income_growth_rate', 'total_crime_rate',
                         'prop_tax_zhvi_ratio', 'mean_travel_time_to_work', 'job_opportunity_ratio',
                         'mean_education_distance']
    for col in normalize_columns:
        df_clean[f'normalized_{col}'] = df_clean[col]
    normalize_columns = ['normalized_mean_education_rating', 'normalized_affordability_ratio',
                         'normalized_income_growth_rate', 'normalized_total_crime_rate',
                         'normalized_prop_tax_zhvi_ratio', 'normalized_mean_travel_time_to_work',
                         'normalized_job_opportunity_ratio', 'normalized_mean_education_distance']

    # normalziation group function
    def normalize_group(group):
        scaler = MinMaxScaler()
        group[normalize_columns] = scaler.fit_transform(group[normalize_columns])
        return group

    # filling in null values with the mean for the other normalilzed values
    df_filled = df_clean.fillna(df_clean[normalize_columns].mean())

    # applying normlaization group function by metro area
    df_filled = df_filled.groupby('metro').apply(normalize_group).reset_index(drop=True)

    # adjusting normalized values for specific features to ensure higher values is a more appealing score
    df_filled['normalized_affordability_ratio'] = 1 - df_filled['normalized_affordability_ratio']
    df_filled['normalized_total_crime_rate'] = 1 - df_filled['normalized_total_crime_rate']
    df_filled['normalized_prop_tax_zhvi_ratio'] = 1 - df_filled['normalized_prop_tax_zhvi_ratio']
    df_filled['normalized_job_opportunity_ratio'] = 1 - df_filled['normalized_job_opportunity_ratio']
    df_filled['normalized_mean_travel_time_to_work'] = 1 - df_filled['normalized_mean_travel_time_to_work']
    df_filled['normalized_mean_education_distance'] = 1 - df_filled['normalized_mean_education_distance']

    # creation of the appeal index
    df_filled['appeal_index'] = df_filled[normalize_columns].sum(axis=1) / df_filled[normalize_columns].notna().sum(
        axis=1)

    df_filled = df_filled.drop(columns=normalize_columns)

    # Load data to AWS RDS
    print('LOADING CURRENT SNAPSHOT DATA TO RDS')
    write_table('all_data_current_snapshot_v1',df=df_filled) #current snapshot

    if time_series_force:
        print('LOADING ZILLOW TIME SERIES DATA TO RDS')
        write_table('zillow_time_series', df=zillow, chunk=5000000)  # Zillow time series (takes a while)

    if gs_force:
        print('LOADING GREAT SCHOOLS DATA TO RDS')
        write_table('great_schools_mean_ratings', df=df_gs) #GreatSchools.org data
    return


if __name__ == "__main__":
    # Set force to True to trigger a hard refresh for all data_sources
    refresh_all_data_in_rds(non_gs_force=False, time_series_force=False, gs_force=False)
