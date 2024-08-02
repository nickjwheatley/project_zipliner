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
    merged_sans_zillow = pd.read_parquet('raw/merged_dataset_before_zillow_V2.parquet')

    # Merge Zillow and non-Zillow current snapshots
    print('CLEANING DATA')
    zillow_current_snapshot.zip_code = zillow_current_snapshot.zip_code.apply(lambda x: f'{x:05}')
    great_schools.zip_code = great_schools.zip_code.apply(lambda x: f'{x:05}')
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

    # Add Appeal Index
    # generating additional features
    df_merged4['prop_tax_zhvi_ratio'] = df_merged4['median_real_estate_taxes'] / df_merged4['zhvi']
    df_merged4['job_opportunity_ratio'] = df_merged4['est_number_of_jobs'] / df_merged4['total_working_age_population']

    # cleaning and preparing the data for normalization
    df_clean = df_merged4.copy()
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


# Set force to True to trigger a hard refresh for all data_sources
# refresh_all_data_in_rds(non_gs_force=False, time_series_force=False, gs_force=False)

# print(get_rds_schema())
# filename = 'processed/all_data_current_snapshot_v4.csv'
# df = pd.read_csv(filename)
# write_table('all_data_current_snapshot_v3',path=filename)

# Create prelim time series dataset for testing/feedback
# desired_metro_areas = [
#     'Atlanta-Sandy Springs-Alpharetta, GA',
#     'Baltimore-Columbia-Towson, MD',
#     'New York-Newark-Jersey City, NY-NJ-PA',
#     'Phoenix-Mesa-Chandler, AZ',
#     'Provo-Orem, UT',
#     'Salt Lake City, UT',
#     'San Francisco-Oakland-Berkeley, CA',
#     'San Jose-Sunnyvale-Santa Clara, CA',
#     'Spokane-Spokane Valley, WA'
# ]
# zillow = get_all_zillow_data(False, False)
# zillow_subset = zillow.loc[zillow.metro.isin(desired_metro_areas)]
# write_table('prelim_zillow_time_series',df=zillow_subset)
