import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from config import config
import os
import datetime as dt


def get_rds_schema():
    """Reads zipliner database scehma from AWS RDS instance returning it as a pandas dataframe"""
    conn = setup_aws_connection('read')
    query = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('information_schema', 'pg_catalog');
    """
    return pd.read_sql(query, conn)


def query_rds(query, chunksize=None):
    """
    Reads data in AWS RDS
    :param query: string postgres sql query
    :param chunksize: integer indicating chunksize if desired
    :return: Pandas Dataframe containing results of the query
    """
    conn = setup_aws_connection('write')
    if chunksize is None:
        return pd.read_sql(query, conn)
    else:
        return pd.read_sql(query, conn, chunksize=chunksize)


def setup_aws_connection(purpose='read'):
    """
    Reads AWS Credentials and sets up connection depending on if the user wants to read or write data
    :param purpose: string indicating 'read' or 'write'
    :return:sqlalchemy engine (write) or psycopg2 connection (read)
    """
    if os.path.exists('SECRETS.ini'):
        aws_credentials = config()
    else:
        raise Exception(f'No filed "SECRETS.ini" found in directory containing AWS RDS credentials')

    db = aws_credentials['database']
    user = aws_credentials['user']
    password = aws_credentials['password']
    host = aws_credentials['host']
    port = aws_credentials['port']

    # if purpose == 'read':
    #     return psycopg2.connect(
    #         dbname=db,
    #         user=user,
    #         password=password,
    #         host=host,
    #         port=port
    #     )
    # else:
    connection_string = f'postgresql://{user}:{password}@{host}:{port}/{db}'
    return create_engine(connection_string)


def write_table(table_name, include_index=False, df=None, path=None):
    """
    Write Pandas Dataframe to AWS RDS table
    :param table_name: string indicating desired name of the table
    :param include_index: bool indicating whether to include the pandas DataFrame index or not
    :param df: pandas Dataframe to be written (if not a filepath to be read)
    :param path: string filepath to data to be written (if not loading an initialized DataFrame)
    :return: None
    """
    conn = setup_aws_connection('write')
    if path is not None:
        if '.csv' in path:
            df = pd.read_csv(path)
        elif '.parquet' in path:
            df = pd.read_parquet(path)

    df.to_sql(table_name, con=conn, if_exists='replace', index=include_index)
    return

# start = dt.datetime.now()
# write_table('prelim_merged_pivoted_snapshot', path='data/processed/prelim_merged_pivoted_data.csv')
# end = dt.datetime.now()
# runtime = end-start
# print(f'DATAFRAME LOAD RUNTIME: {runtime.seconds + round(runtime.microseconds/1e6,2)}s')
#
# query = "SELECT * FROM prelim_merged_pivoted_snapshot"# WHERE metro = 'Springfield, MA';"
# for chunksize in [None,100,500,1000,5000,10000]:
#     start = dt.datetime.now()
#     df = query_rds(query, chunksize)
#     end = dt.datetime.now()
#     runtime = end-start
#     print(f'{chunksize} CHUNKS RUNTIME: {runtime.seconds + round(runtime.microseconds/1e6,2)}s')

# Load Zillow Time Series data
# test_metros = ['San Jose-Sunnyvale-Santa Clara, CA', 'Phoenix-Mesa-Chandler, AZ']
# df_zillow = pd.read_parquet('data/processed/zillow_all_data.parquet')
# df_zillow_snippet = df_zillow.loc[df_zillow.metro.isin(test_metros)]
# write_table('prelim_zillow_time_series', df=df_zillow_snippet)

# Load GreatSchools data
# df_gs = pd.read_csv('data/processed/great_schools_mean_ratings.csv')
# df_gs['extract_year'] = 2024
# write_table('great_schools_mean_ratings', df=df_gs)