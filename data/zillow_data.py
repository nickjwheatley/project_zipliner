import os
import pandas as pd
import datetime as dt

def extract_zillow_snippet(bedrooms=4, force=False, cache=False):
    """
    Extracts Zillow Home Value Index data by zip code from https://www.zillow.com/research/data/
    :param bedrooms: int of desired number of bedrooms (1-5)
    :param force: bool indicating whether to force an extract or pull from cache
    :param cache: bool indicating whether to cache locally
    :return: pandas Dataframe of zillow data for desired number of bedrooms
    """
    cache_filename = f'zillow_zhvi_{bedrooms}br'
    if (os.path.isfile(f'raw/{cache_filename}.parquet') & (not force) & cache):
        zillow = pd.read_parquet(f'raw/{cache_filename}.parquet')
    else:
        print('UPDATING ZILLOW DATA')
        zillow_url = f'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_bdrmcnt_{bedrooms}_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t={int(dt.datetime.now().timestamp())}'
        zillow = pd.read_csv(zillow_url)
        if cache:
            zillow.to_parquet(f'raw/{cache_filename}.parquet', index=False)

    drop_cols = ['RegionID', 'SizeRank', 'RegionType']

    zillow_melted = zillow.drop(drop_cols, axis=1).melt(
        id_vars=['RegionName', 'StateName', 'State', 'City', 'Metro', 'CountyName'],
        var_name='date',
        value_name='zhvi'
    ).rename(columns={'RegionName': 'zip_code', 'StateName': 'state_name', 'CountyName': 'county_name'})

    zillow_melted.columns = [col.lower() for col in zillow_melted.columns]
    if cache:
        zillow_melted.to_parquet(f'processed/{cache_filename}.parquet', index=False)
    zillow_melted['bedrooms'] = bedrooms
    return zillow_melted

def get_all_zillow_data(force=False, cache=False, cache_filepath='processed/zillow_all_data.parquet'):
    """
    Combine all zillow bedroom snapshots
    :param force: bool indicating whether to force a live extract or use the cached data
    :param cache: bool indicating whether to cache locally or not
    :param cache_filepath: string indicating location of the local cache
    :return: pandas DataFrame containing all zillow data
    """
    if (not force) & (os.path.exists(cache_filepath)):
        df = pd.read_parquet(cache_filepath)
    else:
        datas = []
        for br in range(1, 6):
            print(f'extracting/cacheing zillow data for {br} bedroom units')
            datas.append(extract_zillow_snippet(br, force, cache))

        df = pd.concat(datas).reset_index(drop=True)
        if cache:
            df.to_parquet(cache_filepath)
    return df