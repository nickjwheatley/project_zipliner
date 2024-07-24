import pandas as pd
import os
import requests
import configparser
import datetime as dt
import numpy as np

def get_lat_lon_from_zip(zip_code, api_key):
    """
    Get the central latitude and longitude coordinates given a zip code from Google API
    :param zip_code: int zip code
    :param api_key: string googleapi API key
    :return: tuple of (lat, lon)
    """
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address=USA+{zip_code:05d}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            location = data['results'][0]['geometry']['location']
            return (location['lat'], location['lng'])
        else:
            return None
    else:
        return None


def setup_credentials():
    """Get API credentials from local SECRETS.ini"""
    config = configparser.ConfigParser()
    config.read('../SECRETS.ini')
    GS_API_KEY = config['great_schools_api']['key'] # Great Schools API Key
    GG_API_KEY = config['google_geocode_api']['key'] # Google API Key
    return (GS_API_KEY, GG_API_KEY)


def extract_great_schools_data(zip_code, great_schools_key, google_key):
    """
    Get Great Schools data for a single zip code
    :param zip_code: int specifying zip code
    :param great_schools_key: great schools API key
    :param google_key: Google API key
    :return: pandas Dataframe of school ratings for the given zip code
    """
    try:
        lat, lon = get_lat_lon_from_zip(zip_code, google_key)
    except Exception as e:
        print(f'No lat/lon coordinates avaiable for zip code: {zip_code}\nSkipping...')
        return None

    gs_url = f'https://gs-api.greatschools.org/nearby-schools?lat={lat}&lon={lon}&limit=50&distance=20'
    payload = {}
    headers = {
        'Accept': 'application/json',
        'Content': 'application/json',
        'X-API-Key': great_schools_key
    }

    try:
        response_dist = requests.request("GET", gs_url, headers=headers, data=payload)
        df = pd.DataFrame(response_dist.json()['schools'])
        df['query_zip_code'] = zip_code
        df['cache_date'] = dt.datetime.now().date()
    except Exception as e:
        print(f'Zip Code Failed: {zip_code}\nException: {e}\n\nSkipping...')
        return None
    return df


def extract_all_great_schools_data(zip_codes=[], force=False, cache=False,
                                   cache_filepath='raw/great_schools_data.csv'):
    """
    Extracts and combines school ratings from GreatSchools.org for all passed zipcodes (Only needs to run 1x/year)
    :param zip_codes: list of desired zip codes
    :param force: bool indicating to force a data extract
    :param cache: bool indicating whether to cache locally or not
    :param cache_filepath: string showing local filepath
    :return: pandas Dataframe of all gs ratings
    """
    if cache & (not os.path.exists('raw')):
        os.mkdir('raw')

    gs_api_key, gg_api_key = setup_credentials()

    if (not force) & (os.path.exists(cache_filepath)):
        gs_data = pd.read_csv(cache_filepath)
    else:
        dfs = []
        for i,zc in enumerate(zip_codes):
            try:
                df = extract_great_schools_data(zc, gs_api_key, gg_api_key)
            except:
                print(f'Unable to extract Great Schools data for zip code: {zc}\nSkipping....')
                continue
            dfs.append(df)

            if (i+1) % 100 == 0:
                print(f'Zip Codes logged = {i+1}/{len(zip_codes)}')
        gs_data = pd.concat(dfs)
        if cache:
            gs_data.to_csv(cache_filepath, index=False)

    return gs_data


def extract_mean_great_schools_ratings(zip_codes=[], force=False, cache=False,
                                       cache_filepath='processed/great_schools_mean_ratings.csv'):

    if (not force) & os.path.exists(cache_filepath):
        df_gs3 = pd.read_csv(cache_filepath)
    else:

        if cache & (not os.path.exists('processed')):
            os.mkdir('processed')

        great_schools_ratings = extract_all_great_schools_data(zip_codes, force, cache)
        df_gs1 = great_schools_ratings.copy()[['query_zip_code', 'distance', 'type', 'level-codes', 'rating']]
        df_gs1.dropna(subset=['rating'], inplace=True)
        df_gs1.sort_values(['query_zip_code', 'distance'], ascending=True)
        df_gs1.rating = pd.to_numeric(df_gs1.rating)
        level_families = {
            'p': 'Pre-K',
            'e': 'Elementary School',
            'm': 'Middle School',
            'h': 'High School'
        }

        df_gs1['level_family'] = np.nan

        dfs = []
        for code, family in level_families.items():
            df_tmp = df_gs1.loc[df_gs1['level-codes'].str.contains(code)]
            df_tmp['level_family'] = family
            dfs.append(df_tmp)

        df_gs2 = pd.concat(dfs).groupby(['query_zip_code', 'type', 'level_family']).head(3)

        dfs = []
        for st in list(df_gs2.type.unique()):
            for lf in list(df_gs2.level_family.unique()):
                df_tmp = df_gs2.loc[
                    (df_gs2.type == st) &
                    (df_gs2.level_family == lf)
                    ].groupby(['query_zip_code', 'type', 'level_family']).mean()
                dfs.append(df_tmp)

        df_gs3 = pd.concat(dfs).reset_index().sort_values(['query_zip_code', 'type']).rename(
            columns={'query_zip_code': 'zip_code'})
        df_gs3.to_csv(cache_filepath, index=False)

    return df_gs3