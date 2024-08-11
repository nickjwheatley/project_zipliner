import pandas as pd
import csv
import numpy as np
import json

def process_metadata(file_path):
    metadata_dict = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) == 2:
                metadata_dict[row[0]] = row[1]
    return metadata_dict

def process_year_data(year):
    file_path = f'ACSDT5Y{year}.B01002-Data.csv'
    meta_data_file_path = 'ACSDT5Y2018.B01002-Column-Metadata.csv'

    df = pd.read_csv(file_path)
    metadata_dict = process_metadata(meta_data_file_path)
    df = df.rename(columns=metadata_dict)

    df = df.iloc[1:]
    df['Geographic Area Name'] = df['Geographic Area Name'].str.replace('ZCTA5', '', regex=False).str.strip()
    df['Year'] = year

    return df[['Geographic Area Name', 'Estimate!!Median age --!!Total', 'Year']]

def standardize_zip_codes(df, zip_column='zip'):
    df[zip_column] = df[zip_column].astype(str).apply(lambda x: x.zfill(5))
    return df

def get_majority_county(county_weights):
    try:
        weights = json.loads(county_weights.replace("'", '"'))
        return max(weights, key=weights.get)
    except:
        return None

def create_region_mapping(zip_county_df, regions_list):
    regions_dict = {region.lower(): region for region in regions_list[1:]}
    zip_to_region = {}
    for _, row in zip_county_df.iterrows():
        city_state = f"{row['city']}, {row['state_id']}".lower()
        region = regions_dict.get(city_state, 'Other')
        zip_to_region[row['zip']] = region
    return zip_to_region

def map_zips_to_regions(df, zip_to_region):
    df['Region'] = df['zip'].map(zip_to_region)
    df['Region'] = df['Region'].fillna('Other')
    return df

def process_median_age_data():
    # Process data for years 2018 to 2022
    years = range(2018, 2023)
    dfs = [process_year_data(year) for year in years]

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Rename columns
    combined_df = combined_df.rename(columns={
        'Geographic Area Name': 'zip',
        'Estimate!!Median age --!!Total': 'Median Age'
    })

    # Remove Puerto Rico zip codes
    puerto_rico_zip_codes = ['00601', '00602', '00603', '00606', '00610', '00611', '00612', '00616', '00617', '00622', '00623', '00624', '00627', '00631', '00636', '00637', '00638', '00641', '00646', '00647', '00650', '00652', '00653', '00656', '00659', '00660', '00662', '00664', '00667', '00669', '00670', '00674', '00676', '00677', '00678', '00680', '00682', '00683', '00685', '00687', '00688', '00690', '00692', '00693', '00694', '00698', '00703', '00704', '00705', '00707', '00714', '00715', '00716', '00717', '00718', '00719', '00720', '00723', '00725', '00727', '00728', '00729', '00730', '00731', '00735', '00736', '00738', '00739', '00740', '00741', '00745', '00751', '00754', '00757', '00765', '00766', '00767', '00769', '00771', '00772', '00773', '00775', '00777', '00778', '00780', '00782', '00783', '00784', '00786', '00791', '00794', '00795', '00901', '00906', '00907', '00909', '00911', '00912', '00913', '00915', '00917', '00918', '00920', '00921', '00923', '00924', '00925', '00926', '00927', '00934', '00936', '00949', '00950', '00951', '00952', '00953', '00956', '00957', '00959', '00960', '00961', '00962', '00965', '00966', '00968', '00969', '00971', '00976', '00979', '00982', '00983', '00985', '00987']
    combined_df = combined_df[~combined_df['zip'].isin(puerto_rico_zip_codes)]

    # Standardize zip codes
    combined_df = standardize_zip_codes(combined_df)

    # Add county information
    zips_to_counties = pd.read_csv('uszips.xlsx - Sheet1.csv')
    zips_to_counties['zip'] = zips_to_counties['zip'].astype(str).str.zfill(5)
    zip_to_county = {row['zip']: get_majority_county(row['county_weights']) for _, row in zips_to_counties.iterrows()}
    combined_df['county_fips'] = combined_df['zip'].map(zip_to_county)

    # Add region information
    with open('regions_list.txt', 'r') as f:
        regions_list = f.read().splitlines()
    zip_to_region = create_region_mapping(zips_to_counties, regions_list)
    final_df = map_zips_to_regions(combined_df, zip_to_region)

    # Remove rows with NaN values
    final_df = final_df.dropna()

    return final_df

if __name__ == "__main__":
    final_df = process_median_age_data()
    final_df.to_parquet('median_age.parquet')