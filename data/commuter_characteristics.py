"""
This script processes commuter characteristics data from the American Community Survey.
It cleans the data, creates columns/features representing "ranges" of commute times,
and aggregates the data by zip code, county, and region.

The script reads data for multiple years, combines them, and produces a final
cleaned dataset saved as a parquet file.

Download link: `https://data.census.gov/table/ACSST5Y2022.S0801?q=S0801`: Commuting Characteristics by Sex&g=010XX00US$8600000
"""

import pandas as pd
import csv
import numpy as np
import json

def process_metadata(file_path):
    """
    Process the metadata file to create a dictionary for column renaming.

    Args:
    file_path (str): Path to the metadata CSV file.

    Returns:
    dict: A dictionary mapping original column names to descriptive names.
    """
    metadata_dict = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) == 2:
                metadata_dict[row[0]] = row[1]
    return metadata_dict

def process_year_data(year):
    """
    Process commuter data for a specific year.

    Args:
    year (int): The year of data to process.

    Returns:
    pandas.DataFrame: Processed data for the specified year.
    """
    file_path = f'ACSST5Y{year}.S0801-Data.csv'
    meta_data_file_path = 'ACSST5Y2022.S0801-Column-Metadata.csv'

    df = pd.read_csv(file_path)
    metadata_dict = process_metadata(meta_data_file_path)
    df = df.rename(columns=metadata_dict)

    filtered_columns = [
        col for col in df.columns
        if ("TRAVEL TIME TO WORK" in col or "Geography" in col or "Geographic Area Name" in col)
        and ("Margin of Error" not in col and "Female" not in col and "Male" not in col)
    ]

    df = df[filtered_columns]
    df = df.iloc[1:]
    df['Geographic Area Name'] = df['Geographic Area Name'].str.replace('ZCTA5', '', regex=False).str.strip()
    df['Year'] = year

    return df

def calculate_commute_time_groups(df):
    """
    Calculate commute time groups and percentages.

    Args:
    df (pandas.DataFrame): Input dataframe with commute time data.

    Returns:
    pandas.DataFrame: Dataframe with added commute time group percentages.
    """
    commute_columns = [
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!Less than 10 minutes',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!10 to 14 minutes',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!15 to 19 minutes',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!20 to 24 minutes',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!25 to 29 minutes',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!30 to 34 minutes',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!35 to 44 minutes',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!45 to 59 minutes',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!60 or more minutes'
    ]

    for col in commute_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['% with commute time (0-30 minutes)'] = df[commute_columns[:5]].sum(axis=1)
    df['% with commute time (30-60 minutes)'] = df[commute_columns[5:8]].sum(axis=1)
    df['% with commute time (over 60 minutes)'] = df[commute_columns[8]]

    return df

def standardize_zip_codes(df, zip_column='zip'):
    """
    Standardize zip codes to ensure they are 5 digits long.

    Args:
    df (pandas.DataFrame): Input dataframe.
    zip_column (str): Name of the column containing zip codes.

    Returns:
    pandas.DataFrame: Dataframe with standardized zip codes.
    """
    df[zip_column] = df[zip_column].astype(str).apply(lambda x: x.zfill(5))
    return df

def get_majority_county(county_weights):
    """
    Determine the majority county from a JSON string of county weights.

    Args:
    county_weights (str): JSON string of county weights.

    Returns:
    str: The FIPS code of the majority county.
    """
    try:
        weights = json.loads(county_weights.replace("'", '"'))
        return max(weights, key=weights.get)
    except:
        return None

def create_region_mapping(zip_county_df, regions_list):
    """
    Create a mapping of zip codes to regions.

    Args:
    zip_county_df (pandas.DataFrame): Dataframe with zip code and county information.
    regions_list (list): List of region names.

    Returns:
    dict: A dictionary mapping zip codes to regions.
    """
    regions_dict = {region.lower(): region for region in regions_list[1:]}
    zip_to_region = {}
    for _, row in zip_county_df.iterrows():
        city_state = f"{row['city']}, {row['state_id']}".lower()
        region = regions_dict.get(city_state, 'Other')
        zip_to_region[row['zip']] = region
    return zip_to_region

def map_zips_to_regions(df, zip_to_region):
    """
    Map zip codes in the dataframe to their corresponding regions.

    Args:
    df (pandas.DataFrame): Input dataframe.
    zip_to_region (dict): Dictionary mapping zip codes to regions.

    Returns:
    pandas.DataFrame: Dataframe with added 'Region' column.
    """
    df['Region'] = df['zip'].map(zip_to_region)
    df['Region'] = df['Region'].fillna('Other')
    return df

def process_commuter_data():
    """
    Main function to process commuter data.

    This function orchestrates the entire data processing pipeline, including
    reading data for multiple years, combining them, calculating commute range columns,
    and adding geographic information.

    Returns:
    pandas.DataFrame: The final processed dataframe.
    """
    # Process data for years 2018 to 2022
    years = range(2018, 2023)
    dfs = [process_year_data(year) for year in years]

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Calculate commute time groups
    combined_df = calculate_commute_time_groups(combined_df)

    # Select and rename required columns
    final_df = combined_df[[
        'Geographic Area Name',
        'Year',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!Mean travel time to work (minutes)',
        '% with commute time (0-30 minutes)',
        '% with commute time (30-60 minutes)',
        '% with commute time (over 60 minutes)'
    ]]
    final_df = final_df.rename(columns={
        'Geographic Area Name': 'zip',
        'Estimate!!Total!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!Mean travel time to work (minutes)': 'Mean travel time to work'
    })

    # Remove Puerto Rico zip codes
    puerto_rico_zip_codes = ['00601', '00602', '00603', '00606', '00610', '00611', '00612', '00616', '00617', '00622', '00623', '00624', '00627', '00631', '00636', '00637', '00638', '00641', '00646', '00647', '00650', '00652', '00653', '00656', '00659', '00660', '00662', '00664', '00667', '00669', '00670', '00674', '00676', '00677', '00678', '00680', '00682', '00683', '00685', '00687', '00688', '00690', '00692', '00693', '00694', '00698', '00703', '00704', '00705', '00707', '00714', '00715', '00716', '00717', '00718', '00719', '00720', '00723', '00725', '00727', '00728', '00729', '00730', '00731', '00735', '00736', '00738', '00739', '00740', '00741', '00745', '00751', '00754', '00757', '00765', '00766', '00767', '00769', '00771', '00772', '00773', '00775', '00777', '00778', '00780', '00782', '00783', '00784', '00786', '00791', '00794', '00795', '00901', '00906', '00907', '00909', '00911', '00912', '00913', '00915', '00917', '00918', '00920', '00921', '00923', '00924', '00925', '00926', '00927', '00934', '00936', '00949', '00950', '00951', '00952', '00953', '00956', '00957', '00959', '00960', '00961', '00962', '00965', '00966', '00968', '00969', '00971', '00976', '00979', '00982', '00983', '00985', '00987']
    final_df = final_df[~final_df['zip'].isin(puerto_rico_zip_codes)]

    # Standardize zip codes
    final_df = standardize_zip_codes(final_df)

    # Add county information
    zips_to_counties = pd.read_csv('uszips.xlsx - Sheet1.csv')
    zips_to_counties['zip'] = zips_to_counties['zip'].astype(str).str.zfill(5)
    zip_to_county = {row['zip']: get_majority_county(row['county_weights']) for _, row in zips_to_counties.iterrows()}
    final_df['county_fips'] = final_df['zip'].map(zip_to_county)

    # Add region information
    with open('regions_list.txt', 'r') as f:
        regions_list = f.read().splitlines()
    zip_to_region = create_region_mapping(zips_to_counties, regions_list)
    final_df = map_zips_to_regions(final_df, zip_to_region)

    # Remove rows with NaN values
    final_df = final_df.dropna()

    return final_df

if __name__ == "__main__":
    final_df = process_commuter_data()
    final_df.to_parquet('commute.parquet')
