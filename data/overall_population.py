"""
Population Data Processing Script

This script processes population data from the American Community Survey (ACS) for years 2018-2022.
It calculates the working age population for each ZIP code, maps ZIP codes to counties and regions,
and produces a final dataset with population statistics.

The script handles the 2022 data separately from the 2018-2021 data.

Requirements:
- pandas
- numpy
- json

Input files required:
- ACSDP5Y{year}.DP05-Data.csv files for years 2018-2022
- ACSDP5Y2022.DP05-Column-Metadata.csv
- uszips.xlsx - Sheet1.csv
- regions_list.txt

Output:
- overall_population_per_county.parquet
"""

import pandas as pd
import numpy as np
import json
import csv

def process_metadata(file_path):
    """
    Process the metadata CSV file and return a dictionary mapping column names to their descriptions.

    Args:
    file_path (str): Path to the metadata CSV file.

    Returns:
    dict: A dictionary with column names as keys and their descriptions as values.
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
    Process ACS data for a specific year.

    Args:
    year (int): The year of the ACS data to process.

    Returns:
    pandas.DataFrame: Processed dataframe for the specified year.
    """
    file_path = f'ACSDP5Y{year}.DP05-Data.csv'
    meta_data_file_path = 'ACSDP5Y2022.DP05-Column-Metadata.csv'

    df = pd.read_csv(file_path)
    metadata_dict = process_metadata(meta_data_file_path)
    df = df.rename(columns=metadata_dict)

    df = df.iloc[1:]
    df['Geographic Area Name'] = df['Geographic Area Name'].str.replace('ZCTA5', '', regex=False).str.strip()
    df['Year'] = year

    return df

def calculate_working_age_population(df):
    """
    Calculate the total working age population for each geographic area.

    Args:
    df (pandas.DataFrame): Input dataframe containing population data.

    Returns:
    pandas.DataFrame: Dataframe with calculated working age population.
    """
    age_columns = [
        'Estimate!!SEX AND AGE!!Total population!!20 to 24 years',
        'Estimate!!SEX AND AGE!!Total population!!25 to 34 years',
        'Estimate!!SEX AND AGE!!Total population!!35 to 44 years',
        'Estimate!!SEX AND AGE!!Total population!!45 to 54 years',
        'Estimate!!SEX AND AGE!!Total population!!55 to 59 years',
        'Estimate!!SEX AND AGE!!Total population!!60 to 64 years'
    ]

    for column in age_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df['Total Working Age Population'] = df[age_columns].sum(axis=1)
    return df[['Geographic Area Name', 'Total Working Age Population', 'Year']]

def standardize_zip_codes(df, zip_column='zip'):
    """
    Standardize ZIP codes to ensure they are 5 digits long.

    Args:
    df (pandas.DataFrame): Input dataframe containing ZIP codes.
    zip_column (str): Name of the column containing ZIP codes.

    Returns:
    pandas.DataFrame: Dataframe with standardized ZIP codes.
    """
    df[zip_column] = df[zip_column].astype(str).apply(lambda x: x.zfill(5))
    return df

def get_majority_county(county_weights):
    """
    Determine the majority county from a JSON string of county weights.

    Args:
    county_weights (str): JSON string containing county weights.

    Returns:
    str: Name of the majority county, or None if parsing fails.
    """
    try:
        weights = json.loads(county_weights.replace("'", '"'))
        return max(weights, key=weights.get)
    except:
        return None

def create_region_mapping(zip_county_df, regions_list):
    """
    Create a mapping of ZIP codes to regions based on city and state information.

    Args:
    zip_county_df (pandas.DataFrame): Dataframe containing ZIP code, city, and state information.
    regions_list (list): List of region names.

    Returns:
    dict: A dictionary mapping ZIP codes to regions.
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
    Map ZIP codes in the dataframe to their corresponding regions.

    Args:
    df (pandas.DataFrame): Input dataframe containing ZIP codes.
    zip_to_region (dict): Dictionary mapping ZIP codes to regions.

    Returns:
    pandas.DataFrame: Dataframe with added 'Region' column.
    """
    df['Region'] = df['zip'].map(zip_to_region)
    df['Region'] = df['Region'].fillna('Other')
    return df

def process_population_data():
    """
    Process population data for years 2018 to 2022, including working age population calculations,
    ZIP code standardization, county and region mapping, and data cleaning.

    Returns:
    pandas.DataFrame: Final processed dataframe with population statistics by ZIP code, county, and region.
    """
    # Process data for years 2018 to 2021
    years = range(2018, 2022)
    dfs = [process_year_data(year) for year in years]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = calculate_working_age_population(combined_df)

    # Process 2022 data separately
    df_2022 = process_year_data(2022)
    df_2022 = calculate_working_age_population(df_2022)

    # Combine all data
    final_combined_df = pd.concat([combined_df, df_2022], ignore_index=True)
    final_combined_df = final_combined_df.rename(columns={'Geographic Area Name': 'zip'})

    # Remove Puerto Rico zip codes
    puerto_rico_zip_codes = ['00601', '00602', '00603', '00606', '00610', '00611', '00612', '00616', '00617', '00622', '00623', '00624', '00627', '00631', '00636', '00637', '00638', '00641', '00646', '00647', '00650', '00652', '00653', '00656', '00659', '00660', '00662', '00664', '00667', '00669', '00670', '00674', '00676', '00677', '00678', '00680', '00682', '00683', '00685', '00687', '00688', '00690', '00692', '00693', '00694', '00698', '00703', '00704', '00705', '00707', '00714', '00715', '00716', '00717', '00718', '00719', '00720', '00723', '00725', '00727', '00728', '00729', '00730', '00731', '00735', '00736', '00738', '00739', '00740', '00741', '00745', '00751', '00754', '00757', '00765', '00766', '00767', '00769', '00771', '00772', '00773', '00775', '00777', '00778', '00780', '00782', '00783', '00784', '00786', '00791', '00794', '00795', '00901', '00906', '00907', '00909', '00911', '00912', '00913', '00915', '00917', '00918', '00920', '00921', '00923', '00924', '00925', '00926', '00927', '00934', '00936', '00949', '00950', '00951', '00952', '00953', '00956', '00957', '00959', '00960', '00961', '00962', '00965', '00966', '00968', '00969', '00971', '00976', '00979', '00982', '00983', '00985', '00987']
    final_combined_df = final_combined_df[~final_combined_df['zip'].isin(puerto_rico_zip_codes)]

    # Standardize ZIP codes
    final_combined_df = standardize_zip_codes(final_combined_df)

    # Add county information
    zips_to_counties = pd.read_csv('uszips.xlsx - Sheet1.csv')
    zips_to_counties['zip'] = zips_to_counties['zip'].astype(str).str.zfill(5)
    zip_to_county = {row['zip']: get_majority_county(row['county_weights']) for _, row in zips_to_counties.iterrows()}
    final_combined_df['county_fips'] = final_combined_df['zip'].map(zip_to_county)

    # Add region information
    with open('regions_list.txt', 'r') as f:
        regions_list = f.read().splitlines()
    zip_to_region = create_region_mapping(zips_to_counties, regions_list)
    final_df = map_zips_to_regions(final_combined_df, zip_to_region)

    # Clean up the data
    final_df = final_df.dropna()
    final_df['Total Working Age Population'] = pd.to_numeric(final_df['Total Working Age Population'], errors='coerce')

    return final_df

if __name__ == "__main__":
    final_df = process_population_data()
    final_df.to_parquet('overall_population_per_county.parquet')
