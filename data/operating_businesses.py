"""
This script processes and analyzes Census Bureau County Business Patterns (CBP) data for operating businesses
in the United States from 2018 to 2022. It combines data from multiple years, standardizes geography,
maps businesses to regions, calculates economic diversity indices, and engineers additional features.
The final output is a parquet file containing processed data on large business establishments.
"""

import pandas as pd
import numpy as np
import json
import csv

def process_cbp_metadata(file_path):

    """
    Process the CBP metadata file to create a dictionary for column renaming.

    Args:
    file_path (str): Path to the metadata CSV file.

    Returns:
    dict: A dictionary mapping original column names to descriptive names.
    """

    
    cbp_dict = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) == 2:
                cbp_dict[row[0]] = row[1]
    return cbp_dict

def process_year_data(year, secondary_year):
    """
    Process CBP data for a specific year.

    Args:
        year (int): The primary year of the data.
        secondary_year (int): The secondary year used in the file name.

    Returns:
        pandas.DataFrame: Processed dataframe for the specified year.
    """
    file_path = f'CBP{year}.CB{secondary_year}00CBP-Data.csv'
    meta_data_file_path = 'CBP2018.CB1800CBP-Column-Metadata.csv'

    df = pd.read_csv(file_path)
    cbp_dict = process_cbp_metadata(meta_data_file_path)

    df = df.rename(columns=cbp_dict)
    df = df.iloc[1:]  # Remove the first row if it contains column descriptions

    return df

def get_majority_county(county_weights):
    """
    Determine the majority county from a JSON string of county weights.

    Args:
        county_weights (str): JSON string containing county weights.

    Returns:
        str: The FIPS code of the majority county.
    """
    weights = json.loads(county_weights.replace("'", '"'))
    return max(weights, key=weights.get)

def standardize_geography(cbp_df, zip_county_df):
    """
    Standardize geography by mapping county FIPS codes to ZIP codes.

    Args:
        cbp_df (pandas.DataFrame): The main CBP dataframe.
        zip_county_df (pandas.DataFrame): Dataframe containing ZIP code to county mappings.

    Returns:
        pandas.DataFrame: Merged dataframe with standardized geography.
    """
    cbp_df['county_fips'] = cbp_df['Geographic identifier code'].str[-5:].str.zfill(5)

    zip_county_df['majority_county_fips'] = zip_county_df['county_weights'].apply(get_majority_county)
    zip_county_df['zip'] = pd.to_numeric(zip_county_df['zip'], errors='coerce').astype('Int64')

    merged_df = pd.merge(cbp_df, zip_county_df[['zip', 'majority_county_fips']],
                         left_on='county_fips', right_on='majority_county_fips', how='left')

    return merged_df

def create_region_mapping(zip_county_df, regions_list):
    """
    Create a mapping of ZIP codes to custom regions.

    Args:
        zip_county_df (pandas.DataFrame): Dataframe containing ZIP code information.
        regions_list (list): List of custom regions.

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
        df (pandas.DataFrame): The main dataframe.
        zip_to_region (dict): Dictionary mapping ZIP codes to regions.

    Returns:
        pandas.DataFrame: Dataframe with added 'Region' column.
    """

    df['Region'] = df['zip'].map(zip_to_region)
    df['Region'] = df['Region'].fillna('Other')
    return df

def calculate_economic_diversity_index(df):
    """
    Calculate the economic diversity index for each geographic area.

    Args:
        df (pandas.DataFrame): The main dataframe.

    Returns:
        pandas.Series: Series containing economic diversity index for each geographic area.
    """
    if '2017 NAICS code' in df.columns:
        diversity_df = df.groupby(['Geographic Area Name', '2017 NAICS code'])['Number of establishments'].sum().unstack(fill_value=0)
        diversity_df = diversity_df.div(diversity_df.sum(axis=1), axis=0)
        
        def economic_diversity(x):
            x = x[x > 0]  # Remove zero values
            return -np.sum(x * np.log(x))

        return diversity_df.apply(economic_diversity, axis=1)
    else:
        return None

def engineer_features(df):
    """
    Engineer additional features for the dataset such as YoY change in establishments.

    Args:
        df (pandas.DataFrame): The main dataframe.

    Returns:
        pandas.DataFrame: Dataframe with engineered features.
    """
    df['Number of establishments'] = pd.to_numeric(df['Number of establishments'], errors='coerce')

    if 'Year' in df.columns:
        df = df.sort_values(['zip', 'Year'])
        df['Establishment_YoY_Change'] = df.groupby('zip')['Number of establishments'].pct_change()

    economic_diversity_index = calculate_economic_diversity_index(df)
    if economic_diversity_index is not None:
        df = df.merge(economic_diversity_index.rename('economic_diversity_index'),
                      left_on='Geographic Area Name',
                      right_index=True,
                      how='left')

    return df

def standardize_zip_codes(df, zip_column='zip'):
    """
    Standardize ZIP codes to ensure they are 5 digits.

    Args:
        df (pandas.DataFrame): The main dataframe.
        zip_column (str): Name of the column containing ZIP codes.

    Returns:
        pandas.DataFrame: Dataframe with standardized ZIP codes.
    """
    df[zip_column] = df[zip_column].astype(str).apply(lambda x: x.zfill(5))
    return df

def process_operating_businesses_data():
    """
    Main function to process operating businesses data.

    This function runs the entire data processing pipeline, including
    reading data for multiple years, combining them, calculating statistics/derived features,
    and adding geographic information.

    Returns:
    pandas.DataFrame: The final processed dataframe.
    """
    # Process data for years 2018 to 2022
    years = range(2018, 2023)
    secondary_years = range(18, 23)
    zipped_years = dict(zip(years, secondary_years))
    dfs = []

    for year, secondary_year in zipped_years.items():
        try:
            df = process_year_data(year, secondary_year)
            dfs.append(df)
            print(f"Successfully processed data for {year}")
        except FileNotFoundError:
            print(f"File for year {year} not found. Skipping.")
        except Exception as e:
            print(f"Error processing data for {year}: {str(e)}")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('Noise', case=False)]

    # Load zip codes data
    zips = pd.read_csv('uszips.xlsx - Sheet1.csv')

    # Standardize geography
    standardized_df = standardize_geography(combined_df, zips)
    standardized_df = standardized_df.dropna(subset=['zip', 'majority_county_fips'])

    # Remove unnecessary columns
    columns_to_drop = ['Unnamed: 16', 'Annual payroll ($1,000)', 'First-quarter payroll ($1,000)', 
                       'Number of employees', 'Legal form of organization code', 
                       'Meaning of Legal form of organization code']
    standardized_df = standardized_df.drop(columns=[col for col in columns_to_drop if col in standardized_df.columns])

    # Filter for large establishments
    large_establishment_sizes = [
        'Establishments with 100 to 249 employees',
        'Establishments with 250 to 499 employees',
        'Establishments with 500 to 999 employees',
        'Establishments with 1,000 employees or more',
        'Establishments with 1,000 to 1,499 employees',
        'Establishments with 1,500 to 2,499 employees',
        'Establishments with 2,500 to 4,999 employees',
        'Establishments with 5,000 employees or more'
    ]
    standardized_df = standardized_df[standardized_df['Meaning of Employment size of establishments code'].isin(large_establishment_sizes)]

    # Map regions
    with open('regions_list.txt', 'r') as f:
        regions_list = f.read().splitlines()
    zip_to_region = create_region_mapping(zips, regions_list)
    final_df = map_zips_to_regions(standardized_df, zip_to_region)

    # Engineer features
    df_engineered = engineer_features(final_df)

    # Rename and drop columns
    df_engineered = df_engineered.rename(columns={
        'Geographic Area Name': 'County',
        'Meaning of Employment size of establishments code': 'Establishment Size',
        'majority_county_fips': 'county_fips'
    })
    df_engineered = df_engineered.drop(columns=['2017 NAICS code', 'Employment size of establishments code', 
                                                'Geographic identifier code', 'Meaning of NAICS code'])

    # Convert Year to int
    df_engineered['Year'] = df_engineered['Year'].astype('int')

    # Standardize zip codes
    final_df_engineered = standardize_zip_codes(df_engineered)

    # Remove Puerto Rico zip codes
    puerto_rico_zip_codes = ['00601', '00602', '00603', '00606', '00610', '00611', '00612', '00616', '00617', '00622', '00623', '00624', '00627', '00631', '00636', '00637', '00638', '00641', '00646', '00647', '00650', '00652', '00653', '00656', '00659', '00660', '00662', '00664', '00667', '00669', '00670', '00674', '00676', '00677', '00678', '00680', '00682', '00683', '00685', '00687', '00688', '00690', '00692', '00693', '00694', '00698', '00703', '00704', '00705', '00707', '00714', '00715', '00716', '00717', '00718', '00719', '00720', '00723', '00725', '00727', '00728', '00729', '00730', '00731', '00735', '00736', '00738', '00739', '00740', '00741', '00745', '00751', '00754', '00757', '00765', '00766', '00767', '00769', '00771', '00772', '00773', '00775', '00777', '00778', '00780', '00782', '00783', '00784', '00786', '00791', '00794', '00795', '00901', '00906', '00907', '00909', '00911', '00912', '00913', '00915', '00917', '00918', '00920', '00921', '00923', '00924', '00925', '00926', '00927', '00934', '00936', '00949', '00950', '00951', '00952', '00953', '00956', '00957', '00959', '00960', '00961', '00962', '00965', '00966', '00968', '00969', '00971', '00976', '00979', '00982', '00983', '00985', '00987']
    final_df_engineered = final_df_engineered[~final_df_engineered['zip'].isin(puerto_rico_zip_codes)]

    # Remove rows with NaN values
    final_df_engineered = final_df_engineered.dropna()
    final_df_engineered = final_df_engineered.loc[:,~final_df_engineered.columns.duplicated()]
    return final_df_engineered

if __name__ == "__main__":
    final_df = process_operating_businesses_data()
    print(final_df.columns)
    final_df.to_parquet('operating_businesses.parquet')
