"""
This script processes housing costs data from the American Community Survey.
It cleans the data, calculates various housing cost metrics, and aggregates
the data by zip code, county, and region. The script handles data for multiple
years, combines them, and produces a final cleaned dataset saved as a parquet file.

The processed data includes information on median monthly housing costs,
real estate taxes, and derived metrics such as the number of housing units within certain price ranges and
affordability measures.

"""

import requests
import pandas as pd
import numpy as np
import csv
import json

def process_census_data(file_path):
    """
    Process the housing costs metadata file to create a dictionary for column renaming.

    Args:
    file_path (str): Path to the metadata CSV file.

    Returns:
    dict: A dictionary mapping original column names to descriptive names.
    """
    census_data_dict = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) == 2:
                census_data_dict[row[0]] = row[1]
    return census_data_dict

def process_year_data(year):
    """
    Process housing costs data for a specific year.

    Args:
    year (int): The year of data to process.

    Returns:
    pandas.DataFrame: Processed data for the specified year.
    """
    file_path = f'ACSST5Y{year}.S2506-Data.csv'
    meta_data_file_path = 'ACSST5Y2022.S2506-Column-Metadata.csv'

    df = pd.read_csv(file_path)
    census_data_dict = process_census_data(meta_data_file_path)

    df = df.rename(columns=census_data_dict)

    filtered_columns = [
        col for col in df.columns
        if "MONTHLY HOUSING COSTS" in col or "REAL ESTATE TAXES" in col or "Geography" in col or "Geographic Area Name" in col or "Median household income" in col
    ]

    df = df[filtered_columns]
    df = df.iloc[1:]
    df['Geographic Area Name'] = df['Geographic Area Name'].str.replace('ZCTA5', '', regex=False).str.strip()
    df['Year'] = year

    return df

def process_housing_costs(df):
    """
    Process the housing costs data, calculating various metrics and ratios, e.g. the Affordability_Ratio, a ratio of monthly housing cost to median monthly income.

    Args:
    df (pandas.DataFrame): Input dataframe with raw housing costs data.

    Returns:
    pandas.DataFrame: Processed dataframe with calculated housing cost metrics.
    """
    columns_to_keep = [
        'Geographic Area Name',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2022 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!Median (dollars)',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!Less than $200',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$200 to $399',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$400 to $599',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$600 to $799',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$800 to $999',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$1,000 to $1,499',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$1,500 to $1,999',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$2,000 to $2,499',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$2,500 to $2,999',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!$3,000 or more',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!REAL ESTATE TAXES!!Median (dollars)',
        'Year'
    ]

    df_selected = df[columns_to_keep]

    new_column_names = {
        'Geographic Area Name': 'Area',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2022 INFLATION-ADJUSTED DOLLARS)!!Median household income (dollars)': 'Median_Income',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!MONTHLY HOUSING COSTS!!Median (dollars)': 'Median_Monthly_Housing_Cost',
        'Estimate!!Owner-occupied housing units with a mortgage!!Owner-occupied housing units with a mortgage!!REAL ESTATE TAXES!!Median (dollars)': 'Median_Real_Estate_Taxes'
    }
    df_selected.rename(columns=new_column_names, inplace=True)

    numeric_columns = df_selected.columns.drop(['Area', 'Year'])
    for col in numeric_columns:
        df_selected[col] = pd.to_numeric(df_selected[col].replace('-', np.nan), errors='coerce')

    df_selected['No_of_housing_units_that_cost_Less_$1000'] = df_selected.iloc[:, 3:8].sum(axis=1)
    df_selected['No_of_housing_units_that_cost_$1000_to_$1999'] = df_selected.iloc[:, 8:10].sum(axis=1)
    df_selected['No_of_housing_units_that_cost_$2000_to_$2999'] = df_selected.iloc[:, 10:12].sum(axis=1)
    df_selected['No_of_housing_units_that_cost_$3000_Plus'] = df_selected.iloc[:, 12]

    df_selected['Affordability_Ratio'] = df_selected['Median_Monthly_Housing_Cost'] / (df_selected['Median_Income'] / 12)

    final_columns = ['Area', 'Median_Income', 'Median_Monthly_Housing_Cost', 'No_of_housing_units_that_cost_Less_$1000',
                     'No_of_housing_units_that_cost_$1000_to_$1999', 'No_of_housing_units_that_cost_$2000_to_$2999', 'No_of_housing_units_that_cost_$3000_Plus',
                     'Median_Real_Estate_Taxes', 'Affordability_Ratio', 'Year']

    return df_selected[final_columns]

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

def process_housing_data():
    """
    Main function to process housing costs data.

    This function runs the entire data processing pipeline, including
    reading data for multiple years, combining them, calculating statistics,
    and adding geographic information.

    Returns:
    pandas.DataFrame: The final processed dataframe.
    """
    # Process data for years 2018 to 2022
    years = range(2018, 2023)
    dfs = [process_year_data(year) for year in years]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove columns with 'Margin of Error'
    combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('Margin of Error', case=False)]
    
    # Process housing costs
    processed_df = process_housing_costs(combined_df)
    processed_df = processed_df.rename(columns={'Area': 'zip'})
    
    # Load zip to county mapping
    zips_to_counties = pd.read_csv('uszips.xlsx - Sheet1.csv')
    processed_df['zip'] = processed_df['zip'].astype(str).str.zfill(5)
    zips_to_counties['zip'] = zips_to_counties['zip'].astype(str).str.zfill(5)
    
    # Create zip to county mapping
    zip_to_county = {row['zip']: get_majority_county(row['county_weights']) for _, row in zips_to_counties.iterrows()}
    processed_df['county_fips'] = processed_df['zip'].map(zip_to_county)
    
    # Load regions list
    with open('regions_list.txt', 'r') as f:
        regions_list = f.read().splitlines()
    
    # Create and apply region mapping
    zip_to_region = create_region_mapping(zips_to_counties, regions_list)
    final_df = map_zips_to_regions(processed_df, zip_to_region)
    
    # Remove Puerto Rico zip codes
    puerto_rico_zip_codes = ['00601', '00602', '00603', '00606', '00610', '00611', '00612', '00616', '00617', '00622', '00623', '00624', '00627', '00631', '00636', '00637', '00638', '00641', '00646', '00647', '00650', '00652', '00653', '00656', '00659', '00660', '00662', '00664', '00667', '00669', '00670', '00674', '00676', '00677', '00678', '00680', '00682', '00683', '00685', '00687', '00688', '00690', '00692', '00693', '00694', '00698', '00703', '00704', '00705', '00707', '00714', '00715', '00716', '00717', '00718', '00719', '00720', '00723', '00725', '00727', '00728', '00729', '00730', '00731', '00735', '00736', '00738', '00739', '00740', '00741', '00745', '00751', '00754', '00757', '00765', '00766', '00767', '00769', '00771', '00772', '00773', '00775', '00777', '00778', '00780', '00782', '00783', '00784', '00786', '00791', '00794', '00795', '00901', '00906', '00907', '00909', '00911', '00912', '00913', '00915', '00917', '00918', '00920', '00921', '00923', '00924', '00925', '00926', '00927', '00934', '00936', '00949', '00950', '00951', '00952', '00953', '00956', '00957', '00959', '00960', '00961', '00962', '00965', '00966', '00968', '00969', '00971', '00976', '00979', '00982', '00983', '00985', '00987']
    final_df = final_df[~final_df['zip'].isin(puerto_rico_zip_codes)]
    
    # Clean up the dataframe
    final_df = final_df.dropna()
    final_df = final_df.drop(columns=['Median_Income'])
    final_df = standardize_zip_codes(final_df)
    
    return final_df

if __name__ == "__main__":
    final_df = process_housing_data()
    final_df.to_parquet('housing_costs.parquet')
