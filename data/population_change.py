"""
This script processes population change data from the American Community Survey (ACS) for the years 2018-2022.
It focuses on migration patterns, educational attainment, and age demographics across different geographic areas.
The script reads CSV files, processes the data, and outputs a final dataset in Parquet format.

"""

import pandas as pd
import numpy as np
import csv
import json

def process_metadata(file_path):
    """
    Reads a CSV file containing column metadata and returns a dictionary mapping column names to their descriptions.

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
    Processes ACS data for a specific year, renaming columns and filtering relevant information.

    Args:
    year (int): The year of the ACS data to process.

    Returns:
    pandas.DataFrame: A DataFrame containing processed data for the specified year.
    """
    file_path = f'ACSST5Y{year}.S0701-Data.csv'
    meta_data_file_path = 'ACSST5Y2018.S0701-Column-Metadata.csv'

    df = pd.read_csv(file_path)
    metadata_dict = process_metadata(meta_data_file_path)
    df = df.rename(columns=metadata_dict)

    filtered_columns = [
        col for col in df.columns
        if "Moved" in col or "Geography" in col or "Geographic Area Name" in col or 'EDUCATIONAL ATTAINMENT' in col
    ]

    df = df[filtered_columns]
    df = df.iloc[1:]
    df['Geographic Area Name'] = df['Geographic Area Name'].str.replace('ZCTA5', '', regex=False).str.strip()
    df['Year'] = year

    return df

def clean_numeric(x):
    """
    Cleans and formats numeric strings, removing commas and handling ranges.

    Args:
    x: The input value to clean.

    Returns:
    str: A cleaned string containing only digits and decimal points.
    """
    if isinstance(x, str):
        x = x.replace(',', '')
        if '-' in x:
            x = x.split('-')[0]
        x = ''.join(c for c in x if c.isdigit() or c == '.')
    return x

def safe_divide(a, b):
    """
    Performs division while handling division by zero.

    Args:
    a (numpy.array): Numerator array.
    b (numpy.array): Denominator array.

    Returns:
    numpy.array: Result of a/b, with NaN where b is zero.
    """
    return np.where(b != 0, a / b, np.nan)

def process_population_data(df):
    """
    Processes population movement data, calculating various demographic percentages such as percentage of higher education individuals who have moved to a certain jurisdiction, and totals.

    Args:
    df (pandas.DataFrame): Input DataFrame containing population data.

    Returns:
    pandas.DataFrame: Processed DataFrame with additional calculated columns.
    """
    columns_to_keep = {
        'Geographic Area Name': 'Area',
        'Year': 'Year',
        'Estimate!!Moved; within same county!!Population 1 year and over': 'Moved_Within_County',
        'Estimate!!Moved; from different county, same state!!Population 1 year and over': 'Moved_Within_State',
        'Estimate!!Moved; from different  state!!Population 1 year and over': 'Moved_From_Other_State',
        'Estimate!!Moved; within same county!!Population 1 year and over!!AGE!!25 to 34 years': 'Age_25_34_Within_County',
        'Estimate!!Moved; within same county!!Population 1 year and over!!AGE!!35 to 44 years': 'Age_35_44_Within_County',
        'Estimate!!Moved; within same county!!Population 1 year and over!!AGE!!45 to 54 years': 'Age_45_54_Within_County',
        'Estimate!!Moved; from different  state!!Population 1 year and over!!AGE!!25 to 34 years': 'Age_25_34_From_Other_State',
        'Estimate!!Moved; from different  state!!Population 1 year and over!!AGE!!35 to 44 years': 'Age_35_44_From_Other_State',
        'Estimate!!Moved; from different  state!!Population 1 year and over!!AGE!!45 to 54 years': 'Age_45_54_From_Other_State',
        'Estimate!!Moved; from different county, same state!!Population 1 year and over!!AGE!!25 to 34 years': 'Age_25_34_Within_State',
        'Estimate!!Moved; from different county, same state!!Population 1 year and over!!AGE!!35 to 44 years': 'Age_35_44_Within_State',
        'Estimate!!Moved; from different county, same state!!Population 1 year and over!!AGE!!45 to 54 years': 'Age_45_54_Within_State',
        'Estimate!!Moved; within same county!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor\'s degree': 'Bachelors_Degree_Same_County',
        'Estimate!!Moved; from different county, same state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor\'s degree': 'Bachelors_Degree_Different_County_Same_State',
        'Estimate!!Moved; from different  state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor\'s degree': 'Bachelors_Degree_Different_State',
        'Estimate!!Moved; within same county!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Graduate or professional degree': 'Graduate_Degree_Same_County',
        'Estimate!!Moved; from different county, same state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Graduate or professional degree': 'Graduate_Degree_Different_County_Same_State',
        'Estimate!!Moved; from different  state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Graduate or professional degree': 'Graduate_Degree_Different_State',
        'Estimate!!Moved; within same county!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Less than high school graduate': 'Less_HS_Graduate_Same_County',
        'Estimate!!Moved; from different county, same state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Less than high school graduate': 'Less_HS_Graduate_Different_County_Same_State',
        'Estimate!!Moved; from different  state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Less than high school graduate': 'Less_HS_Graduate_Different_State',
        'Estimate!!Moved; within same county!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!High school graduate (includes equivalency)': 'HS_Graduate_Same_County',
        'Estimate!!Moved; from different county, same state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!High school graduate (includes equivalency)': 'HS_Graduate_Different_County_Same_State',
        'Estimate!!Moved; from different  state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!High school graduate (includes equivalency)': 'HS_Graduate_Different_State',
        'Estimate!!Moved; within same county!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Some college or associate\'s degree': 'Some_College_Associate_Same_County',
        'Estimate!!Moved; from different county, same state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Some college or associate\'s degree': 'Some_College_Associate_Different_County_Same_State',
        'Estimate!!Moved; from different  state!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Some college or associate\'s degree': 'Some_College_Associate_Different_State',
    }

    df_selected = df[columns_to_keep.keys()].rename(columns=columns_to_keep)

    numeric_columns = df_selected.columns.drop(['Area', 'Year'])
    for col in numeric_columns:
        df_selected[col] = df_selected[col].apply(clean_numeric).replace('', np.nan).astype(float)

    df_selected['Higher_Education'] = df_selected['Bachelors_Degree_Same_County'] + df_selected['Bachelors_Degree_Different_County_Same_State'] + df_selected['Bachelors_Degree_Different_State'] + \
                                      df_selected['Graduate_Degree_Same_County'] + df_selected['Graduate_Degree_Different_County_Same_State'] + df_selected['Graduate_Degree_Different_State']

    df_selected['Young_Adults_Total'] = df_selected['Age_25_34_Within_County'] + df_selected['Age_25_34_Within_State'] + df_selected['Age_25_34_From_Other_State']
    df_selected['Middle_Aged_Adults_Total'] = (
        df_selected['Age_35_44_Within_County'] + df_selected['Age_35_44_Within_State'] + df_selected['Age_35_44_From_Other_State'] +
        df_selected['Age_45_54_Within_County'] + df_selected['Age_45_54_Within_State'] + df_selected['Age_45_54_From_Other_State']
    )

    total_movers = df_selected['Young_Adults_Total'] + df_selected['Middle_Aged_Adults_Total']

    total_edu_movers = (
        df_selected['Bachelors_Degree_Same_County'] +
        df_selected['Bachelors_Degree_Different_County_Same_State'] +
        df_selected['Bachelors_Degree_Different_State'] +
        df_selected['Graduate_Degree_Same_County'] +
        df_selected['Graduate_Degree_Different_County_Same_State'] +
        df_selected['Graduate_Degree_Different_State'] +
        df_selected['Less_HS_Graduate_Same_County'] +
        df_selected['Less_HS_Graduate_Different_County_Same_State'] +
        df_selected['Less_HS_Graduate_Different_State'] +
        df_selected['HS_Graduate_Same_County'] +
        df_selected['HS_Graduate_Different_County_Same_State'] +
        df_selected['HS_Graduate_Different_State'] +
        df_selected['Some_College_Associate_Same_County'] +
        df_selected['Some_College_Associate_Different_County_Same_State'] +
        df_selected['Some_College_Associate_Different_State']
    )

    df_selected['Pct_Young_Adults'] = safe_divide(df_selected['Young_Adults_Total'], total_movers)
    df_selected['Pct_Middle_Aged_Adults'] = safe_divide(df_selected['Middle_Aged_Adults_Total'], total_movers)
    df_selected['Pct_Higher_Education'] = safe_divide(df_selected['Higher_Education'], total_edu_movers)

    return df_selected

def standardize_zip_codes(df, zip_column='zip'):
    """
    Standardizes ZIP codes to ensure they are 5 digits long.

    Args:
    df (pandas.DataFrame): Input DataFrame.
    zip_column (str): Name of the column containing ZIP codes.

    Returns:
    pandas.DataFrame: DataFrame with standardized ZIP codes.
    """
    df[zip_column] = df[zip_column].astype(str).apply(lambda x: x.zfill(5))
    return df

def get_majority_county(county_weights):
    """
    Determines the majority county from a JSON string of county weights.

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
    Creates a mapping of ZIP codes to regions based on city and state information.

    Args:
    zip_county_df (pandas.DataFrame): DataFrame containing ZIP code, city, and state information.
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
    Maps ZIP codes in a DataFrame to their corresponding regions.

    Args:
    df (pandas.DataFrame): Input DataFrame containing ZIP codes.
    zip_to_region (dict): Dictionary mapping ZIP codes to regions.

    Returns:
    pandas.DataFrame: DataFrame with an additional 'Region' column.
    """
    df['Region'] = df['zip'].map(zip_to_region)
    df['Region'] = df['Region'].fillna('Other')
    return df

def process_population_change_data():
    """
    Main function to process population change data across multiple years.

    This function runs the entire data processing pipeline, including:
    - Processing data for each year from 2018 to 2022
    - Combining data from all years
    - Processing population data
    - Removing Puerto Rico ZIP codes
    - Adding county and region information
    - Cleaning and finalizing the dataset

    Returns:
    pandas.DataFrame: Final processed DataFrame containing population change data.
    """
    # Process data for years 2018 to 2022
    years = range(2018, 2023)
    dfs = [process_year_data(year) for year in years]

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Process population data
    processed_df = process_population_data(combined_df)

    # Remove Puerto Rico zip codes
    puerto_rico_zip_codes = ['00601', '00602', '00603', '00606', '00610', '00611', '00612', '00616', '00617', '00622', '00623', '00624', '00627', '00631', '00636', '00637', '00638', '00641', '00646', '00647', '00650', '00652', '00653', '00656', '00659', '00660', '00662', '00664', '00667', '00669', '00670', '00674', '00676', '00677', '00678', '00680', '00682', '00683', '00685', '00687', '00688', '00690', '00692', '00693', '00694', '00698', '00703', '00704', '00705', '00707', '00714', '00715', '00716', '00717', '00718', '00719', '00720', '00723', '00725', '00727', '00728', '00729', '00730', '00731', '00735', '00736', '00738', '00739', '00740', '00741', '00745', '00751', '00754', '00757', '00765', '00766', '00767', '00769', '00771', '00772', '00773', '00775', '00777', '00778', '00780', '00782', '00783', '00784', '00786', '00791', '00794', '00795', '00901', '00906', '00907', '00909', '00911', '00912', '00913', '00915', '00917', '00918', '00920', '00921', '00923', '00924', '00925', '00926', '00927', '00934', '00936', '00949', '00950', '00951', '00952', '00953', '00956', '00957', '00959', '00960', '00961', '00962', '00965', '00966', '00968', '00969', '00971', '00976', '00979', '00982', '00983', '00985', '00987']
    processed_df = processed_df[~processed_df['Area'].isin(puerto_rico_zip_codes)]

    # Rename 'Area' to 'zip'
    processed_df = processed_df.rename(columns={'Area': 'zip'})

    # Standardize zip codes
    processed_df = standardize_zip_codes(processed_df)

    # Add county information
    zips_to_counties = pd.read_csv('uszips.xlsx - Sheet1.csv')
    zips_to_counties['zip'] = zips_to_counties['zip'].astype(str).str.zfill(5)
    zip_to_county = {row['zip']: get_majority_county(row['county_weights']) for _, row in zips_to_counties.iterrows()}
    processed_df['county_fips'] = processed_df['zip'].map(zip_to_county)

    # Add region information
    with open('regions_list.txt', 'r') as f:
        regions_list = f.read().splitlines()
    zip_to_region = create_region_mapping(zips_to_counties, regions_list)
    final_df = map_zips_to_regions(processed_df, zip_to_region)

    # Remove rows with NaN values
    final_df = final_df.dropna()

    return final_df

if __name__ == "__main__":
    final_df = process_population_change_data()
    final_df.to_parquet('population_change.parquet')
