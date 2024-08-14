"""
Median Income Data Processing Script

This script processes median income data from various CSV files, cleans and combines the data,
and produces a final dataset with income statistics by zip code and region.

The script performs the following main steps:
1. Loads and processes median income data from multiple years
2. Cleans and standardizes the data
3. Calculates income growth rates
4. Maps zip codes to counties and regions
5. Saves the final processed dataset as a Parquet file

Requirements:
- pandas
- numpy

Input files required:
- ACSST5Y{year}.S1903-Data.csv files for years 2018-2022
- ACSST5Y2022.S1903-Column-Metadata.csv
- uszips.xlsx - Sheet1.csv
- regions_list.txt

Output:
- median_income.parquet

Download link: `https://data.census.gov/table?q=S1903:%20Median%20Income%20in%20the%20Past%2012%20Months%20(in%202022%20Inflation-Adjusted%20Dollars)&g=010XX00US$8600000`
"""

import pandas as pd
import numpy as np
import csv
import json

def process_median_income_data(file_path):
    """
    Process median income data from a CSV file.

    Args:
    file_path (str): Path to the CSV file containing median income data.

    Returns:
    dict: A dictionary mapping column names to their corresponding values.
    """
    median_income_data_dict = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) == 2:
                median_income_data_dict[row[0]] = row[1]
    return median_income_data_dict

def process_year_data(year):
    """
    Process median income data for a specific year.

    Args:
    year (int): The year for which to process data.

    Returns:
    pandas.DataFrame: Processed data for the specified year.
    """
    file_path = f'ACSST5Y{year}.S1903-Data.csv'
    meta_data_file_path = 'ACSST5Y2022.S1903-Column-Metadata.csv'

    df = pd.read_csv(file_path)
    median_income_data_dict = process_median_income_data(meta_data_file_path)
    df = df.rename(columns=median_income_data_dict)

    filtered_columns = [
        col for col in df.columns
        if "Median income" in col or "Geography" in col or "Geographic Area Name" in col
    ]

    df = df[filtered_columns]
    df = df.iloc[1:]
    df['Geographic Area Name'] = df['Geographic Area Name'].str.replace('ZCTA5', '', regex=False).str.strip()
    df['Year'] = year

    return df

def process_median_income(df):
    """
    Process and clean median income data.

    Args:
    df (pandas.DataFrame): Input DataFrame containing median income data.

    Returns:
    pandas.DataFrame: Processed and cleaned median income data.
    """
    columns_to_keep = [
        'Geographic Area Name',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households!!One race--!!White',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households!!One race--!!Black or African American',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households!!One race--!!Asian',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households!!Hispanic or Latino origin (of any race)',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!25 to 44 years',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!45 to 64 years',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!65 years and over',
        'Estimate!!Median income (dollars)!!FAMILIES!!Families',
        'Estimate!!Median income (dollars)!!FAMILIES!!Families!!With own children of householder under 18 years',
        'Estimate!!Median income (dollars)!!NONFAMILY HOUSEHOLDS!!Nonfamily households',
        'Year'
    ]

    df_selected = df[columns_to_keep]

    new_column_names = {
        'Geographic Area Name': 'Area',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households': 'Median_Income',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households!!One race--!!White': 'Median_Income_White',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households!!One race--!!Black or African American': 'Median_Income_Black',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households!!One race--!!Asian': 'Median_Income_Asian',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households!!Hispanic or Latino origin (of any race)': 'Median_Income_Hispanic',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!25 to 44 years': 'Median_Income_25_44',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!45 to 64 years': 'Median_Income_45_64',
        'Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY AGE OF HOUSEHOLDER!!65 years and over': 'Median_Income_65_plus',
        'Estimate!!Median income (dollars)!!FAMILIES!!Families': 'Median_Income_Families',
        'Estimate!!Median income (dollars)!!FAMILIES!!Families!!With own children of householder under 18 years': 'Median_Income_Families_With_Children',
        'Estimate!!Median income (dollars)!!NONFAMILY HOUSEHOLDS!!Nonfamily households': 'Median_Income_Nonfamily'
    }
    df_selected.rename(columns=new_column_names, inplace=True)

    numeric_columns = df_selected.columns.drop(['Area', 'Year'])
    for col in numeric_columns:
        df_selected[col] = pd.to_numeric(df_selected[col].replace({'-': np.nan, '2,500-': np.nan}), errors='coerce')

    df_selected['Median_Income_Family_Nonfamily_Ratio'] = df_selected['Median_Income_Families'] / df_selected['Median_Income_Nonfamily']
    df_selected['Median_Income_Young_Old_Ratio'] = df_selected['Median_Income_25_44'] / df_selected['Median_Income_65_plus']

    return df_selected

def calculate_income_growth_rate(group):
    """
    Calculate the income growth rate for a group of data.

    Args:
    group (pandas.DataFrame): A group of data for a specific area.

    Returns:
    pandas.Series: Income growth rate and the reason for the calculation.
    """
    group = group.sort_values('Year')
    group = group.dropna(subset=['Median_Income'])

    if len(group) < 2:
        return pd.Series({'Income_Growth_Rate': np.nan, 'Growth_Rate_Reason': 'Insufficient data points'})

    first_year = group.iloc[0]
    last_year = group.iloc[-1]
    years_diff = last_year['Year'] - first_year['Year']

    if years_diff == 0:
        return pd.Series({'Income_Growth_Rate': np.nan, 'Growth_Rate_Reason': 'Same year data'})

    if first_year['Median_Income'] <= 0 or last_year['Median_Income'] <= 0:
        return pd.Series({'Income_Growth_Rate': np.nan, 'Growth_Rate_Reason': 'Non-positive income value'})

    cagr = (last_year['Median_Income'] / first_year['Median_Income']) ** (1 / years_diff) - 1
    return pd.Series({'Income_Growth_Rate': cagr, 'Growth_Rate_Reason': 'Calculated'})

def standardize_zip_codes(df, zip_column='Area'):
    """
    Standardize zip codes in the DataFrame.

    Args:
    df (pandas.DataFrame): Input DataFrame containing zip codes.
    zip_column (str): Name of the column containing zip codes.

    Returns:
    pandas.DataFrame: DataFrame with standardized zip codes.
    """
    df[zip_column] = df[zip_column].astype(str)

    def pad_zip(zip_code):
        try:
            return zip_code.zfill(5)
        except AttributeError:
            return '00000'

    df[zip_column] = df[zip_column].apply(pad_zip)

    return df

def create_region_mapping(zip_county_df, regions_list):
    """
    Create a mapping of zip codes to regions.

    Args:
    zip_county_df (pandas.DataFrame): DataFrame containing zip code and county information.
    regions_list (list): List of region names.

    Returns:
    dict: Mapping of zip codes to regions.
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
    Map zip codes to regions in the DataFrame.

    Args:
    df (pandas.DataFrame): Input DataFrame containing zip codes.
    zip_to_region (dict): Mapping of zip codes to regions.

    Returns:
    pandas.DataFrame: DataFrame with added region information.
    """
    df['Region'] = df['zip'].map(zip_to_region)
    df['Region'] = df['Region'].fillna('Other')
    return df

def main():
    """
    Main function to process median income data and create the final dataset.
    """
    # Process data for years 2018 to 2022
    years = range(2018, 2023)
    dfs = [process_year_data(year) for year in years]

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Process median income data
    processed_df = process_median_income(combined_df)

    # Calculate income growth rate
    result = processed_df.groupby('Area').apply(calculate_income_growth_rate).reset_index()
    processed_df = processed_df.merge(result, on='Area', how='left')

    # Filter and clean data
    processed_df = processed_df[processed_df['Growth_Rate_Reason'] == 'Calculated']
    processed_df = processed_df.drop(columns=['Median_Income_White', 'Median_Income_Black', 'Median_Income_Asian', 'Median_Income_Hispanic', 'Growth_Rate_Reason'])
    processed_df = processed_df.dropna()

    # Standardize zip codes
    processed_df = standardize_zip_codes(processed_df)
    processed_df = processed_df.rename(columns={'Area': 'zip'})

    # Load zip to county mapping
    zips_to_counties = pd.read_csv('uszips.xlsx - Sheet1.csv')
    zips_to_counties['zip'] = zips_to_counties['zip'].astype(str).str.zfill(5)

    # Add county information
    def get_majority_county(county_weights):
        try:
            weights = json.loads(county_weights.replace("'", '"'))
            return max(weights, key=weights.get)
        except:
            return None

    zip_to_county = {row['zip']: get_majority_county(row['county_weights']) for _, row in zips_to_counties.iterrows()}
    processed_df['county_fips'] = processed_df['zip'].map(zip_to_county)

    # Load regions list and create region mapping
    with open('regions_list.txt', 'r') as f:
        regions_list = f.read().splitlines()

    zip_to_region = create_region_mapping(zips_to_counties, regions_list)

    # Map zips to regions
    final_df = map_zips_to_regions(processed_df, zip_to_region)
    final_df = final_df.dropna()
    # Save the final dataset
    final_df.to_parquet('median_income.parquet')

if __name__ == "__main__":
    main()
