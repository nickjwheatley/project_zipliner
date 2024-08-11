import pandas as pd
import numpy as np
import csv
import json

def process_median_income_data(file_path):
    median_income_data_dict = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in csv_reader:
            if len(row) == 2:
                median_income_data_dict[row[0]] = row[1]
    return median_income_data_dict

def process_year_data(year):
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
    df[zip_column] = df[zip_column].astype(str)
    df[zip_column] = df[zip_column].apply(lambda x: x.zfill(5))
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

def process_median_income_data():
    # Process data for years 2018 to 2022
    years = range(2018, 2023)
    dfs = [process_year_data(year) for year in years]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove columns with 'Margin of Error'
    combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('Margin of Error', case=False)]

    # Process median income
    processed_df = process_median_income(combined_df)
    processed_df = processed_df.rename(columns={'Area': 'zip'})

    # Remove Puerto Rico zip codes
    puerto_rico_zip_codes = ['00601', '00602', '00603', '00606', '00610', '00611', '00612', '00616', '00617', '00622', '00623', '00624', '00627', '00631', '00636', '00637', '00638', '00641', '00646', '00647', '00650', '00652', '00653', '00656', '00659', '00660', '00662', '00664', '00667', '00669', '00670', '00674', '00676', '00677', '00678', '00680', '00682', '00683', '00685', '00687', '00688', '00690', '00692', '00693', '00694', '00698', '00703', '00704', '00705', '00707', '00714', '00715', '00716', '00717', '00718', '00719', '00720', '00723', '00725', '00727', '00728', '00729', '00730', '00731', '00735', '00736', '00738', '00739', '00740', '00741', '00745', '00751', '00754', '00757', '00765', '00766', '00767', '00769', '00771', '00772', '00773', '00775', '00777', '00778', '00780', '00782', '00783', '00784', '00786', '00791', '00794', '00795', '00901', '00906', '00907', '00909', '00911', '00912', '00913', '00915', '00917', '00918', '00920', '00921', '00923', '00924', '00925', '00926', '00927', '00934', '00936', '00949', '00950', '00951', '00952', '00953', '00956', '00957', '00959', '00960', '00961', '00962', '00965', '00966', '00968', '00969', '00971', '00976', '00979', '00982', '00983', '00985', '00987']
    processed_df = processed_df[~processed_df['zip'].isin(puerto_rico_zip_codes)]

    # Calculate income growth rate
    result = processed_df.groupby('zip').apply(calculate_income_growth_rate).reset_index()
    processed_df = processed_df.merge(result, on='zip', how='left')
    processed_df = processed_df[processed_df['Growth_Rate_Reason'] == 'Calculated']

    # Drop unnecessary columns
    processed_df = processed_df.drop(columns=['Median_Income_White', 'Median_Income_Black', 'Median_Income_Asian', 'Median_Income_Hispanic', 'Growth_Rate_Reason'])

    # Remove rows with NaN values
    processed_df = processed_df.dropna()

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

    return final_df

if __name__ == "__main__":
    final_df = process_median_income_data()
    final_df.to_parquet('median_income.parquet')