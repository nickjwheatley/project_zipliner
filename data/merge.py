"""
This script merges multiple datasets related to housing costs, median income,
operating businesses, population change, crime rates, overall population,
median age, and commute characteristics. It uses Dask for efficient processing
of large datasets and outputs a single merged parquet file.

The script handles data standardization, merging, and basic analysis of the
resulting dataset, including NaN percentage checks and memory usage estimation.
"""

import pandas as pd
import dask.dataframe as dd
import pyarrow

pd.set_option('display.max_columns', None)

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

def merge_datasets():
    """
    Merge multiple datasets using Dask for efficient processing.

    This function reads multiple parquet files, performs an outer join on
    specified columns, and writes the result to a new parquet file.

    Returns:
    dask.dataframe.DataFrame: Merged Dask dataframe.
    """
    # List of parquet files to merge
    file_list = [
        'housing_costs.parquet',
        'median_income.parquet',
        'operating_businesses.parquet',
        'population_change.parquet',
        'crime.parquet',
        'overall_population_per_county.parquet',
        'median_age.parquet',
        'commute.parquet'
    ]

    # Read each parquet file into a Dask dataframe
    dfs = [dd.read_parquet(file) for file in file_list]

    # Perform the outer join
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=['zip', 'county_fips', 'Region', 'Year'], how='outer')

    # Ensure 'Year' is of type Int64
    merged_df['Year'] = merged_df['Year'].astype('int64')

    # Compute the result and write to a new parquet file
    merged_df.to_parquet('merged_result.parquet', engine='pyarrow')

    return merged_df

def process_crime_data():
    """
    Process crime data from CSV, standardize zip codes, and save as parquet.

    This function reads crime data from a CSV file, standardizes the zip codes,
    and saves the result as a parquet file for later merging.
    """
    crime = pd.read_csv('crime_data_w_zip.xlsx - Sheet1.csv')
    crime['zip'] = crime['zip'].astype('str')
    crime = standardize_zip_codes(crime)
    crime.to_parquet('crime.parquet')

def print_merge_statistics(merged_df):
    """
    Print statistics about the merged dataset.

    Args:
    merged_df (dask.dataframe.DataFrame): The merged Dask dataframe.

    This function computes and prints the number of rows, columns,
    column names, and approximate memory usage of the merged dataset.
    """
    print("Number of rows:", len(merged_df.compute()))
    print("Number of columns:", len(merged_df.columns))
    print("Column names:", merged_df.columns)
    print("Approximate memory usage:", merged_df.memory_usage(deep=True).sum().compute() / 1e9, "GB")

def check_nan_percentage(df):
    """
    Calculate and print the percentage of NaN values in each column.

    Args:
    df (pandas.DataFrame): The dataframe to check for NaN values.
    """
    nan_percentage = df.isna().mean() * 100
    print("NaN percentage per column:")
    print(nan_percentage)

if __name__ == "__main__":
    # Process crime data
    process_crime_data()

    # Merge datasets
    merged_df = merge_datasets()

    # Print merge statistics
    print_merge_statistics(merged_df)

    # Load the merged result for further analysis
    final_df = pd.read_parquet('merged_result.parquet')

    # Check NaN percentage
    check_nan_percentage(final_df)

    print("Sample of the merged dataset:")
    print(final_df.sample(10))
