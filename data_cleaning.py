import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Callable, Union

def calculate_indicator_summary(
    df: pd.DataFrame, 
    indicators: Dict[str, str], 
    calculations: Union[str, List[str]] = 'median'
) -> pd.DataFrame:
    """
    Calculates summary statistics (like median, mean, etc.) across year columns 
    for specific indicators in a country data DataFrame.

    Args:
        df: The input DataFrame containing country and indicator data. 
        indicators: A dictionary mapping 'Series Code' (key) to a 
            descriptive indicator name (value).
        calculations: A string or list of strings specifying the calculations 
                      to perform (e.g., 'median', 'mean', 'min'). 
                      Uses 'median' by default.

    Returns:
        A DataFrame containing for the summarized data.
    """
    country_data = df[df['Series Code'].isin(indicators.keys())].copy()
    year_cols = country_data.columns[4:]
    country_data[year_cols] = country_data[year_cols].apply(
        pd.to_numeric, 
        errors='coerce'
    )
    

    # --- SUMMARIZING THE DATA ---
    if isinstance(calculations, str):
        calculations = [calculations]

    calc_map: Dict[str, Callable] = {
        'median': lambda df: df.median(axis=1),
        'mean': lambda df: df.mean(axis=1),
        'min': lambda df: df.min(axis=1),
        'max': lambda df: df.max(axis=1),
    }

    results = []

    for calc_name in calculations:
        if calc_name not in calc_map:
            print(f"Warning: Calculation type '{calc_name}' not supported. Skipping.")
            continue
            
        # Calculation across the year columns (axis=1)
        calculated_values = calc_map[calc_name](country_data[year_cols])
        
        # Create the result DataFrame slice for this calculation
        result_df = country_data[['Country Name', 'Country Code','Series Code']].copy()
        result_df['value'] = calculated_values
        result_df['calculation_type'] = calc_name # The new calculation column
        
        results.append(result_df)
        
    summary_df = pd.concat(results, ignore_index=True)
    summary_df['indicator'] = summary_df['Series Code'].map(indicators)
    if len(calculations)>1:
        summary_df['indicator'] = summary_df[['indicator', 'calculation_type']].agg('-'.join, axis=1)
    
    return summary_df


def summarize_data_wide(df: pd.DataFrame, 
    indicators: Dict[str, str], 
    calculations: Union[str, List[str]] = 'median'
) -> pd.DataFrame:
    """
    Calculates summary statistics for selected indicators and converts the result 
    to a wide-format DataFrame.

    Args:
        df: The input DataFrame with country, indicator ('Series Code'), and year data.
        indicators: A map from 'Series Code' to descriptive indicator names.
        calculations: A string or list of summary functions (e.g., 'median', 'mean') 
                      to apply across the year columns. Defaults to 'median'.

    Returns:
        A wide DataFrame with the summarized data
    """
    
    summarized_data = calculate_indicator_summary(df, indicators, calculations)

    sumarized_wide = summarized_data.pivot_table(
        index=['Country Name', 'Country Code'],
        columns='indicator',
        values='value',
        aggfunc='median'
    ).reset_index()

    return sumarized_wide

def input_missing_data(
    df: pd.DataFrame,
    region_df: pd.DataFrame, 
    verbose = False
) -> pd.DataFrame:
    """
    Fills missing indicator values in the main DataFrame (df) by using 
    corresponding region-level summary values from a separate DataFrame 
    (region_df).

    Args:
        df: The wide-format DataFrame containing country data, region assignments, 
            and potentially missing indicator columns.
        region_df: A DataFrame where the index or a column is the region name, 
                   and columns are the summarized indicator values for each region.
        verbose: Boolen to check which values are being filled

    Returns:
        A copy of the input DataFrame with missing indicator values filled 
        where a corresponding, non-missing regional aggregate value was available.
    """
    filled_df = df.copy()
    summarized_indicators = df.columns.values.tolist()
    filled_countries = [] # Fills missing country indicators from region-level values
    for ind in summarized_indicators:
        missing_idx = df[ind].isna()
        for idx in df[missing_idx].index:
            region = df.loc[idx, 'region']
            if pd.notna(region):
                if region in region_df['region'].values:
                    if ind in region_df.columns:
                        region_row = region_df[region_df['region'] == region]
                        region_value = region_row[ind].values[0]
                        if pd.notna(region_value):
                            filled_df.at[idx, ind] = region_value
                            filled_countries.append((df.loc[idx, 'Country Name'], ind, region))

    if verbose:
        print("\nFilled missing indicators from regions:")
        for c, ind, r in filled_countries:
            print(f"{c}: {ind} filled from {r}")
    
    return filled_df

def standardize(
    df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardizes the numeric indicator columns of the input DataFrame using 
    the StandardScaler (Z-score normalization).

    Args:
        df: The DataFrame containing country names, regions, and numeric indicator data.

    Returns:
        A tuple containing three NumPy arrays:
        1. values (np.ndarray): The Z-score standardized numeric data.
        2. countries (np.ndarray): An array of country names corresponding to the rows.
        3. regions (np.ndarray): An array of region names corresponding to the rows.
    """
    numeric_cols = [c for c in df.columns if c not in ['Country Name', 'Country Code', 'region']]
    values = StandardScaler().fit_transform(df[numeric_cols])
    countries = df['Country Name'].values
    regions = df['region'].values

    return values, countries, regions