"""
Data loading and preprocessing utilities for causal economic analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_data(file_path: str, 
              treatment_col: str = 'has_platform',
              outcome_col: str = 'gdp_growth_annual_share',
              country_col: str = 'country',
              year_col: str = 'year') -> Dict:
    """
    Load and basic validation of economic panel data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the data
    treatment_col : str
        Column name indicating treatment status
    outcome_col : str
        Column name for the outcome variable
    country_col : str
        Column name for country identifier
    year_col : str
        Column name for year identifier
        
    Returns:
    --------
    dict : Data dictionary with basic structure
    """
    # Load data
    df = pd.read_csv(file_path)
    df[year_col] = df[year_col].astype(int)
    
    print(f"Loaded data: {df.shape}")
    print(f"Years: {df[year_col].min()} - {df[year_col].max()}")
    print(f"Countries: {df[country_col].nunique()}")
    
    # Validate required columns
    required_cols = [country_col, year_col, treatment_col, outcome_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return {
        'df': df,
        'treatment_col': treatment_col,
        'outcome_col': outcome_col,
        'country_col': country_col,
        'year_col': year_col
    }


def preprocess_panel_data(data_dict: Dict, 
                         min_periods: int = 5,
                         fill_method: str = 'ffill_bfill') -> Dict:
    """
    Preprocess panel data for causal analysis.
    
    Parameters:
    -----------
    data_dict : dict
        Data dictionary from load_data()
    min_periods : int
        Minimum number of periods required per country
    fill_method : str
        Method for handling missing values ('ffill_bfill', 'zero', 'drop')
        
    Returns:
    --------
    dict : Enhanced data dictionary with panel structure and metadata
    """
    df = data_dict['df'].copy()
    treatment_col = data_dict['treatment_col']
    outcome_col = data_dict['outcome_col']
    country_col = data_dict['country_col']
    year_col = data_dict['year_col']
    
    # Identify feature columns (exclude identifiers and treatment)
    id_cols = [country_col, year_col, treatment_col]
    feature_cols = [col for col in df.columns if col not in id_cols]
    
    # Sort by country and year
    df = df.sort_values([country_col, year_col])
    
    # Handle missing values
    if fill_method == 'ffill_bfill':
        df[feature_cols] = df.groupby(country_col)[feature_cols].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        # Fill any remaining NaNs with 0
        df[feature_cols] = df[feature_cols].fillna(0)
    elif fill_method == 'zero':
        df[feature_cols] = df[feature_cols].fillna(0)
    elif fill_method == 'drop':
        df = df.dropna(subset=feature_cols)
    
    # Create panel structure
    panel = df.pivot_table(index=year_col, columns=country_col, values=outcome_col)
    has_platform = df.pivot_table(index=year_col, columns=country_col, values=treatment_col).fillna(0)
    
    # Identify treated countries and treatment years
    treated_countries = []
    treatment_years = {}
    
    for country in has_platform.columns:
        platform_years = has_platform[country][has_platform[country] > 0]
        if len(platform_years) > 0:
            treatment_year = platform_years.index[0]
            treated_countries.append(country)
            treatment_years[country] = treatment_year
    
    # Create country metadata
    countries = df[country_col].unique()
    country_metadata = {}
    
    for country in countries:
        country_data = df[df[country_col] == country]
        years = country_data[year_col].values
        
        if country in treatment_years:
            treat_year = treatment_years[country]
            pre_treat_periods = sum(years < treat_year)
            is_treated = True
        else:
            treat_year = None
            pre_treat_periods = len(years)
            is_treated = False
            
        country_metadata[country] = {
            'treatment_year': treat_year,
            'pre_treatment_periods': pre_treat_periods,
            'is_treated': is_treated,
            'years': years,
            'total_periods': len(years)
        }
    
    # Filter countries with insufficient data
    valid_countries = [c for c, m in country_metadata.items() 
                      if m['total_periods'] >= min_periods]
    df = df[df[country_col].isin(valid_countries)]
    
    # Update structures after filtering
    panel = panel[valid_countries]
    country_metadata = {c: m for c, m in country_metadata.items() if c in valid_countries}
    treated_countries = [c for c in treated_countries if c in valid_countries]
    
    print(f"After preprocessing:")
    print(f"  Valid countries: {len(valid_countries)}")
    print(f"  Treated countries: {len(treated_countries)}")
    print(f"  Control countries: {len(valid_countries) - len(treated_countries)}")
    print(f"  Feature columns: {len(feature_cols)}")
    
    return {
        'df': df,
        'panel': panel,
        'has_platform': has_platform,
        'treated_countries': treated_countries,
        'treatment_years': treatment_years,
        'country_metadata': country_metadata,
        'feature_cols': feature_cols,
        'treatment_col': treatment_col,
        'outcome_col': outcome_col,
        'country_col': country_col,
        'year_col': year_col
    }


def create_gemini_data_dict(data_dict: Dict, gemini_df_path: Optional[str] = None) -> Dict:
    """
    Create data dictionary compatible with gemini donor pool functions.
    
    Parameters:
    -----------
    data_dict : dict
        Preprocessed data dictionary
    gemini_df_path : str, optional
        Path to gemini data (top_geminis.csv or gemini_clusters.csv)
        
    Returns:
    --------
    dict : Data dictionary for gemini functions
    """
    gemini_df = None
    if gemini_df_path:
        gemini_df = pd.read_csv(gemini_df_path)
    
    return {
        'panel': data_dict['panel'],
        'treated_countries': data_dict['treated_countries'],
        'gemini_df': gemini_df,
        'df': data_dict['df']
    }
