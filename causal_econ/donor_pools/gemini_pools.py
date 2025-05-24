"""
Donor pool generation using gemini similarity relationships.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from ..core.base import standardize_country_names, country_name_mapper


def create_expert_pools() -> Dict[str, List[str]]:
    """Define expert-knowledge based donor pools."""
    return {
        'United States': ['Canada', 'United Kingdom', 'Australia', 'Germany', 'France'],
        'China': ['Vietnam', 'Thailand', 'Malaysia', 'Indonesia', 'Philippines'],
        'Japan': ['Korea, Rep.', 'Singapore', 'Malaysia', 'Thailand'],
        'Germany': ['France', 'Netherlands', 'Italy', 'Belgium', 'Austria'],
        'India': ['Pakistan', 'Bangladesh', 'Indonesia', 'Philippines', 'Thailand'],
    }


def get_donor_pools(data_dict: Dict, 
                   method: str = 'e2v', 
                   n_donors: int = 20,
                   custom_pools: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """
    Create donor pools for each treated country.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data containing:
        - 'panel': Panel data DataFrame
        - 'treated_countries': List of treated countries
        - 'gemini_df': DataFrame with gemini relationships (optional)
    method : str
        Method to use for donor selection: 'e2v', 'expert', 'all', 'custom'
    n_donors : int
        Number of donors to include
    custom_pools : dict, optional
        Custom donor pools mapping {treated_country: [donor_countries]}
        
    Returns:
    --------
    dict : Dictionary mapping treated countries to donor pools
    """
    panel = data_dict['panel']
    treated_countries = data_dict['treated_countries']
    gemini_df = data_dict.get('gemini_df', None)
    
    # Expert donor pools
    expert_pools = create_expert_pools()
    
    # All untreated countries pool
    all_donors = {
        country: [c for c in panel.columns if c not in treated_countries and c != country]
        for country in treated_countries
    }
    
    # Return appropriate pools based on method
    if method == 'expert':
        return {country: expert_pools.get(country, all_donors[country][:n_donors]) 
                for country in treated_countries}
    
    elif method == 'all':
        return {country: all_donors[country][:n_donors] for country in treated_countries}
    
    elif method == 'custom':
        if custom_pools is None:
            raise ValueError("custom_pools must be provided when method='custom'")
        return custom_pools
    
    elif method == 'e2v' and gemini_df is not None:
        return _create_e2v_pools(treated_countries, gemini_df, panel, all_donors, n_donors)
    
    else:
        # Default to all donors
        print(f"Warning: Method '{method}' not recognized or gemini_df not provided. Using 'all' method.")
        return {country: all_donors[country][:n_donors] for country in treated_countries}


def _create_e2v_pools(treated_countries: List[str], 
                     gemini_df: pd.DataFrame,
                     panel: pd.DataFrame,
                     all_donors: Dict[str, List[str]],
                     n_donors: int) -> Dict[str, List[str]]:
    """Create E2V-based donor pools using gemini relationships."""
    
    e2v_pools = {}
    
    # Country name mapping for consistency
    country_map = country_name_mapper()
    reverse_map = {v: k for k, v in country_map.items()}
    
    for country in treated_countries:
        # Handle country name differences
        gemini_country = country_map.get(country, country)
        
        if gemini_country not in gemini_df['Country'].values:
            print(f"Warning: {gemini_country} not found in gemini data, using all donors")
            e2v_pools[country] = all_donors[country][:n_donors]
            continue
        
        # Check if we're using top_geminis.csv or gemini_clusters.csv
        if 'Gemini' in gemini_df.columns and 'Cluster' not in gemini_df.columns:
            # For top_geminis.csv (chains) - directly get the similar countries
            donors = gemini_df[gemini_df['Country'] == gemini_country]['Gemini'].tolist()
            if not donors:
                donors = all_donors[country][:n_donors]
        else:
            # For gemini_clusters.csv - find countries in same cluster
            row = gemini_df[gemini_df['Country'] == gemini_country]
            if row.empty:
                print(f"Warning: {gemini_country} not found in clusters, using all donors")
                e2v_pools[country] = all_donors[country][:n_donors]
                continue
                
            cluster = row['Cluster'].values[0]
            cluster_countries = gemini_df[(gemini_df['Country'] != gemini_country) & 
                                        (gemini_df['Cluster'] == cluster)]
            cluster_countries = cluster_countries.sort_values('Similarity_Score', ascending=False)
            donors = cluster_countries['Country'].tolist()
        
        # Map back to original names and filter to available countries
        donors = [reverse_map.get(d, d) for d in donors]
        donors = [d for d in donors if d in panel.columns and d not in treated_countries]
        
        e2v_pools[country] = donors[:n_donors]
    
    return e2v_pools


def create_donor_pools(data_dict: Dict,
                      gemini_df: Optional[pd.DataFrame] = None,
                      method: str = 'e2v',
                      n_donors: int = 20,
                      custom_pools: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """
    Comprehensive donor pool creation with multiple methods.
    
    Parameters:
    -----------
    data_dict : dict
        Preprocessed data dictionary
    gemini_df : pd.DataFrame, optional
        Gemini relationships DataFrame
    method : str
        Donor selection method
    n_donors : int
        Number of donors per treated country
    custom_pools : dict, optional
        Custom donor pool specification
        
    Returns:
    --------
    dict : Donor pools for each treated country
    """
    # Add gemini_df to data_dict if provided
    data_dict_with_gemini = data_dict.copy()
    if gemini_df is not None:
        data_dict_with_gemini['gemini_df'] = gemini_df
    
    donor_pools = get_donor_pools(
        data_dict_with_gemini,
        method=method,
        n_donors=n_donors,
        custom_pools=custom_pools
    )
    
    # Validate donor pools
    validated_pools = validate_donor_pools(donor_pools, data_dict)
    
    # Print summary
    print(f"\nDonor pools created using '{method}' method:")
    for country, donors in validated_pools.items():
        print(f"  {country}: {len(donors)} donors")
    
    return validated_pools


def validate_donor_pools(donor_pools: Dict[str, List[str]], 
                        data_dict: Dict) -> Dict[str, List[str]]:
    """
    Validate and clean donor pools.
    
    Parameters:
    -----------
    donor_pools : dict
        Raw donor pools
    data_dict : dict
        Data dictionary with panel information
        
    Returns:
    --------
    dict : Validated donor pools
    """
    panel = data_dict['panel']
    treated_countries = data_dict['treated_countries']
    
    validated_pools = {}
    
    for country, donors in donor_pools.items():
        # Filter donors to only include countries in panel
        valid_donors = [d for d in donors if d in panel.columns]
        
        # Remove treated countries from donor pool
        valid_donors = [d for d in valid_donors if d not in treated_countries]
        
        # Remove the country itself if somehow included
        valid_donors = [d for d in valid_donors if d != country]
        
        validated_pools[country] = valid_donors
        
        # Warn if no valid donors
        if not valid_donors:
            print(f"Warning: No valid donors found for {country}")
    
    return validated_pools


def get_donor_pool_info(donor_pools: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Get summary information about donor pools.
    
    Parameters:
    -----------
    donor_pools : dict
        Donor pools mapping
        
    Returns:
    --------
    pd.DataFrame : Summary information
    """
    info_rows = []
    
    for treated_country, donors in donor_pools.items():
        info_rows.append({
            'treated_country': treated_country,
            'n_donors': len(donors),
            'donors': ', '.join(donors[:5]) + ('...' if len(donors) > 5 else '')
        })
    
    return pd.DataFrame(info_rows)


def create_placebo_pools(donor_pools: Dict[str, List[str]], 
                        data_dict: Dict,
                        n_placebos: int = 20) -> Dict[str, List[str]]:
    """
    Create placebo pools for testing, ensuring each donor country 
    gets its own appropriate donor pool.
    
    Parameters:
    -----------
    donor_pools : dict
        Original donor pools for treated countries
    data_dict : dict
        Data dictionary
    n_placebos : int
        Number of placebo countries to use
        
    Returns:
    --------
    dict : Placebo pools for testing
    """
    treated_countries = data_dict['treated_countries']
    panel = data_dict['panel']
    gemini_df = data_dict.get('gemini_df', None)
    
    placebo_pools = {}
    
    for treated_country, donors in donor_pools.items():
        # Use top donors as placebo countries
        placebo_countries = donors[:min(n_placebos, len(donors))]
        
        # For each placebo country, create its own donor pool
        for placebo_country in placebo_countries:
            if gemini_df is not None:
                # Create a temporary data dict for this placebo
                temp_data_dict = {
                    'panel': panel,
                    'treated_countries': [placebo_country],  # Treat placebo as treated
                    'gemini_df': gemini_df
                }
                
                # Get donor pool for this placebo country
                placebo_donor_pools = get_donor_pools(temp_data_dict, method='e2v', n_donors=20)
                
                if placebo_country in placebo_donor_pools:
                    # Remove the original treated country and the placebo itself
                    placebo_donors = [d for d in placebo_donor_pools[placebo_country] 
                                    if d != treated_country and d != placebo_country]
                    placebo_pools[f"{placebo_country}_for_{treated_country}"] = placebo_donors
    
    return placebo_pools
