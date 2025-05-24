"""
Synthetic Control Method implementation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import warnings

from ..core.base import CausalResults, calculate_treatment_effect, validate_panel_data


def get_sc_weights(treated: str, donors: List[str], panel: pd.DataFrame, 
                   pre_treatment_years: List[int]) -> Tuple[Optional[Dict], Optional[float]]:
    """
    Calculate synthetic control weights using optimization.
    
    Parameters:
    -----------
    treated : str
        Name of treated country
    donors : list
        List of donor countries
    panel : DataFrame
        Panel data with countries as columns and years as index
    pre_treatment_years : list
        List of pre-treatment years
        
    Returns:
    --------
    tuple : (weights_dict, pre_treatment_rmse)
    """
    available_years = [y for y in pre_treatment_years if y in panel.index]
    available_donors = [d for d in donors if d in panel.columns]
    
    if not available_years or not available_donors:
        return None, None
    
    X_treated = panel.loc[available_years, treated].values
    X_donors = panel.loc[available_years, available_donors].values
    
    # Handle missing values
    mask = ~np.isnan(X_treated)
    if not np.any(mask):
        return None, None
    
    X_treated = X_treated[mask]
    X_donors = X_donors[mask]
    
    # Remove donors with missing values
    valid_donors = [i for i, col in enumerate(X_donors.T) if not np.any(np.isnan(col))]
    if not valid_donors:
        return None, None
    
    X_donors = X_donors[:, valid_donors]
    valid_donor_names = [available_donors[i] for i in valid_donors]
    
    # Define objective function (RMSE)
    def objective(w):
        return np.sqrt(np.mean((X_treated - X_donors @ w)**2))
    
    # Set up optimization problem
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(len(valid_donor_names))]
    
    # Initial weights (uniform)
    initial_weights = np.ones(len(valid_donor_names))/len(valid_donor_names)
    
    # Solve optimization problem
    result = minimize(
        objective,
        initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    # Create weights dictionary
    weights_dict = dict(zip(valid_donor_names, result.x))
    all_weights = {d: weights_dict.get(d, 0) for d in donors}
    
    return all_weights, result.fun


def run_synthetic_control(country: str, donor_pool: List[str], data_dict: Dict) -> Optional[Dict]:
    """
    Run synthetic control analysis for a single country.
    
    Parameters:
    -----------
    country : str
        Country name
    donor_pool : list
        List of donor countries
    data_dict : dict
        Dictionary with preprocessed data
        
    Returns:
    --------
    dict : Synthetic control results
    """
    panel = data_dict['panel']
    treatment_years = data_dict['treatment_years']
    
    if country not in treatment_years or country not in panel.columns:
        return None
    
    treatment_year = treatment_years[country]
    all_years = sorted(panel.index)
    pre_treatment_years = [y for y in all_years if y < treatment_year]
    post_treatment_years = [y for y in all_years if y >= treatment_year]
    
    # Get weights
    weights, pre_rmse = get_sc_weights(country, donor_pool, panel, pre_treatment_years)
    if weights is None:
        return None
    
    # Construct synthetic control
    synth_values = {}
    for year in all_years:
        if year in panel.index:
            donor_values = [weights[d] * panel.loc[year, d] 
                           if d in panel.columns and not pd.isna(panel.loc[year, d]) else 0
                           for d in weights.keys()]
            synth_values[year] = sum(donor_values)
    
    # Calculate treatment effects
    effect = {}
    for year in synth_values:
        if year in panel.index and not pd.isna(panel.loc[year, country]):
            effect[year] = panel.loc[year, country] - synth_values[year]
    
    # Calculate post-treatment RMSE
    post_years = [y for y in post_treatment_years if y in panel.index]
    post_values = [panel.loc[y, country] if not pd.isna(panel.loc[y, country]) 
                  else np.nan for y in post_years]
    synth_post_values = [synth_values[y] for y in post_years]
    
    valid_idx = [i for i, v in enumerate(post_values) 
                if not np.isnan(v) and not np.isnan(synth_post_values[i])]
    
    if valid_idx:
        post_rmse = np.sqrt(np.mean([(post_values[i] - synth_post_values[i])**2 
                                    for i in valid_idx]))
        att = np.mean([effect[post_years[i]] for i in valid_idx])
    else:
        post_rmse = np.nan
        att = np.nan
    
    # Calculate RMSE ratio
    rmse_ratio = post_rmse / pre_rmse if pre_rmse and pre_rmse > 0 else np.nan
    
    return {
        'country': country,
        'treatment_year': treatment_year,
        'weights': weights,
        'synthetic': synth_values,
        'effect': effect,
        'pre_rmse': pre_rmse,
        'post_rmse': post_rmse,
        'rmse_ratio': rmse_ratio,
        'att': att,
        'donor_pool': donor_pool
    }


def run_placebo_test_sc(country: str, sc_result: Dict, data_dict: Dict, 
                       n_placebos: int = 20) -> Dict:
    """
    Run placebo tests using each country's own donor pool from gemini data.
    
    Parameters:
    -----------
    country : str
        Country name for which SC was run
    sc_result : dict
        Synthetic control results
    data_dict : dict
        Dictionary with preprocessed data
    n_placebos : int
        Maximum number of placebo tests to run
        
    Returns:
    --------
    dict : Placebo test results
    """
    if sc_result is None:
        return {'p_value_ratio': np.nan, 'placebos': []}
    
    panel = data_dict['panel']
    treated_countries = data_dict['treated_countries']
    treatment_years = data_dict['treatment_years'].copy()
    gemini_df = data_dict.get('gemini_df')
    
    # Get weighted donors from original synthetic control
    weighted_donors = [donor for donor, weight in sc_result['weights'].items() 
                     if weight > 0.01]
    
    # Only use weighted donors as placebo countries
    placebo_countries = [d for d in weighted_donors 
                       if d not in treated_countries][:n_placebos]
    
    # If we don't have enough, add from the original donor pool
    if len(placebo_countries) < min(n_placebos, len(sc_result['donor_pool'])):
        additional = [d for d in sc_result['donor_pool'] 
                    if d not in placebo_countries and d not in treated_countries]
        placebo_countries.extend(additional[:n_placebos - len(placebo_countries)])
    
    print(f"Using {len(placebo_countries)} placebo countries for {country}")
    
    # Store placebo results
    placebo_results = []
    
    # Check which gemini data format we're using
    if gemini_df is not None:
        is_top_geminis = 'Gemini' in gemini_df.columns and 'Cluster' not in gemini_df.columns
        
        # Country name mapping for consistency
        country_map = {'Korea, Rep.': 'Korea', 'Russian Federation': 'Russia'}
        reverse_map = {v: k for k, v in country_map.items()}
        
        # Run placebo tests
        for placebo_country in placebo_countries:
            # Get this placebo country's own donor pool
            placebo_gemini_country = country_map.get(placebo_country, placebo_country)
            
            if is_top_geminis:
                gemini_matches = gemini_df[gemini_df['Country'] == placebo_gemini_country]['Gemini'].tolist()
                placebo_donors = [reverse_map.get(g, g) for g in gemini_matches 
                                 if reverse_map.get(g, g) in panel.columns]
            else:
                try:
                    row = gemini_df[gemini_df['Country'] == placebo_gemini_country]
                    if row.empty:
                        continue
                    
                    cluster = row['Cluster'].values[0]
                    cluster_countries = gemini_df[gemini_df['Cluster'] == cluster]['Country'].tolist()
                    placebo_donors = [reverse_map.get(c, c) for c in cluster_countries 
                                    if c != placebo_gemini_country and 
                                    reverse_map.get(c, c) in panel.columns]
                except Exception:
                    continue
            
            # Remove treated countries and the placebo country itself
            placebo_donors = [d for d in placebo_donors 
                            if d != placebo_country and d not in treated_countries]
            
            if len(placebo_donors) < 2:
                continue
            
            # Temporarily mark the placebo as treated
            treatment_years[placebo_country] = sc_result['treatment_year']
            
            try:
                temp_data_dict = data_dict.copy()
                temp_data_dict['treatment_years'] = treatment_years
                
                placebo_result = run_synthetic_control(
                    placebo_country, 
                    placebo_donors, 
                    temp_data_dict
                )
                
                if placebo_result and not np.isnan(placebo_result['rmse_ratio']):
                    placebo_results.append(placebo_result)
            
            except Exception:
                continue
    
    # Calculate p-values
    if placebo_results:
        ratio = sc_result['rmse_ratio']
        if np.isnan(ratio):
            p_value_ratio = np.nan
        else:
            p_value_ratio = sum(1 for p in placebo_results 
                              if p['rmse_ratio'] >= ratio) / len(placebo_results)
        
        return {
            'p_value_ratio': p_value_ratio,
            'placebos': placebo_results,
            'placebo_countries': [p['country'] for p in placebo_results]
        }
    else:
        return {
            'p_value_ratio': np.nan, 
            'placebos': [], 
            'placebo_countries': []
        }


def run_sc_analysis(data_dict: Dict, donor_pools: Dict[str, List[str]], 
                   n_placebos: int = 20) -> CausalResults:
    """
    Run synthetic control analysis for all treated countries.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    donor_pools : dict
        Dictionary mapping treated countries to donor pools
    n_placebos : int
        Number of placebo tests to run
        
    Returns:
    --------
    CausalResults : Standardized results object
    """
    results = CausalResults('Synthetic Control')
    
    treated_countries = data_dict['treated_countries']
    all_country_results = {}
    
    for country in treated_countries:
        print(f"Analyzing country: {country}")
        
        if country not in donor_pools or not donor_pools[country]:
            print(f"  No donor pool for {country}, skipping.")
            continue
        
        donor_pool = donor_pools[country]
        print(f"  Using {len(donor_pool)} donors")
        
        try:
            # Run synthetic control
            sc_result = run_synthetic_control(country, donor_pool, data_dict)
            
            if sc_result is None:
                print(f"  Failed to run SC for {country}, skipping.")
                continue
            
            # Run placebo test
            placebo_result = run_placebo_test_sc(country, sc_result, data_dict, n_placebos)
            
            # Store results
            all_country_results[country] = {
                'sc_result': sc_result,
                'placebo_result': placebo_result
            }
            
            # Add to results object
            country_metrics = {
                'att': sc_result['att'],
                'pre_rmse': sc_result['pre_rmse'],
                'post_rmse': sc_result['post_rmse'],
                'rmse_ratio': sc_result['rmse_ratio'],
                'p_value': placebo_result.get('p_value_ratio', np.nan),
                'significant': placebo_result.get('p_value_ratio', 1) <= 0.1 if placebo_result.get('p_value_ratio') is not None else False
            }
            results.add_country_result(country, country_metrics)
            
        except Exception as e:
            print(f"  Error analyzing {country}: {e}")
            continue
    
    # Create summary table
    summary_data = []
    for country, country_results in all_country_results.items():
        sc_result = country_results['sc_result']
        placebo_result = country_results['placebo_result']
        
        p_value = placebo_result.get('p_value_ratio', np.nan)
        
        summary_data.append({
            'country': country,
            'ATT': sc_result['att'],
            'RMSE pre': sc_result['pre_rmse'],
            'RMSE post': sc_result['post_rmse'],
            'relation': sc_result['rmse_ratio'],
            'p-val': p_value,
            'significant': 'Yes' if p_value is not None and p_value <= 0.1 else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('p-val')
    
    results.summary_table = summary_df
    results.raw_results = all_country_results
    
    return results


def create_weights_table(results: CausalResults) -> pd.DataFrame:
    """Create consolidated weights dataframe for all countries."""
    all_weights = []
    
    for country, country_data in results.raw_results.items():
        sc_result = country_data.get('sc_result')
        if sc_result and 'weights' in sc_result:
            weights = sc_result['weights']
            for donor, weight in weights.items():
                if weight > 0.01:  # Only include meaningful weights
                    all_weights.append({
                        'treated_country': country,
                        'donor_country': donor,
                        'weight': weight
                    })
    
    weights_df = pd.DataFrame(all_weights).sort_values(['treated_country', 'weight'], ascending=[True, False])
    return weights_df
