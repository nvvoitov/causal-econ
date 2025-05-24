"""
Difference-in-Differences with Propensity Score Matching implementation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings

from ..core.base import CausalResults, calculate_treatment_effect, validate_panel_data


def calculate_propensity_scores(data_dict: Dict, treated_country: str, donor_pool: List[str], 
                               feature_cols: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate propensity scores for potential matches using logistic regression.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    treated_country : str
        Name of the treated country
    donor_pool : list
        List of potential donor countries
    feature_cols : list, optional
        List of feature columns to use for matching
        
    Returns:
    --------
    dict : Dictionary mapping countries to propensity scores
    """
    df = data_dict['df']
    treatment_year = data_dict['treatment_years'][treated_country]
    
    # Use only pre-treatment data
    pre_treatment_data = df[df['year'] < treatment_year]
    
    # If no feature columns specified, use everything except identifiers
    if feature_cols is None:
        exclude_cols = ['year', 'country', 'has_platform', 'gdp_growth_annual_share']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Calculate average pre-treatment values for features
    countries = [treated_country] + donor_pool
    pre_data = []
    
    for country in countries:
        country_data = pre_treatment_data[pre_treatment_data['country'] == country]
        if len(country_data) > 0:
            avg_values = country_data[feature_cols].mean().to_dict()
            avg_values['country'] = country
            avg_values['is_treated'] = country == treated_country
            pre_data.append(avg_values)
    
    # Create dataframe and prepare for modeling
    pre_df = pd.DataFrame(pre_data)
    
    if len(pre_df) < 2:
        print(f"Not enough data for {treated_country} and donors")
        return {}
    
    # Prepare feature matrix
    X = pre_df[feature_cols].fillna(0)
    y = pre_df['is_treated']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression model
    try:
        model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
        model.fit(X_scaled, y)
        
        # Calculate propensity scores
        scores = model.predict_proba(X_scaled)[:, 1]
        
        # Create dictionary mapping countries to scores
        propensity_scores = dict(zip(pre_df['country'], scores))
        
        return propensity_scores
    except Exception as e:
        print(f"Error calculating propensity scores for {treated_country}: {e}")
        return {}


def select_best_match(data_dict: Dict, treated_country: str, donor_pool: List[str], 
                     propensity_scores: Optional[Dict[str, float]] = None,
                     feature_cols: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[float], Dict]:
    """
    Select the country with highest propensity score as the match.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    treated_country : str
        Name of the treated country
    donor_pool : list
        List of potential donor countries
    propensity_scores : dict, optional
        Dictionary mapping countries to propensity scores
    feature_cols : list, optional
        List of feature columns to use for matching
        
    Returns:
    --------
    tuple : (best_match, propensity_score, propensity_scores)
    """
    # Calculate propensity scores if not provided
    if propensity_scores is None:
        propensity_scores = calculate_propensity_scores(data_dict, treated_country, donor_pool, feature_cols)
    
    # Filter to only include donor countries
    donor_scores = {country: score for country, score in propensity_scores.items() 
                   if country in donor_pool}
    
    # Select country with highest score
    if donor_scores:
        best_match = max(donor_scores.items(), key=lambda x: x[1])
        return best_match[0], best_match[1], propensity_scores
    else:
        return None, None, propensity_scores


def calculate_parallel_shift(data_dict: Dict, treated_country: str, synth_match: str, 
                            treatment_year: Optional[int] = None) -> float:
    """
    Calculate parallel shift constant to align pre-treatment trends.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    treated_country : str
        Name of the treated country
    synth_match : str
        Name of the synthetic match country
    treatment_year : int, optional
        Treatment year (if None, taken from data_dict)
        
    Returns:
    --------
    float : Parallel shift constant
    """
    panel = data_dict['panel']
    
    # Check if countries exist in panel
    if treated_country not in panel.columns:
        raise ValueError(f"Treated country '{treated_country}' not found in panel data")
    if synth_match not in panel.columns:
        raise ValueError(f"Synthetic match '{synth_match}' not found in panel data")
    
    # Get treatment year if not provided
    if treatment_year is None:
        treatment_year = data_dict['treatment_years'][treated_country]
    
    # Get pre-treatment data
    pre_years = [y for y in panel.index if y < treatment_year]
    
    if not pre_years:
        raise ValueError(f"No pre-treatment data available for {treated_country}")
    
    # Calculate average difference in pre-treatment period
    treated_pre_data = panel.loc[pre_years, treated_country]
    match_pre_data = panel.loc[pre_years, synth_match]
    
    # Remove any NaN values
    valid_years = [y for y in pre_years if not pd.isna(treated_pre_data[y]) and not pd.isna(match_pre_data[y])]
    
    if not valid_years:
        raise ValueError(f"No valid pre-treatment data for both {treated_country} and {synth_match}")
    
    treated_pre = treated_pre_data.loc[valid_years].mean()
    match_pre = match_pre_data.loc[valid_years].mean()
    
    # Calculate parallel shift
    shift = treated_pre - match_pre
    
    return shift


def run_did_analysis(data_dict: Dict, treated_country: str, synth_match: str) -> Dict:
    """
    Run DiD analysis for a treated country and its synthetic match.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    treated_country : str
        Name of treated country
    synth_match : str
        Name of the synthetic match country
        
    Returns:
    --------
    dict : DiD analysis results
    """
    panel = data_dict['panel']
    
    # Check if countries exist in panel
    if treated_country not in panel.columns:
        raise ValueError(f"Treated country '{treated_country}' not found in panel data")
    if synth_match not in panel.columns:
        raise ValueError(f"Synthetic match '{synth_match}' not found in panel data")
    
    treatment_year = data_dict['treatment_years'][treated_country]
    
    # Get data for treated and control
    treated_data = panel[treated_country]
    control_data = panel[synth_match]
    
    # Make sure we have enough data
    valid_years = [y for y in panel.index if not pd.isna(treated_data[y]) and not pd.isna(control_data[y])]
    if len(valid_years) < 5:  # Minimum data requirement
        raise ValueError(f"Not enough valid data points for {treated_country} and {synth_match}")
    
    # Calculate parallel shift
    try:
        shift = calculate_parallel_shift(data_dict, treated_country, synth_match)
    except Exception as e:
        raise ValueError(f"Error calculating parallel shift: {str(e)}")
    
    # Apply shift to control data
    shifted_control = control_data + shift
    
    # Separate pre and post treatment periods
    all_years = sorted(panel.index)
    pre_treatment_years = [y for y in all_years if y < treatment_year]
    post_treatment_years = [y for y in all_years if y >= treatment_year]
    
    # Make sure we have both pre and post treatment data
    if len([y for y in pre_treatment_years if y in valid_years]) < 2:
        raise ValueError(f"Not enough pre-treatment data for {treated_country} and {synth_match}")
    if len([y for y in post_treatment_years if y in valid_years]) < 2:
        raise ValueError(f"Not enough post-treatment data for {treated_country} and {synth_match}")
    
    # Calculate effect
    effect = {}
    for year in all_years:
        if year in panel.index and not pd.isna(treated_data[year]) and not pd.isna(shifted_control[year]):
            effect[year] = treated_data[year] - shifted_control[year]
    
    # Calculate pre and post treatment RMSE
    pre_years = [y for y in pre_treatment_years if y in effect]
    post_years = [y for y in post_treatment_years if y in effect]
    
    pre_rmse = np.sqrt(np.mean([effect[y]**2 for y in pre_years])) if pre_years else np.nan
    post_rmse = np.sqrt(np.mean([effect[y]**2 for y in post_years])) if post_years else np.nan
    
    # Calculate average treatment effect on treated (ATT)
    att = np.mean([effect[y] for y in post_years]) if post_years else np.nan
    
    # Calculate RMSE ratio
    rmse_ratio = post_rmse / pre_rmse if pre_rmse and pre_rmse > 0 else np.nan
    
    # Store results
    return {
        'country': treated_country,
        'synth_match': synth_match,
        'treatment_year': treatment_year,
        'shift': shift,
        'effect': effect,
        'pre_rmse': pre_rmse,
        'post_rmse': post_rmse,
        'rmse_ratio': rmse_ratio,
        'att': att,
        'treated_values': treated_data.to_dict(),
        'control_values': control_data.to_dict(),
        'shifted_control': shifted_control.to_dict()
    }


def run_placebo_tests_did(data_dict: Dict, treated_country: str, donor_pool: List[str], 
                         did_result: Dict, n_placebos: int = 20,
                         feature_cols: Optional[List[str]] = None) -> Dict:
    """
    Run placebo tests using donor countries as placebo treated units.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    treated_country : str
        Name of the treated country
    donor_pool : list
        List of donor countries
    did_result : dict
        DiD analysis results for the treated country
    n_placebos : int
        Number of placebo tests to run
    feature_cols : list, optional
        List of feature columns for PSM
        
    Returns:
    --------
    dict : Placebo test results
    """
    panel = data_dict['panel']
    treatment_year = data_dict['treatment_years'][treated_country]
    treated_countries = data_dict['treated_countries']
    
    # Use only the top donors for placebo tests
    placebo_countries = [c for c in donor_pool if c in panel.columns][:min(n_placebos, len(donor_pool))]
    
    # Store placebo results
    placebo_results = []
    
    for placebo_country in placebo_countries:
        # Create a new donor pool for the placebo country,
        # excluding the original treated country and the placebo country itself
        placebo_donor_pool = [d for d in donor_pool 
                            if d != placebo_country 
                            and d != treated_country
                            and d in panel.columns]
        
        if len(placebo_donor_pool) < 2:
            continue
        
        # Create a copy of the data dictionary to avoid modifying the original
        placebo_data_dict = data_dict.copy()
        
        # Create a copy of the treatment years
        placebo_treatment_years = data_dict['treatment_years'].copy()
        
        # Set the treatment year for the placebo country
        placebo_treatment_years[placebo_country] = treatment_year
        placebo_data_dict['treatment_years'] = placebo_treatment_years
        
        try:
            # Calculate propensity scores for the placebo country
            propensity_scores = calculate_propensity_scores(placebo_data_dict, 
                                                          placebo_country, 
                                                          placebo_donor_pool,
                                                          feature_cols)
            
            if not propensity_scores:
                continue
                
            # Select best match for the placebo country
            best_match, score, _ = select_best_match(placebo_data_dict, 
                                                   placebo_country, 
                                                   placebo_donor_pool, 
                                                   propensity_scores,
                                                   feature_cols)
            
            if best_match is None:
                continue
            
            # Run DiD analysis treating the placebo country as treated
            placebo_result = run_did_analysis(placebo_data_dict, placebo_country, best_match)
            
            # Store placebo result
            placebo_results.append(placebo_result)
            
        except Exception:
            continue
    
    # Calculate p-values
    if placebo_results:
        # For RMSE ratio
        ratio = did_result['rmse_ratio']
        if np.isnan(ratio):
            p_value_ratio = np.nan
        else:
            p_value_ratio = sum(1 for p in placebo_results 
                              if not np.isnan(p['rmse_ratio']) and p['rmse_ratio'] >= ratio) / len(placebo_results)
        
        return {
            'p_value_ratio': p_value_ratio,
            'placebos': placebo_results,
            'placebo_countries': [p['country'] for p in placebo_results]
        }
    else:
        return {'p_value_ratio': np.nan, 'placebos': [], 'placebo_countries': []}


def run_did_analysis_pipeline(data_dict: Dict, donor_pools: Dict[str, List[str]], 
                             n_placebos: int = 20, feature_cols: Optional[List[str]] = None) -> CausalResults:
    """
    Run complete DiD analysis pipeline for all treated countries.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    donor_pools : dict
        Dictionary mapping treated countries to donor pools
    n_placebos : int
        Number of placebo tests to run
    feature_cols : list, optional
        List of feature columns to use for propensity score matching
        
    Returns:
    --------
    CausalResults : Standardized results object
    """
    results = CausalResults('Difference-in-Differences')
    
    treated_countries = data_dict['treated_countries']
    panel = data_dict['panel']
    all_results = {}
    
    for country in treated_countries:
        print(f"Analyzing country: {country}")
        
        # Check if treated country is in panel
        if country not in panel.columns:
            print(f"  Country {country} not found in panel data, skipping.")
            continue
        
        # Skip if no donor pool
        if country not in donor_pools or not donor_pools[country]:
            print(f"  No donor pool for {country}, skipping.")
            continue
        
        # Get donor pool
        donor_pool = donor_pools[country]
        if len(donor_pool) < 2:
            print(f"  Not enough donors for {country}, skipping.")
            continue
            
        print(f"  Using {len(donor_pool)} donors")
        
        try:
            # Calculate propensity scores
            propensity_scores = calculate_propensity_scores(data_dict, country, donor_pool, feature_cols)
            
            if not propensity_scores:
                print(f"  Failed to calculate propensity scores for {country}, skipping.")
                continue
                
            # Select best match
            best_match, score, _ = select_best_match(data_dict, country, donor_pool, propensity_scores, feature_cols)
            
            if best_match is None:
                print(f"  Could not find suitable match for {country}, skipping.")
                continue
            
            print(f"  Best match: {best_match} (score: {score:.4f})")
            
            # Check if best match is in panel
            if best_match not in panel.columns:
                print(f"  Match {best_match} not found in panel data, skipping.")
                continue
            
            # Run DiD analysis
            try:
                did_result = run_did_analysis(data_dict, country, best_match)
                
                if did_result is None:
                    print(f"  Failed to run DiD for {country}, skipping.")
                    continue
                
                # Run placebo test
                placebo_result = run_placebo_tests_did(data_dict, country, donor_pool, did_result, n_placebos, feature_cols)
                
                # Store results
                all_results[country] = {
                    'did_result': did_result,
                    'placebo_result': placebo_result,
                    'match_info': {
                        'best_match': best_match,
                        'score': score,
                        'propensity_scores': propensity_scores
                    }
                }
                
                # Add to results object
                country_metrics = {
                    'att': did_result['att'],
                    'pre_rmse': did_result['pre_rmse'],
                    'post_rmse': did_result['post_rmse'],
                    'rmse_ratio': did_result['rmse_ratio'],
                    'p_value': placebo_result.get('p_value_ratio', np.nan),
                    'significant': placebo_result.get('p_value_ratio', 1) <= 0.1 if placebo_result.get('p_value_ratio') is not None else False
                }
                results.add_country_result(country, country_metrics)
                
            except ValueError as ve:
                print(f"  {ve}")
                continue
                
        except Exception as e:
            print(f"  Error analyzing {country}: {str(e)}")
            continue
    
    # Create summary tables
    summary_data = []
    for country, country_results in all_results.items():
        did_result = country_results['did_result']
        placebo_result = country_results['placebo_result']
        
        p_value = placebo_result.get('p_value_ratio', np.nan)
        
        summary_data.append({
            'country': country,
            'ATT': did_result['att'],
            'RMSE pre': did_result['pre_rmse'],
            'RMSE post': did_result['post_rmse'],
            'RMSE ratio': did_result['rmse_ratio'],
            'p-val': p_value,
            'significant': 'Yes' if p_value is not None and p_value <= 0.1 else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('p-val')
    
    results.summary_table = summary_df
    results.raw_results = all_results
    
    return results


def create_match_table(results: CausalResults) -> pd.DataFrame:
    """Create table showing propensity scores and selected matches."""
    match_data = []
    
    for country, country_data in results.raw_results.items():
        did_result = country_data.get('did_result')
        match_info = country_data.get('match_info', {})
        
        if did_result:
            match_data.append({
                'treated_country': country,
                'synth_match': did_result['synth_match'],
                'propensity_score': match_info.get('score', np.nan),
                'treatment_year': did_result['treatment_year'],
                'parallel_shift': did_result['shift']
            })
    
    match_df = pd.DataFrame(match_data)
    if not match_df.empty:
        match_df = match_df.sort_values('propensity_score', ascending=False)
    
    return match_df
