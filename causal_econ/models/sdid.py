"""
Synthetic Difference-in-Differences implementation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, List, Optional, Tuple
import warnings

from ..core.base import CausalResults, calculate_treatment_effect, validate_panel_data


def calculate_regularization(data: pd.DataFrame, outcome_col: str, year_col: str, 
                           state_col: str, treat_col: str, post_col: str) -> float:
    """
    Calculate the regularization parameter ζ for unit weights according to SDID methodology.
    
    Parameters:
    -----------
    data : DataFrame
        Panel data
    outcome_col, year_col, state_col, treat_col, post_col : str
        Column names for outcome, year, unit/state, treatment indicator, post-treatment indicator
    
    Returns:
    --------
    float : regularization parameter ζ
    """
    # Get the number of treated units (count unique units, not observations)
    n_treated = data[data[treat_col]][state_col].nunique()
    
    # Get the number of post-treatment periods (count unique periods, not observations)
    n_post = data[data[post_col]][year_col].nunique()
    
    # Compute pre-period first differences for unexposed units
    control_pre = data[~data[treat_col] & ~data[post_col]].sort_values([state_col, year_col])
    
    # Calculate first differences within each control unit
    diffs = []
    for unit, group in control_pre.groupby(state_col):
        if len(group) > 1:
            sorted_group = group.sort_values(year_col)
            unit_diffs = sorted_group[outcome_col].diff().dropna().values
            diffs.extend(unit_diffs)
    
    # Compute standard deviation of first differences
    sigma = np.std(diffs) if diffs else 1.0
    
    # Calculate regularization parameter: ζ = (Nᵗʳ*Tᵖᵒˢᵗ)^(1/4) * σ
    zeta = (n_treated * n_post)**(1/4) * sigma
    
    return zeta


def calculate_unit_weights(data: pd.DataFrame, outcome_col: str, year_col: str, 
                         state_col: str, treat_col: str, post_col: str) -> Tuple[float, Dict[str, float]]:
    """
    Calculate unit weights for SDiD following the optimization problem.
    
    Parameters:
    -----------
    data : DataFrame
        Panel data
    outcome_col, year_col, state_col, treat_col, post_col : str
        Column names
    
    Returns:
    --------
    tuple : (omega_0, omega_dict) - intercept and unit weights dictionary
    """
    
    # Calculate regularization parameter
    zeta = calculate_regularization(data, outcome_col, year_col, state_col, treat_col, post_col)
    
    # Get pre-treatment data for control units
    control_pre = data[(~data[treat_col]) & (~data[post_col])]
    
    # Get pre-treatment data for treated units
    treated_pre = data[(data[treat_col]) & (~data[post_col])]
    
    # Create matrices for optimization
    control_units = control_pre[state_col].unique()
    treated_units = treated_pre[state_col].unique()
    pre_years = control_pre[year_col].unique()
    
    # Pivot data to create matrices
    control_matrix = control_pre.pivot(index=year_col, columns=state_col, values=outcome_col)
    treated_matrix = treated_pre.pivot(index=year_col, columns=state_col, values=outcome_col)
    
    # Ensure both matrices have the same years as index
    common_years = sorted(set(control_matrix.index).intersection(set(treated_matrix.index)))
    if len(common_years) < len(pre_years):
        print(f"Warning: Only {len(common_years)} out of {len(pre_years)} pre-treatment years have data for both control and treated units")
    
    control_matrix = control_matrix.loc[common_years]
    treated_matrix = treated_matrix.loc[common_years]
    
    # Average across treated units for each pre-treatment year
    treated_means = treated_matrix.mean(axis=1).values
    
    # Handle missing values in control matrix
    for col in control_matrix.columns:
        col_mean = control_matrix[col].mean()
        control_matrix[col] = control_matrix[col].fillna(col_mean)
    
    # If any NaNs remain, fill with overall mean
    control_matrix = control_matrix.fillna(control_matrix.values.mean())
    
    # Define the objective function
    def objective(params):
        omega_0 = params[0]
        omega = params[1:]
        
        # Weighted sum of control units for each year
        weighted_controls = control_matrix.values @ omega
        
        # Difference from treated mean
        diff = weighted_controls + omega_0 - treated_means
        
        # Compute sum of squared differences + regularization
        return np.sum(diff**2) + zeta**2 * len(common_years) * np.sum(omega**2)
    
    # Constraints: weights sum to 1 and are non-negative
    n_controls = len(control_units)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1}]
    bounds = [(None, None)] + [(0, None)] * n_controls
    
    # Initial guess: intercept 0, equal weights
    initial_weights = np.zeros(n_controls + 1)
    initial_weights[1:] = 1.0 / n_controls
    
    # Solve the optimization problem
    result = minimize(
        objective,
        initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    # Extract results
    omega_0 = result.x[0]
    omega_values = result.x[1:]
    omega_dict = dict(zip(control_units, omega_values))
    
    return omega_0, omega_dict


def calculate_time_weights(data: pd.DataFrame, outcome_col: str, year_col: str, 
                         state_col: str, treat_col: str, post_col: str) -> Tuple[float, Dict[int, float]]:
    """
    Calculate time weights for SDiD according to equation (2.3) in the paper.
    
    Parameters:
    -----------
    data : DataFrame
        Panel data
    outcome_col, year_col, state_col, treat_col, post_col : str
        Column names
    
    Returns:
    --------
    tuple : (lambda_0, lambda_dict) - intercept and time weights dictionary
    """
    
    # Get data for control units
    control_data = data[~data[treat_col]]
    
    # Pre and post treatment periods 
    pre_periods = sorted(control_data[~control_data[post_col]][year_col].unique())
    post_periods = sorted(control_data[control_data[post_col]][year_col].unique())
    
    # Create matrices - use pivot for more direct and readable code
    pivot_control = control_data.pivot_table(
        index=state_col, 
        columns=year_col, 
        values=outcome_col,
        aggfunc='mean'  # Handles duplicate entries
    )
    
    # Extract pre and post period columns
    pre_cols = [col for col in pivot_control.columns if col in pre_periods]
    post_cols = [col for col in pivot_control.columns if col in post_periods]
    
    # Skip units without complete data
    pivot_control_filtered = pivot_control.dropna(subset=post_cols, how='all')
    pivot_control_filtered = pivot_control_filtered.dropna(subset=pre_cols, how='all')
    
    # If no units with both pre and post data, use equal weights
    if len(pivot_control_filtered) == 0:
        lambda_0 = 0
        lambda_dict = {p: 1.0/len(pre_periods) for p in pre_periods}
        return lambda_0, lambda_dict
    
    # Calculate post-period means for each control unit
    Y_post = pivot_control_filtered[post_cols].mean(axis=1).values
    
    # Extract pre-period matrix (potentially with NaNs)
    Y_pre_with_na = pivot_control_filtered[pre_cols].values
    
    # Fill NaNs in Y_pre with column means (period-specific means)
    Y_pre = Y_pre_with_na.copy()
    for j in range(Y_pre.shape[1]):
        col = Y_pre[:, j]
        mask = np.isnan(col)
        if mask.all():
            # If entire column is NaN, set to 0 (will be handled by intercept)
            Y_pre[:, j] = 0
        elif mask.any():
            # Replace NaNs with column mean
            col_mean = np.nanmean(col)
            col[mask] = col_mean
            Y_pre[:, j] = col
    
    # Define the objective function
    def objective(params):
        lambda_0 = params[0]
        lambda_pre = params[1:]
        
        # Weighted pre-period values for each unit
        weighted_pre = Y_pre @ lambda_pre
        
        # Difference from post-period mean
        diff = lambda_0 + weighted_pre - Y_post
        
        # Compute sum of squared differences
        return np.sum(diff**2)
    
    # Constraints: pre-period weights sum to 1 and are non-negative
    n_pre = len(pre_cols)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[1:]) - 1}]
    bounds = [(None, None)] + [(0, None)] * n_pre
    
    # Initial guess: intercept 0, equal weights
    initial_params = np.zeros(n_pre + 1)
    initial_params[1:] = 1.0 / n_pre
    
    # Solve the optimization problem
    result = minimize(
        objective,
        initial_params,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    # Extract results
    lambda_0 = result.x[0]
    lambda_values = result.x[1:]
    lambda_dict = dict(zip(pre_cols, lambda_values))
    
    # Add zero weights for any pre-periods that were dropped
    for period in pre_periods:
        if period not in lambda_dict:
            lambda_dict[period] = 0.0
    
    return lambda_0, lambda_dict


def run_sdid(data: pd.DataFrame, unit_weights: Dict[str, float], time_weights: Dict[int, float], 
            omega_0: float, lambda_0: float, outcome_col: str, year_col: str, 
            state_col: str, treat_col: str, post_col: str) -> float:
    """
    Run the SDiD estimator using calculated weights.
    
    This implementation uses the weighted double-differencing approach which is
    mathematically equivalent to the weighted two-way fixed effects regression.
    
    Parameters:
    -----------
    data : DataFrame
        Panel data
    unit_weights, time_weights : dict
        Unit and time weights
    omega_0, lambda_0 : float
        Intercepts from weight calculations
    outcome_col, year_col, state_col, treat_col, post_col : str
        Column names
    
    Returns:
    --------
    float : SDiD treatment effect estimate
    """
    # Create pivot table for the outcome
    pivot_data = data.pivot_table(index=year_col, columns=state_col, values=outcome_col)
    
    # Get treated and control units
    treated_units = data[data[treat_col]][state_col].unique()
    control_units = [c for c in pivot_data.columns if c not in treated_units]
    
    # Get pre and post treatment periods
    pre_periods = sorted(data[~data[post_col]][year_col].unique())
    post_periods = sorted(data[data[post_col]][year_col].unique())
    
    # Ensure all required periods and units exist in the pivot table
    for period in pre_periods + post_periods:
        if period not in pivot_data.index:
            raise ValueError(f"Period {period} not found in pivot data")
    
    for unit in treated_units:
        if unit not in pivot_data.columns:
            raise ValueError(f"Treated unit {unit} not found in pivot data")
    
    # Calculate the four components of the double difference
    
    # 1. Y_tr,post: Average of treated units in post-treatment period
    y_tr_post = 0
    weight_sum = 0
    
    for unit in treated_units:
        # Extract post-treatment values for this unit
        unit_post_values = pivot_data.loc[post_periods, unit]
        
        # Skip if all values are NaN
        if unit_post_values.isna().all():
            continue
            
        # Calculate mean, ignoring NaNs
        unit_post_mean = unit_post_values.mean(skipna=True)
        
        # Equal weights for treated units (1/Ntr)
        unit_weight = 1.0 / len(treated_units)
        
        y_tr_post += unit_weight * unit_post_mean
        weight_sum += unit_weight
    
    # Normalize if any weights were applied
    if weight_sum > 0:
        y_tr_post /= weight_sum
    else:
        raise ValueError("No valid post-treatment data for treated units")
    
    # 2. Y_tr,pre: Weighted average of treated units in pre-treatment period
    y_tr_pre = 0
    weight_sum = 0
    
    for unit in treated_units:
        unit_weight = 1.0 / len(treated_units)
        period_weighted_sum = 0
        period_weight_sum = 0
        
        for period in pre_periods:
            # Skip periods not in time_weights or with NaN values
            if period not in time_weights or pd.isna(pivot_data.loc[period, unit]):
                continue
                
            period_weighted_sum += time_weights[period] * pivot_data.loc[period, unit]
            period_weight_sum += time_weights[period]
        
        # Only add this unit if it has valid data
        if period_weight_sum > 0:
            y_tr_pre += unit_weight * (period_weighted_sum / period_weight_sum)
            weight_sum += unit_weight
    
    # Normalize if any weights were applied
    if weight_sum > 0:
        y_tr_pre /= weight_sum
    else:
        raise ValueError("No valid pre-treatment data for treated units with the given time weights")
    
    # 3. Y_co,post: Weighted average of control units in post-treatment period
    y_co_post = 0
    weight_sum = 0
    
    for unit in control_units:
        # Skip units not in unit_weights
        if unit not in unit_weights:
            continue
            
        # Extract post-treatment values for this unit
        unit_post_values = pivot_data.loc[post_periods, unit]
        
        # Skip if all values are NaN
        if unit_post_values.isna().all():
            continue
            
        # Calculate mean, ignoring NaNs
        unit_post_mean = unit_post_values.mean(skipna=True)
        
        y_co_post += unit_weights[unit] * unit_post_mean
        weight_sum += unit_weights[unit]
    
    # Normalize if any weights were applied
    if weight_sum > 0:
        y_co_post /= weight_sum
    else:
        raise ValueError("No valid post-treatment data for control units with the given unit weights")
    
    # 4. Y_co,pre: Weighted average of control units in pre-treatment period
    y_co_pre = 0
    weight_sum = 0
    
    for unit in control_units:
        # Skip units not in unit_weights
        if unit not in unit_weights:
            continue
            
        period_weighted_sum = 0
        period_weight_sum = 0
        
        for period in pre_periods:
            # Skip periods not in time_weights or with NaN values
            if period not in time_weights or pd.isna(pivot_data.loc[period, unit]):
                continue
                
            period_weighted_sum += time_weights[period] * pivot_data.loc[period, unit]
            period_weight_sum += time_weights[period]
        
        # Only add this unit if it has valid data
        if period_weight_sum > 0:
            y_co_pre += unit_weights[unit] * (period_weighted_sum / period_weight_sum)
            weight_sum += unit_weights[unit]
    
    # Normalize if any weights were applied
    if weight_sum > 0:
        y_co_pre /= weight_sum
    else:
        raise ValueError("No valid pre-treatment data for control units with the given weights")
    
    # Add intercepts - this accounts for the fixed effects in the regression formulation
    y_tr_pre += lambda_0
    y_co_pre += lambda_0
    y_co_post += omega_0
    
    # Calculate SDiD estimator - the double difference
    tau_sdid = (y_tr_post - y_tr_pre) - (y_co_post - y_co_pre)
    
    return tau_sdid


def run_sdid_analysis(data_dict: Dict, country: str, donor_pool: List[str], 
                     outcome_col: str = 'gdp_growth_annual_share', 
                     country_col: str = 'country', year_col: str = 'year',
                     use_regression: bool = False) -> Optional[Dict]:
    """
    Run complete SDiD analysis for a single country.
    
    Parameters:
    -----------
    data_dict : dict
        Data dictionary from preprocessing
    country : str
        Name of treated country
    donor_pool : list
        List of donor countries
    outcome_col : str, optional
        Name of outcome column
    country_col : str, optional
        Name of country/unit column
    year_col : str, optional
        Name of year/time column
    use_regression : bool, optional
        Whether to also calculate SDID using the regression approach
    
    Returns:
    --------
    dict : Complete SDiD results including weights, estimates, etc.
    """
    
    # Get necessary data
    df = data_dict['df']
    panel = data_dict.get('panel')
    treatment_years = data_dict.get('treatment_years')
    
    # Create panel if not provided
    if panel is None and outcome_col in df.columns:
        panel = df.pivot_table(index=year_col, columns=country_col, values=outcome_col)
    
    # Check if country is in panel
    if country not in panel.columns:
        print(f"Country {country} not found in panel data")
        return None
    
    # Get treatment year
    treatment_year = treatment_years.get(country)
    if treatment_year is None:
        print(f"No treatment year found for {country}")
        return None
    
    # Prepare data for SDID
    sdid_data = df.copy()
    
    # Keep only relevant countries
    all_countries = [country] + donor_pool
    sdid_data = sdid_data[sdid_data[country_col].isin(all_countries)]
    
    # Filter out rows with NaN values in the outcome
    sdid_data = sdid_data.dropna(subset=[outcome_col])
    
    # Create treatment and post indicators
    sdid_data['treated'] = (sdid_data[country_col] == country)
    sdid_data['after_treatment'] = (sdid_data[year_col] >= treatment_year)
    
    # Sort data for clarity
    sdid_data = sdid_data.sort_values([country_col, year_col])
    
    # Check if we have enough data
    if sdid_data[sdid_data['treated']].shape[0] == 0:
        print(f"No data for treated country {country}")
        return None
    
    if sdid_data[~sdid_data['treated']].shape[0] == 0:
        print(f"No data for control countries")
        return None
    
    # Check for pre-treatment data
    treated_pre = sdid_data[(sdid_data['treated']) & (~sdid_data['after_treatment'])]
    if treated_pre.shape[0] == 0:
        print(f"No pre-treatment data for {country}")
        return None
    
    # Check for post-treatment data
    treated_post = sdid_data[(sdid_data['treated']) & (sdid_data['after_treatment'])]
    if treated_post.shape[0] == 0:
        print(f"No post-treatment data for {country}")
        return None
    
    # Calculate unit weights
    omega_0, unit_weights = calculate_unit_weights(
        sdid_data, outcome_col, year_col, country_col, 'treated', 'after_treatment'
    )
    
    # Calculate time weights
    lambda_0, time_weights = calculate_time_weights(
        sdid_data, outcome_col, year_col, country_col, 'treated', 'after_treatment'
    )
    
    # Run SDID estimator using double-differencing approach
    tau_sdid = run_sdid(
        sdid_data, unit_weights, time_weights, omega_0, lambda_0,
        outcome_col, year_col, country_col, 'treated', 'after_treatment'
    )
    
    # Generate synthetic values for plotting and analysis
    synth_values = {}
    actual_values = {}
    effect = {}
    
    # Create a pivot table for the original data
    pivot_data = sdid_data.pivot_table(
        index=year_col, columns=country_col, values=outcome_col
    )
    
    # Get all years in the data
    all_years = sorted(pivot_data.index)
    
    # Calculate weighted average of control units for each year
    for year in all_years:
        if year in pivot_data.index:
            # Get actual values for treated unit
            if country in pivot_data.columns and not pd.isna(pivot_data.loc[year, country]):
                actual_values[year] = pivot_data.loc[year, country]
            
            # Calculate synthetic value using unit weights
            weights_sum = 0
            value_sum = 0
            
            for control_country, weight in unit_weights.items():
                if control_country in pivot_data.columns and not pd.isna(pivot_data.loc[year, control_country]):
                    weights_sum += weight
                    value_sum += weight * pivot_data.loc[year, control_country]
            
            # Only store if we have sufficient data
            if weights_sum > 0:
                synth_values[year] = value_sum / weights_sum
                
                # Calculate effect if we have both actual and synthetic values
                if year in actual_values:
                    effect[year] = actual_values[year] - synth_values[year]
    
    # Calculate pre and post treatment RMSE
    pre_years = [y for y in all_years if y < treatment_year]
    post_years = [y for y in all_years if y >= treatment_year]
    
    pre_effects = [effect[y] for y in pre_years if y in effect]
    post_effects = [effect[y] for y in post_years if y in effect]
    
    pre_rmse = np.sqrt(np.mean([e**2 for e in pre_effects])) if pre_effects else np.nan
    post_rmse = np.sqrt(np.mean([e**2 for e in post_effects])) if post_effects else np.nan
    
    # Calculate RMSE ratio
    rmse_ratio = post_rmse / pre_rmse if pre_rmse and pre_rmse > 0 else np.nan
    
    # Calculate average treatment effect
    att = np.mean(post_effects) if post_effects else np.nan
    
    # Store results
    results = {
        'country': country,
        'treatment_year': treatment_year,
        'tau_sdid': tau_sdid,
        'omega_0': omega_0,
        'lambda_0': lambda_0,
        'unit_weights': unit_weights,
        'time_weights': time_weights,
        'actual_values': actual_values,
        'synthetic_values': synth_values,
        'effect': effect,
        'pre_rmse': pre_rmse,
        'post_rmse': post_rmse,
        'rmse_ratio': rmse_ratio,
        'att': att,
        'donor_pool': donor_pool
    }
    
    return results


def run_placebo_test_sdid(country: str, sdid_result: Dict, data_dict: Dict, 
                         n_placebos: int = 20) -> Dict:
    """
    Run placebo tests using donor countries as placebo treated units.
    
    Parameters:
    -----------
    country : str
        Name of the treated country
    sdid_result : dict
        SDID analysis results for the treated country
    data_dict : dict
        Data dictionary
    n_placebos : int
        Maximum number of placebo tests to run
        
    Returns:
    --------
    dict : Placebo test results
    """
    if sdid_result is None:
        return {'p_value_ratio': np.nan, 'placebos': []}
    
    panel = data_dict['panel']
    treated_countries = data_dict['treated_countries']
    treatment_years = data_dict['treatment_years'].copy()
    
    # Get weighted donors from original SDID
    weighted_donors = [donor for donor, weight in sdid_result['unit_weights'].items() 
                     if weight > 0.01]
    
    # Use weighted donors as placebo countries
    placebo_countries = [d for d in weighted_donors 
                       if d not in treated_countries][:n_placebos]
    
    # If we don't have enough, add from the original donor pool
    if len(placebo_countries) < min(n_placebos, len(sdid_result['donor_pool'])):
        additional = [d for d in sdid_result['donor_pool'] 
                    if d not in placebo_countries and d not in treated_countries]
        placebo_countries.extend(additional[:n_placebos - len(placebo_countries)])
    
    print(f"Using {len(placebo_countries)} placebo countries for {country}")
    
    # Store placebo results
    placebo_results = []
    
    # Run placebo tests
    for placebo_country in placebo_countries:
        # Create a donor pool for the placebo country (excluding the real treated country and the placebo itself)
        placebo_donors = [d for d in sdid_result['donor_pool'] 
                        if d != placebo_country and d != country]
        
        if len(placebo_donors) < 2:
            continue
        
        # Set the placebo country to be treated at the same time as the real treated country
        treatment_years[placebo_country] = sdid_result['treatment_year']
        
        # Create a copy of the data dictionary with the modified treatment years
        placebo_data_dict = data_dict.copy()
        placebo_data_dict['treatment_years'] = treatment_years
        
        try:
            # Run SDID for the placebo
            placebo_result = run_sdid_analysis(placebo_data_dict, placebo_country, placebo_donors)
            
            if placebo_result and not np.isnan(placebo_result['rmse_ratio']):
                placebo_results.append(placebo_result)
        except Exception:
            continue
    
    # Calculate p-values
    if placebo_results:
        # For RMSE ratio
        ratio = sdid_result['rmse_ratio']
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
        return {'p_value_ratio': np.nan, 'placebos': [], 'placebo_countries': []}


def run_sdid_pipeline(data_dict: Dict, donor_pools: Dict[str, List[str]], 
                     n_placebos: int = 20) -> CausalResults:
    """
    Run complete SDID pipeline for all treated countries.
    
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
    results = CausalResults('Synthetic Difference-in-Differences')
    
    treated_countries = data_dict['treated_countries']
    all_results = {}
    
    for country in treated_countries:
        print(f"Analyzing country: {country}")
        
        # Skip if no donor pool
        if country not in donor_pools or not donor_pools[country]:
            print(f"  No donor pool for {country}, skipping.")
            continue
        
        # Get donor pool
        donor_pool = donor_pools[country]
        print(f"  Using {len(donor_pool)} donors")
        
        try:
            # Run SDID analysis
            sdid_result = run_sdid_analysis(data_dict, country, donor_pool)
            
            if sdid_result is None:
                print(f"  Failed to run SDID for {country}, skipping.")
                continue
            
            # Run placebo test
            placebo_result = run_placebo_test_sdid(country, sdid_result, data_dict, n_placebos)
            
            # Store results
            all_results[country] = {
                'sdid_result': sdid_result,
                'placebo_result': placebo_result
            }
            
            # Add to results object
            country_metrics = {
                'att': sdid_result['att'],
                'pre_rmse': sdid_result['pre_rmse'],
                'post_rmse': sdid_result['post_rmse'],
                'rmse_ratio': sdid_result['rmse_ratio'],
                'p_value': placebo_result.get('p_value_ratio', np.nan),
                'significant': placebo_result.get('p_value_ratio', 1) <= 0.1 if placebo_result.get('p_value_ratio') is not None else False
            }
            results.add_country_result(country, country_metrics)
            
        except Exception as e:
            print(f"  Error analyzing {country}: {e}")
            continue
    
    # Create summary table
    summary_data = []
    for country, country_results in all_results.items():
        sdid_result = country_results['sdid_result']
        placebo_result = country_results['placebo_result']
        
        p_value = placebo_result.get('p_value_ratio', np.nan)
        
        summary_data.append({
            'country': country,
            'ATT': sdid_result['att'],
            'RMSE pre': sdid_result['pre_rmse'],
            'RMSE post': sdid_result['post_rmse'],
            'relation': sdid_result['rmse_ratio'],
            'p-val': p_value,
            'significant': 'Yes' if p_value is not None and p_value <= 0.1 else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('p-val')
    
    results.summary_table = summary_df
    results.raw_results = all_results
    
    return results


def create_weights_tables(results: CausalResults) -> Dict[str, pd.DataFrame]:
    """
    Create tables showing unit and time weights.
    
    Parameters:
    -----------
    results : CausalResults
        Results object from SDID analysis
        
    Returns:
    --------
    dict : Dictionary containing 'unit_weights' and 'time_weights' DataFrames
    """
    unit_weights_data = []
    time_weights_data = []
    
    for country, country_data in results.raw_results.items():
        sdid_result = country_data.get('sdid_result')
        
        if sdid_result:
            # Unit weights
            for donor, weight in sdid_result['unit_weights'].items():
                if weight > 0.01:  # Only include meaningful weights
                    unit_weights_data.append({
                        'treated_country': country,
                        'donor_country': donor,
                        'weight': weight
                    })
            
            # Time weights
            for year, weight in sdid_result['time_weights'].items():
                if weight > 0.01:  # Only include meaningful weights
                    time_weights_data.append({
                        'treated_country': country,
                        'year': year,
                        'weight': weight
                    })
    
    # Create DataFrames and sort
    unit_weights_df = pd.DataFrame(unit_weights_data)
    time_weights_df = pd.DataFrame(time_weights_data)
    
    if not unit_weights_df.empty:
        unit_weights_df = unit_weights_df.sort_values(['treated_country', 'weight'], ascending=[True, False])
    
    if not time_weights_df.empty:
        time_weights_df = time_weights_df.sort_values(['treated_country', 'year'])
    
    return {
        'unit_weights': unit_weights_df,
        'time_weights': time_weights_df
    }
