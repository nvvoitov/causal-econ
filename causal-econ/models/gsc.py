"""
Generalized Synthetic Control (Interactive Fixed Effects) implementation.
"""

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Tuple
import warnings

from ..core.base import CausalResults, calculate_treatment_effect, validate_panel_data


def estimate_ife_model(Y: np.ndarray, mask: np.ndarray, X: Optional[np.ndarray] = None, 
                      num_factors: int = 2, max_iter: int = 100, tol: float = 1e-8, 
                      min_iter: int = 10) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Estimate Interactive Fixed Effects model by iteratively updating factors and loadings.
    
    Parameters:
    -----------
    Y : np.ndarray
        Outcome data (T x Nco)
    mask : np.ndarray
        Boolean mask for non-missing values (True for observed)
    X : np.ndarray, optional
        Covariates (T x k)
    num_factors : int
        Number of factors to include
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    min_iter : int
        Minimum iterations before checking convergence
        
    Returns:
    --------
    tuple : (F, lambda_co, beta) - estimated factors, factor loadings, and coefficients
    """
    T, Nco = Y.shape
    
    # Initialize with PCA
    Y_copy = Y.copy()
    Y_copy[~mask] = 0  # Zero out missing values
    
    # SVD for initialization
    U, s, Vt = linalg.svd(Y_copy, full_matrices=False)
    F_old = np.zeros((T, num_factors))
    F_old[:, :min(num_factors, len(s))] = U[:, :min(num_factors, len(s))] * np.sqrt(T)
    lambda_co_old = np.zeros((Nco, num_factors))
    lambda_co_old[:, :min(num_factors, len(s))] = Vt.T[:, :min(num_factors, len(s))] * s[:min(num_factors, len(s))] / np.sqrt(T)
    
    beta_old = np.zeros(X.shape[1]) if X is not None else None
    
    # Iterative estimation
    for iter in range(max_iter):
        # 1. Update beta (if covariates are provided)
        if X is not None:
            # Residuals after removing factor component
            R = Y - F_old @ lambda_co_old.T
            
            # Weighted regression to handle missing values
            X_masked = X.copy()
            R_masked = R.copy()
            
            # Flatten and filter only observed values
            X_flat = X_masked.reshape(-1, X.shape[1])[mask.flatten()]
            R_flat = R_masked.flatten()[mask.flatten()]
            
            # Solve weighted least squares
            beta_new = linalg.lstsq(X_flat, R_flat)[0]
        else:
            beta_new = None
        
        # 2. Update factors and loadings
        # Residuals after removing covariate component
        R = Y - (X @ beta_new).reshape(-1, 1) if X is not None else Y
        
        # Set missing values to their predicted values
        R_filled = R.copy()
        R_filled[~mask] = (F_old @ lambda_co_old.T)[~mask]
        
        # Update via reduced-rank regression (SVD)
        U, s, Vt = linalg.svd(R_filled, full_matrices=False)
        
        # Normalize factors
        F_new = np.zeros((T, num_factors))
        F_new[:, :min(num_factors, len(s))] = U[:, :min(num_factors, len(s))] * np.sqrt(T)
        
        lambda_co_new = np.zeros((Nco, num_factors))
        lambda_co_new[:, :min(num_factors, len(s))] = Vt.T[:, :min(num_factors, len(s))] * s[:min(num_factors, len(s))] / np.sqrt(T)
        
        # Ensure factors are orthogonal
        Q, R_qr = linalg.qr(F_new, mode='economic')
        F_new = Q * np.sqrt(T)
        lambda_co_new = lambda_co_new @ R_qr.T / np.sqrt(T)
        
        # Check convergence
        F_diff = np.sum((F_new - F_old)**2) / np.sum(F_old**2) if np.sum(F_old**2) > 0 else np.inf
        lambda_diff = np.sum((lambda_co_new - lambda_co_old)**2) / np.sum(lambda_co_old**2) if np.sum(lambda_co_old**2) > 0 else np.inf
        
        if beta_new is not None and beta_old is not None:
            beta_diff = np.sum((beta_new - beta_old)**2) / np.sum(beta_old**2) if np.sum(beta_old**2) > 0 else np.inf
            converged = max(F_diff, lambda_diff, beta_diff) < tol
        else:
            converged = max(F_diff, lambda_diff) < tol
        
        # Update old values
        F_old = F_new
        lambda_co_old = lambda_co_new
        beta_old = beta_new
        
        if converged and iter >= min_iter-1:
            print(f"Converged after {iter+1} iterations")
            break
    
    return F_new, lambda_co_new, beta_new


def cross_validate_factors(Y: np.ndarray, mask: np.ndarray, X: Optional[np.ndarray] = None, 
                          max_factors: int = 10) -> int:
    """
    Cross-validate to select optimal number of factors.
    
    Parameters:
    -----------
    Y : np.ndarray
        Outcome data (T x Nco)
    mask : np.ndarray
        Boolean mask for non-missing values (True for observed)
    X : np.ndarray, optional
        Covariates (T x k)
    max_factors : int
        Maximum number of factors to try
        
    Returns:
    --------
    int : Optimal number of factors
    """
    # Create a validation set by randomly masking 20% of observed values
    val_mask = mask.copy()
    obs_indices = np.where(mask)
    num_obs = len(obs_indices[0])
    val_size = int(0.2 * num_obs)
    
    # Randomly select indices for validation
    val_indices = np.random.choice(num_obs, val_size, replace=False)
    train_mask = val_mask.copy()
    
    # Set validation indices to False in training mask
    train_mask[obs_indices[0][val_indices], obs_indices[1][val_indices]] = False
    
    # Evaluate models with different numbers of factors
    cv_errors = []
    
    for num_factors in range(1, max_factors + 1):
        # Estimate model on training set
        F, lambda_co, beta = estimate_ife_model(Y, train_mask, X, num_factors, max_iter=50)
        
        # Calculate predictions
        pred = F @ lambda_co.T
        if X is not None and beta is not None:
            pred += (X @ beta).reshape(-1, 1)
        
        # Evaluate on validation set
        val_indices_tuple = (obs_indices[0][val_indices], obs_indices[1][val_indices])
        val_pred = pred[val_indices_tuple]
        val_actual = Y[val_indices_tuple]
        
        # Mean squared error
        mse = np.mean((val_pred - val_actual)**2)
        cv_errors.append(mse)
    
    # Find optimal number of factors (1-based index)
    optimal_factors = np.argmin(cv_errors) + 1
    
    return optimal_factors


def calculate_factor_model(control_data: pd.DataFrame, covariates: Optional[pd.DataFrame] = None, 
                          num_factors: Optional[int] = None, max_factors: int = 10) -> Dict:
    """
    Estimate Interactive Fixed Effects (IFE) model using only control group data.
    
    Parameters:
    -----------
    control_data : pd.DataFrame
        Panel data for control units where each column is a unit and each row is a time period
    covariates : pd.DataFrame, optional
        Time-varying covariates with same row/column structure as control_data
    num_factors : int, optional
        Number of factors to include. If None, determined by cross-validation
    max_factors : int, optional
        Maximum number of factors to try in cross-validation
        
    Returns:
    --------
    dict : Estimated IFE model containing:
        - 'factors': Array of estimated factors
        - 'factor_loadings': Dictionary mapping unit names to factor loadings
        - 'coefficients': Estimated coefficients for covariates (if any)
        - 'eigenvalues': Array of eigenvalues (importance of each factor)
        - 'num_factors': Number of factors used
    """
    
    # Prepare data
    Y = control_data.values  # T x Nco
    T, Nco = Y.shape
    
    # Handle covariates if provided
    X = None
    if covariates is not None:
        if isinstance(covariates, pd.DataFrame):
            X = covariates.values  # Assuming same structure as panel
        else:
            raise ValueError("covariates must be a DataFrame")
    
    # Initial preprocessing - demean by unit and time
    Y_demean = Y.copy()
    unit_means = np.nanmean(Y, axis=0)
    time_means = np.nanmean(Y, axis=1)
    global_mean = np.nanmean(unit_means)
    
    for i in range(Nco):
        Y_demean[:, i] = Y[:, i] - unit_means[i]
    
    for t in range(T):
        Y_demean[t, :] = Y_demean[t, :] - (time_means[t] - global_mean)
    
    # Handle missing values
    mask = ~np.isnan(Y_demean)
    Y_filled = Y_demean.copy()
    Y_filled[~mask] = 0
    
    # If num_factors is not specified, use cross-validation
    if num_factors is None:
        num_factors = cross_validate_factors(Y_filled, mask, X, max_factors)
        print(f"Cross-validation selected {num_factors} factors")
    
    # Estimate the model
    beta = np.zeros(X.shape[1]) if X is not None else None
    F, lambda_co, beta = estimate_ife_model(Y_filled, mask, X, num_factors, max_iter=100)
    
    # Calculate eigenvalues/importance of factors
    Y_residuals = Y_filled - (X @ beta).reshape(-1, 1) if X is not None else Y_filled
    Y_residuals[~mask] = 0  # Zero out missing values
    
    # Compute eigenvalues using SVD
    U, s, Vt = linalg.svd(Y_residuals, full_matrices=False)
    eigenvalues = s[:num_factors]**2 / (T * Nco)  # Normalize
    
    # Create dictionary of factor loadings
    factor_loadings = {}
    for i, unit in enumerate(control_data.columns):
        factor_loadings[unit] = lambda_co[i, :].tolist()
    
    return {
        'factors': F,
        'factor_loadings': factor_loadings,
        'coefficients': beta,
        'eigenvalues': eigenvalues,
        'num_factors': num_factors
    }


def estimate_factor_loadings(treated_data: pd.DataFrame, factor_model: Dict, 
                            pre_treatment_mask: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
    """
    Estimate factor loadings for treated units by projecting onto estimated factors.
    
    Parameters:
    -----------
    treated_data : pd.DataFrame
        Data for treated units in pretreatment periods
    factor_model : dict
        Estimated factor model from calculate_factor_model()
    pre_treatment_mask : np.ndarray, optional
        Boolean mask for pretreatment periods (True for pretreatment)
        
    Returns:
    --------
    dict : Factor loadings for treated units
    """
    # Extract factors and coefficients from factor model
    factors = factor_model['factors']
    beta = factor_model.get('coefficients', None)
    covariates = factor_model.get('covariates', None)
    
    # Prepare data
    Y = treated_data.values  # T x Nt
    
    # Handle missing values
    if pre_treatment_mask is not None:
        mask = ~np.isnan(Y) & pre_treatment_mask
    else:
        mask = ~np.isnan(Y)
    
    # Initialize factor loadings dictionary
    factor_loadings = {}
    
    # Estimate factor loadings for each treated unit
    for i, unit in enumerate(treated_data.columns):
        # Get pretreatment data for this unit
        unit_mask = mask[:, i]
        y_pre = Y[unit_mask, i]
        
        # Skip if insufficient pretreatment data
        if len(y_pre) < factors.shape[1]:
            print(f"Warning: Insufficient pretreatment data for {unit}, using zeros for factor loadings")
            factor_loadings[unit] = np.zeros(factors.shape[1]).tolist()
            continue
        
        # Adjust for covariates if available
        if covariates is not None and beta is not None:
            X_pre = covariates[unit_mask]
            y_pre = y_pre - X_pre @ beta
        
        # Get pretreatment factors
        F_pre = factors[unit_mask]
        
        # Estimate factor loadings via OLS
        try:
            lambda_i = np.linalg.lstsq(F_pre, y_pre, rcond=None)[0]
            factor_loadings[unit] = lambda_i.tolist()
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix for {unit}, using zeros for factor loadings")
            factor_loadings[unit] = np.zeros(factors.shape[1]).tolist()
    
    return factor_loadings


def generate_counterfactuals(data_dict: Dict, factor_model: Dict, 
                           treated_factor_loadings: Optional[Dict] = None) -> Dict:
    """
    Generate counterfactuals for treated units based on estimated factors and loadings.
    
    Parameters:
    -----------
    data_dict : dict
        Panel data including treated units
    factor_model : dict
        Estimated factor model from calculate_factor_model()
    treated_factor_loadings : dict, optional
        Factor loadings for treated units. If None, will be estimated
        
    Returns:
    --------
    dict : Counterfactuals and treatment effects
        - 'counterfactuals': DataFrame of counterfactual outcomes
        - 'effects': DataFrame of treatment effects (actual - counterfactual)
        - 'att': Average treatment effect on treated for each time period
    """
    # Extract data
    panel = data_dict['panel']
    treated_units = data_dict.get('treated_countries', [])
    treatment_years = data_dict.get('treatment_years', {})
    
    # Extract factors and coefficients from factor model
    factors = factor_model['factors']
    beta = factor_model.get('coefficients', None)
    covariates = factor_model.get('covariates', None)
    
    # If factor loadings not provided, estimate them
    if treated_factor_loadings is None:
        # Create pretreatment mask
        pre_treatment_mask = np.ones((len(panel.index), len(treated_units)), dtype=bool)
        for i, unit in enumerate(treated_units):
            if unit in treatment_years:
                treatment_year = treatment_years[unit]
                for t, year in enumerate(panel.index):
                    if year >= treatment_year:
                        pre_treatment_mask[t, i] = False
        
        # Estimate factor loadings
        treated_data = panel[treated_units]
        treated_factor_loadings = estimate_factor_loadings(
            treated_data, factor_model, pre_treatment_mask
        )
    
    # Generate counterfactuals
    counterfactuals = pd.DataFrame(index=panel.index, columns=treated_units)
    
    for unit in treated_units:
        # Get factor loadings for this unit
        if unit not in treated_factor_loadings:
            print(f"Warning: No factor loadings for {unit}, skipping")
            continue
        
        lambda_i = np.array(treated_factor_loadings[unit])
        
        # Calculate counterfactual for all periods
        counterfactual = factors @ lambda_i
        
        # Add covariate effect if available
        if covariates is not None and beta is not None:
            X_unit = covariates.get(unit, None)
            if X_unit is not None:
                counterfactual += X_unit @ beta
        
        # Store in DataFrame
        counterfactuals[unit] = counterfactual
    
    # Calculate treatment effects (actual - counterfactual)
    effects = panel[treated_units] - counterfactuals
    
    # Calculate average treatment effect on treated (ATT)
    att = pd.Series(index=panel.index)
    
    for t, time in enumerate(panel.index):
        # Count treated units at this time
        treated_at_t = []
        for unit in treated_units:
            if unit in treatment_years and time >= treatment_years[unit]:
                treated_at_t.append(unit)
        
        if treated_at_t:
            # Calculate ATT for this time period
            att[time] = effects.loc[time, treated_at_t].mean()
        else:
            att[time] = np.nan
    
    return {
        'counterfactuals': counterfactuals,
        'effects': effects,
        'att': att
    }


def run_placebo_test_gsc(country: str, gsc_result: Dict, data_dict: Dict, 
                        factor_model: Dict, n_placebos: int = 20) -> Dict:
    """
    Run placebo tests for Generalized Synthetic Control by assigning placebo treatments to control units.
    
    Parameters:
    -----------
    country : str
        Name of the treated country
    gsc_result : dict
        GSC analysis results for the treated country
    data_dict : dict
        Data dictionary
    factor_model : dict
        Estimated factor model
    n_placebos : int
        Number of placebo tests to run
        
    Returns:
    --------
    dict : Placebo test results
    """
    if gsc_result is None:
        return {'p_value_ratio': np.nan, 'p_value_att': np.nan, 'placebos': []}
    
    panel = data_dict['panel']
    treatment_year = gsc_result['treatment_year']
    donor_pool = gsc_result.get('donor_pool', [])
    
    # Just use the donor pool as placebo countries, up to n_placebos
    placebo_countries = donor_pool[:n_placebos]
    
    print(f"Using {len(placebo_countries)} placebo countries for {country}")
    
    # Store placebo results
    placebo_results = []
    
    # Run placebo tests
    for placebo_country in placebo_countries:
        try:
            # Create pretreatment mask
            pre_treatment_mask = np.ones((len(panel.index), 1), dtype=bool)
            for t, year in enumerate(panel.index):
                if year >= treatment_year:
                    pre_treatment_mask[t, 0] = False
            
            # Set placebo as treated
            placebo_data = {
                'panel': panel[[placebo_country]],
                'treated_countries': [placebo_country],
                'treatment_years': {placebo_country: treatment_year}
            }
            
            # Estimate factor loadings
            placebo_factor_loadings = estimate_factor_loadings(
                placebo_data['panel'],
                factor_model,
                pre_treatment_mask
            )
            
            # Generate counterfactuals
            counterfactual_results = generate_counterfactuals(
                placebo_data,
                factor_model,
                placebo_factor_loadings
            )
            
            # Extract results and calculate metrics
            counterfactuals = counterfactual_results['counterfactuals']
            effects = counterfactual_results['effects']
            years = panel.index.tolist()
            
            # Convert to dictionaries for easier access
            actual_values = {year: panel.loc[year, placebo_country] for year in years}
            counterfactual_values = {year: counterfactuals.loc[year, placebo_country] 
                                 if year in counterfactuals.index and placebo_country in counterfactuals.columns 
                                 and not pd.isna(counterfactuals.loc[year, placebo_country]) else None 
                                 for year in years}
            effect_values = {year: effects.loc[year, placebo_country] 
                          if year in effects.index and placebo_country in effects.columns 
                          and not pd.isna(effects.loc[year, placebo_country]) else None 
                          for year in years}
            
            # Calculate metrics
            pre_years = [y for y in years if y < treatment_year]
            post_years = [y for y in years if y >= treatment_year]
            
            pre_effects = [effect_values[y] for y in pre_years if effect_values.get(y) is not None]
            post_effects = [effect_values[y] for y in post_years if effect_values.get(y) is not None]
            
            pre_rmse = np.sqrt(np.mean([e**2 for e in pre_effects])) if pre_effects else np.nan
            post_rmse = np.sqrt(np.mean([e**2 for e in post_effects])) if post_effects else np.nan
            rmse_ratio = post_rmse / pre_rmse if pre_rmse and pre_rmse > 0 else np.nan
            att = np.mean(post_effects) if post_effects else np.nan
            
            # Create placebo result
            placebo_result = {
                'country': placebo_country,
                'treatment_year': treatment_year,
                'actual_values': actual_values,
                'counterfactual_values': counterfactual_values,
                'effect': effect_values,
                'pre_rmse': pre_rmse,
                'post_rmse': post_rmse,
                'rmse_ratio': rmse_ratio,
                'att': att
            }
            
            if not np.isnan(rmse_ratio):
                placebo_results.append(placebo_result)
                
        except Exception as e:
            print(f"Error in placebo test for {placebo_country}: {e}")
            continue
    
    # Calculate p-values
    if placebo_results:
        # For RMSE ratio
        ratio = gsc_result['rmse_ratio']
        if np.isnan(ratio):
            p_value_ratio = np.nan
        else:
            p_value_ratio = sum(1 for p in placebo_results 
                              if p['rmse_ratio'] >= ratio) / len(placebo_results)
        
        # For ATT (absolute value)
        att = abs(gsc_result['att'])
        if np.isnan(att):
            p_value_att = np.nan
        else:
            p_value_att = sum(1 for p in placebo_results 
                            if abs(p['att']) >= att) / len(placebo_results)
        
        return {
            'p_value_ratio': p_value_ratio,
            'p_value_att': p_value_att,
            'placebos': placebo_results,
            'placebo_countries': [p['country'] for p in placebo_results]
        }
    else:
        return {'p_value_ratio': np.nan, 'p_value_att': np.nan, 'placebos': [], 'placebo_countries': []}


def run_gsc_analysis(data_dict: Dict, donor_pools: Dict[str, List[str]], 
                    num_factors: Optional[int] = None, covariates: Optional[List[str]] = None,
                    n_placebos: int = 20) -> CausalResults:
    """
    Run Generalized Synthetic Control analysis for all treated countries.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    donor_pools : dict
        Dictionary mapping treated countries to donor pools
    num_factors : int, optional
        Number of factors to use (if None, determined by cross-validation)
    covariates : list, optional
        List of covariate column names
    n_placebos : int
        Number of placebo tests to run
        
    Returns:
    --------
    CausalResults : Standardized results object
    """
    results = CausalResults('Generalized Synthetic Control')
    
    # Extract necessary data
    df = data_dict['df']
    panel = data_dict['panel']
    treatment_years = data_dict['treatment_years']
    treated_countries = data_dict['treated_countries']
    
    # Get all control countries
    all_control_countries = [c for c in panel.columns if c not in treated_countries]
    
    # Prepare covariate data if specified
    covariate_data = None
    if covariates is not None:
        covariate_panels = {}
        for cov in covariates:
            cov_panel = df.pivot_table(index='year', columns='country', values=cov)
            covariate_panels[cov] = cov_panel
        covariate_data = covariate_panels
    
    # Step 1: Estimate factor model using all control countries
    print(f"Estimating factor model using {len(all_control_countries)} control countries...")
    control_panel = panel[all_control_countries]
    control_covariates = None
    if covariate_data is not None:
        control_covariates = pd.DataFrame(index=panel.index)
        for cov in covariates:
            control_covariates[cov] = covariate_data[cov][all_control_countries].mean(axis=1)
    
    factor_model = calculate_factor_model(
        control_panel, 
        covariates=control_covariates,
        num_factors=num_factors
    )
    print(f"Factor model estimated with {factor_model['num_factors']} factors")
    
    # Store all country results
    all_country_results = {}
    
    # Process each treated country
    for country in treated_countries:
        print(f"\nAnalyzing country: {country}")
        
        # Check if country is in panel
        if country not in panel.columns:
            print(f"  Country {country} not found in panel data, skipping.")
            continue
        
        # Check if treatment year exists
        if country not in treatment_years:
            print(f"  No treatment year found for {country}, skipping.")
            continue
        
        treatment_year = treatment_years[country]
        print(f"  Treatment year: {treatment_year}")
        
        # Get donor pool
        donor_pool = donor_pools.get(country, [])
        print(f"  Using {len(donor_pool)} donors")
        
        try:
            # Step 2: Estimate factor loadings for treated country
            treated_data = panel[[country]]
            pre_treatment_mask = np.ones((len(panel.index), 1), dtype=bool)
            for t, year in enumerate(panel.index):
                if year >= treatment_year:
                    pre_treatment_mask[t, 0] = False
            
            print(f"  Estimating factor loadings for {country}...")
            treated_factor_loadings = estimate_factor_loadings(
                treated_data, factor_model, pre_treatment_mask
            )
            
            # Step 3: Generate counterfactuals
            print(f"  Generating counterfactuals for {country}...")
            counterfactual_results = generate_counterfactuals(
                {'panel': panel[[country]], 'treated_countries': [country], 'treatment_years': {country: treatment_year}},
                factor_model,
                treated_factor_loadings
            )
            
            # Extract results
            counterfactuals = counterfactual_results['counterfactuals']
            effects = counterfactual_results['effects']
            
            # Prepare data for result dictionary
            years = panel.index.tolist()
            actual_values = {year: panel.loc[year, country] for year in years}
            counterfactual_values = {year: counterfactuals.loc[year, country] if not np.isnan(counterfactuals.loc[year, country]) else None for year in years}
            effect_values = {year: effects.loc[year, country] if not np.isnan(effects.loc[year, country]) else None for year in years}
            
            # Calculate metrics
            pre_years = [y for y in years if y < treatment_year]
            post_years = [y for y in years if y >= treatment_year]
            
            pre_effects = [effect_values[y] for y in pre_years if effect_values[y] is not None]
            post_effects = [effect_values[y] for y in post_years if effect_values[y] is not None]
            
            pre_rmse = np.sqrt(np.mean([e**2 for e in pre_effects])) if pre_effects else np.nan
            post_rmse = np.sqrt(np.mean([e**2 for e in post_effects])) if post_effects else np.nan
            rmse_ratio = post_rmse / pre_rmse if pre_rmse and pre_rmse > 0 else np.nan
            att = np.mean(post_effects) if post_effects else np.nan
            
            # Create result dictionary
            country_result = {
                'country': country,
                'treatment_year': treatment_year,
                'actual_values': actual_values,
                'counterfactual_values': counterfactual_values,
                'effect': effect_values,
                'pre_rmse': pre_rmse,
                'post_rmse': post_rmse,
                'rmse_ratio': rmse_ratio,
                'att': att,
                'donor_pool': donor_pool,
                'factor_loadings': treated_factor_loadings[country]
            }
            
            # Run placebo test
            print(f"  Running placebo tests for {country}...")
            placebo_result = run_placebo_test_gsc(
                country, country_result, data_dict, factor_model, n_placebos=n_placebos
            )
            
            # Add p-values to country result
            if placebo_result:
                country_result['p_value_ratio'] = placebo_result.get('p_value_ratio', np.nan)
                country_result['p_value_att'] = placebo_result.get('p_value_att', np.nan)
            
            # Store raw results
            all_country_results[country] = {
                'gsc_result': country_result,
                'placebo_result': placebo_result
            }
            
            # Add to results object
            country_metrics = {
                'att': country_result['att'],
                'pre_rmse': country_result['pre_rmse'],
                'post_rmse': country_result['post_rmse'],
                'rmse_ratio': country_result['rmse_ratio'],
                'p_value': country_result.get('p_value_ratio', np.nan),
                'significant': country_result.get('p_value_ratio', 1) <= 0.1 if country_result.get('p_value_ratio') is not None else False
            }
            results.add_country_result(country, country_metrics)
            
        except Exception as e:
            print(f"  Error analyzing {country}: {e}")
            continue
    
    # Create summary table
    summary_data = []
    for country, country_results in all_country_results.items():
        gsc_result = country_results['gsc_result']
        placebo_result = country_results['placebo_result']
        
        p_value = placebo_result.get('p_value_ratio', np.nan)
        
        summary_data.append({
            'country': country,
            'treatment_year': gsc_result['treatment_year'],
            'ATT': gsc_result['att'],
            'RMSE pre': gsc_result['pre_rmse'],
            'RMSE post': gsc_result['post_rmse'],
            'RMSE ratio': gsc_result['rmse_ratio'],
            'p-value': p_value,
            'significant': 'Yes' if p_value is not None and p_value <= 0.05 else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('p-value', na_position='last')
    
    results.summary_table = summary_df
    results.raw_results = all_country_results
    
    # Store factor model in results for reference
    results.factor_model = factor_model
    
    return results
