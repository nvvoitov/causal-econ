"""
Impact analysis and table generation for causal inference results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import warnings

from ..core.base import CausalResults
from ..visualization.distributions import prepare_analysis_data


def create_impact_table(results: CausalResults, 
                       method_name: str,
                       data_dict: Optional[Dict] = None) -> pd.DataFrame:
    """
    Create impact table showing effects grouped by positive/negative ATT.
    
    Parameters:
    -----------
    results : CausalResults
        Results from causal analysis
    method_name : str
        Name of the method
    data_dict : dict, optional
        Original data dictionary
        
    Returns:
    --------
    pd.DataFrame : Impact table
    """
    
    # Prepare analysis data
    analysis_data = prepare_analysis_data(results, method_name, data_dict)
    growth_df = analysis_data['growth_df']
    rmse_df = analysis_data.get('rmse_df', pd.DataFrame())
    
    if growth_df.empty:
        return pd.DataFrame()
    
    # Get summary table for ATT values
    summary_table = results.summary_table
    if summary_table is None or summary_table.empty:
        return pd.DataFrame()
    
    # Standardize ATT column name
    att_col = None
    for col in ['ATT', 'ATE']:
        if col in summary_table.columns:
            att_col = col
            break
    
    if att_col is None:
        return pd.DataFrame()
    
    # Split countries by effect direction
    positive_att_countries = summary_table[summary_table[att_col] > 0]['country'].tolist()
    negative_att_countries = summary_table[summary_table[att_col] <= 0]['country'].tolist()
    
    # Filter dataframes
    positive_growth_df = growth_df[growth_df['country'].isin(positive_att_countries)]
    negative_growth_df = growth_df[growth_df['country'].isin(negative_att_countries)]
    
    # Calculate significance percentages
    if not positive_growth_df.empty:
        unique_pos = positive_growth_df.drop_duplicates('country')
        pos_sig_pct = (unique_pos['significant'].sum() / len(unique_pos) * 100) if len(unique_pos) > 0 else 0
    else:
        pos_sig_pct = 0
    
    if not negative_growth_df.empty:
        unique_neg = negative_growth_df.drop_duplicates('country')
        neg_sig_pct = (unique_neg['significant'].sum() / len(unique_neg) * 100) if len(unique_neg) > 0 else 0
    else:
        neg_sig_pct = 0
    
    rows = []
    
    # Process positive effect countries
    if positive_att_countries:
        pos_metrics = _calculate_group_metrics(
            positive_growth_df, rmse_df, positive_att_countries, pos_sig_pct
        )
        pos_metrics['Group'] = f"Positive Effect Countries (n={len(positive_att_countries)})"
        rows.append(pos_metrics)
    
    # Process negative effect countries
    if negative_att_countries:
        neg_metrics = _calculate_group_metrics(
            negative_growth_df, rmse_df, negative_att_countries, neg_sig_pct
        )
        neg_metrics['Group'] = f"Negative Effect Countries (n={len(negative_att_countries)})"
        rows.append(neg_metrics)
    
    return pd.DataFrame(rows)


def _calculate_group_metrics(growth_df: pd.DataFrame, 
                           rmse_df: pd.DataFrame,
                           countries: List[str],
                           significance_pct: float) -> Dict[str, Any]:
    """Calculate metrics for a group of countries."""
    
    # GDP growth metrics
    pre_growth = growth_df[growth_df['period'] == 'Pre-Intervention']['gdp_growth'].mean()
    post_actual = growth_df[growth_df['period'] == 'Post-Intervention Actual']['gdp_growth'].mean()
    post_counter = growth_df[growth_df['period'] == 'Post-Intervention Counterfactual']['gdp_growth'].mean()
    
    # Calculate effects
    absolute_effect = post_actual - post_counter if not (np.isnan(post_actual) or np.isnan(post_counter)) else np.nan
    relative_effect = (absolute_effect / abs(post_counter)) * 100 if post_counter != 0 else np.nan
    
    # RMSE metrics
    pre_rmse = np.nan
    post_rmse = np.nan
    rmse_ratio = np.nan
    
    if not rmse_df.empty:
        country_rmse = rmse_df[rmse_df['country'].isin(countries)]
        
        pre_rmse_vals = country_rmse[country_rmse['period'] == 'Pre-Intervention RMSE']['rmse']
        post_rmse_vals = country_rmse[country_rmse['period'] == 'Post-Intervention RMSE']['rmse']
        rmse_ratio_vals = country_rmse[country_rmse['period'] == 'RMSE Ratio (Post/Pre)']['rmse']
        
        pre_rmse = pre_rmse_vals.mean() if not pre_rmse_vals.empty else np.nan
        post_rmse = post_rmse_vals.mean() if not post_rmse_vals.empty else np.nan
        rmse_ratio = rmse_ratio_vals.mean() if not rmse_ratio_vals.empty else np.nan
    
    return {
        'Pre-Intervention Growth': round(pre_growth, 2) if not np.isnan(pre_growth) else np.nan,
        'Post-Intervention Actual': round(post_actual, 2) if not np.isnan(post_actual) else np.nan,
        'Post-Intervention Counterfactual': round(post_counter, 2) if not np.isnan(post_counter) else np.nan,
        'Absolute Effect': round(absolute_effect, 2) if not np.isnan(absolute_effect) else np.nan,
        'Relative Effect (%)': round(relative_effect, 1) if not np.isnan(relative_effect) else np.nan,
        'Significant (%)': round(significance_pct, 1),
        'Pre-Intervention RMSE': round(pre_rmse, 2) if not np.isnan(pre_rmse) else np.nan,
        'Post-Intervention RMSE': round(post_rmse, 2) if not np.isnan(post_rmse) else np.nan,
        'RMSE Ratio': round(rmse_ratio, 2) if not np.isnan(rmse_ratio) else np.nan
    }


def create_country_impact_details(results: CausalResults,
                                method_name: str) -> pd.DataFrame:
    """
    Create detailed impact table for individual countries.
    
    Parameters:
    -----------
    results : CausalResults
        Results from causal analysis
    method_name : str
        Name of the method
        
    Returns:
    --------
    pd.DataFrame : Detailed country impact table
    """
    
    country_details = []
    
    for country, country_data in results.raw_results.items():
        # Extract method-specific result
        result_key = f"{method_name.lower().replace(' ', '_').replace('-', '_')}_result"
        country_result = None
        
        if isinstance(country_data, dict):
            for key in [result_key, 'sc_result', 'did_result', 'sdid_result', 'gsc_result', 'ce2v_result']:
                if key in country_data:
                    country_result = country_data[key]
                    break
        else:
            country_result = country_data
            
        if country_result is None:
            continue
        
        # Extract placebo results
        placebo_result = None
        if isinstance(country_data, dict):
            placebo_result = country_data.get('placebo_result')
        
        # Calculate detailed metrics
        treatment_year = country_result.get('treatment_year')
        att = country_result.get('att', np.nan)
        pre_rmse = country_result.get('pre_rmse', np.nan)
        post_rmse = country_result.get('post_rmse', np.nan)
        rmse_ratio = country_result.get('rmse_ratio', np.nan)
        
        # Get p-value
        p_value = np.nan
        if placebo_result:
            p_value = placebo_result.get('p_value_ratio', 
                      placebo_result.get('p_value_att', np.nan))
        
        # Calculate pre/post growth if time series data available
        pre_growth = np.nan
        post_growth = np.nan
        post_counterfactual = np.nan
        
        actual_key = next((k for k in ['actual_values', 'treated_values'] if k in country_result), None)
        counterfactual_key = next((k for k in ['counterfactual_values', 'synthetic_values', 'synthetic', 'shifted_control'] if k in country_result), None)
        
        if actual_key and counterfactual_key and treatment_year:
            actual_values = country_result[actual_key]
            counterfactual_values = country_result[counterfactual_key]
            
            if isinstance(actual_values, dict):
                years = sorted(actual_values.keys())
                pre_years = [y for y in years if y < treatment_year]
                post_years = [y for y in years if y >= treatment_year]
                
                if pre_years:
                    pre_vals = [actual_values[y] for y in pre_years if not np.isnan(actual_values.get(y, np.nan))]
                    pre_growth = np.mean(pre_vals) if pre_vals else np.nan
                
                if post_years:
                    post_vals = [actual_values[y] for y in post_years if not np.isnan(actual_values.get(y, np.nan))]
                    post_growth = np.mean(post_vals) if post_vals else np.nan
                    
                    counter_vals = [counterfactual_values[y] for y in post_years if not np.isnan(counterfactual_values.get(y, np.nan))]
                    post_counterfactual = np.mean(counter_vals) if counter_vals else np.nan
        
        country_details.append({
            'country': country,
            'method': method_name,
            'treatment_year': treatment_year,
            'pre_intervention_growth': pre_growth,
            'post_intervention_actual': post_growth,
            'post_intervention_counterfactual': post_counterfactual,
            'ATT': att,
            'pre_RMSE': pre_rmse,
            'post_RMSE': post_rmse,
            'RMSE_ratio': rmse_ratio,
            'p_value': p_value,
            'significant': 'Yes' if not np.isnan(p_value) and p_value <= 0.1 else 'No'
        })
    
    return pd.DataFrame(country_details)


def create_cross_method_impact_comparison(results_dict: Dict[str, CausalResults]) -> pd.DataFrame:
    """
    Create impact comparison across methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
        
    Returns:
    --------
    pd.DataFrame : Cross-method impact comparison
    """
    
    method_impacts = []
    
    for method_name, results in results_dict.items():
        try:
            impact_table = create_impact_table(results, method_name)
            if not impact_table.empty:
                # Add method identifier
                impact_table['method'] = method_name
                method_impacts.append(impact_table)
        except Exception as e:
            warnings.warn(f"Failed to create impact table for {method_name}: {e}")
    
    if not method_impacts:
        return pd.DataFrame()
    
    # Combine all impact tables
    combined_impacts = pd.concat(method_impacts, ignore_index=True)
    
    # Reorganize for comparison
    comparison_data = []
    
    # Get all effect directions (positive/negative) and methods
    effect_types = set()
    methods = set()
    
    for _, row in combined_impacts.iterrows():
        group = row['Group']
        method = row['method']
        effect_type = 'Positive' if 'Positive' in group else 'Negative'
        
        effect_types.add(effect_type)
        methods.add(method)
        
        comparison_data.append({
            'effect_type': effect_type,
            'method': method,
            'n_countries': int(group.split('n=')[1].split(')')[0]),
            'absolute_effect': row['Absolute Effect'],
            'relative_effect': row['Relative Effect (%)'],
            'significant_pct': row['Significant (%)'],
            'rmse_ratio': row['RMSE Ratio']
        })
    
    return pd.DataFrame(comparison_data)


def export_impact_tables(results_dict: Dict[str, CausalResults],
                        output_dir: str):
    """
    Export all impact tables to CSV files.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
    output_dir : str
        Output directory
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual method impact tables
    for method_name, results in results_dict.items():
        try:
            impact_table = create_impact_table(results, method_name)
            if not impact_table.empty:
                filename = f"{method_name.lower().replace(' ', '_')}_impact.csv"
                impact_table.to_csv(os.path.join(output_dir, filename), index=False)
            
            # Detailed country impacts
            country_details = create_country_impact_details(results, method_name)
            if not country_details.empty:
                filename = f"{method_name.lower().replace(' ', '_')}_country_details.csv"
                country_details.to_csv(os.path.join(output_dir, filename), index=False)
                
        except Exception as e:
            warnings.warn(f"Failed to export impact tables for {method_name}: {e}")
    
    # Cross-method comparison
    try:
        cross_method_impact = create_cross_method_impact_comparison(results_dict)
        if not cross_method_impact.empty:
            cross_method_impact.to_csv(os.path.join(output_dir, "cross_method_impact_comparison.csv"), index=False)
    except Exception as e:
        warnings.warn(f"Failed to export cross-method impact comparison: {e}")
    
    print(f"Impact tables exported to {output_dir}")


def calculate_aggregate_impact_metrics(results_dict: Dict[str, CausalResults]) -> pd.DataFrame:
    """
    Calculate aggregate impact metrics across all methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
        
    Returns:
    --------
    pd.DataFrame : Aggregate impact metrics
    """
    
    aggregate_data = []
    
    for method_name, results in results_dict.items():
        try:
            impact_table = create_impact_table(results, method_name)
            if impact_table.empty:
                continue
            
            # Calculate weighted averages
            total_countries = 0
            total_positive = 0
            total_negative = 0
            
            weighted_abs_effect = 0
            weighted_rel_effect = 0
            weighted_significance = 0
            
            for _, row in impact_table.iterrows():
                group = row['Group']
                n_countries = int(group.split('n=')[1].split(')')[0])
                
                total_countries += n_countries
                
                if 'Positive' in group:
                    total_positive += n_countries
                else:
                    total_negative += n_countries
                
                # Weight by number of countries
                weight = n_countries / total_countries if total_countries > 0 else 0
                
                abs_effect = row['Absolute Effect']
                if not np.isnan(abs_effect):
                    weighted_abs_effect += abs_effect * weight
                
                rel_effect = row['Relative Effect (%)']
                if not np.isnan(rel_effect):
                    weighted_rel_effect += rel_effect * weight
                
                sig_pct = row['Significant (%)']
                if not np.isnan(sig_pct):
                    weighted_significance += sig_pct * weight
            
            aggregate_data.append({
                'method': method_name,
                'total_countries': total_countries,
                'positive_effect_countries': total_positive,
                'negative_effect_countries': total_negative,
                'positive_effect_rate': (total_positive / total_countries * 100) if total_countries > 0 else 0,
                'weighted_absolute_effect': weighted_abs_effect,
                'weighted_relative_effect': weighted_rel_effect,
                'weighted_significance_rate': weighted_significance
            })
            
        except Exception as e:
            warnings.warn(f"Failed to calculate aggregate metrics for {method_name}: {e}")
    
    return pd.DataFrame(aggregate_data)
