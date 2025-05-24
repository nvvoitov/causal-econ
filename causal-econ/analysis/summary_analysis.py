"""
Standardized summary table generation for causal inference results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import warnings

from ..core.base import CausalResults


def create_standardized_summary_table(results: CausalResults, 
                                     method_name: str) -> pd.DataFrame:
    """
    Create standardized summary table from any CausalResults object.
    
    Parameters:
    -----------
    results : CausalResults
        Results from any causal method
    method_name : str
        Name of the method
        
    Returns:
    --------
    pd.DataFrame : Standardized summary table
    """
    
    if results.summary_table is not None and not results.summary_table.empty:
        # Use existing summary table if available
        summary_df = results.summary_table.copy()
        
        # Standardize column names
        column_mapping = {
            'ATT': 'ATT',
            'ATE': 'ATT',  # Standardize to ATT
            'RMSE pre': 'RMSE_pre',
            'RMSE post': 'RMSE_post', 
            'RMSE ratio': 'RMSE_ratio',
            'relation': 'RMSE_ratio',  # Standardize to RMSE_ratio
            'p-val': 'p_value',
            'p-value': 'p_value',
            'significant': 'significant'
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in summary_df.columns:
                summary_df = summary_df.rename(columns={old_name: new_name})
        
    else:
        # Generate summary from raw results
        summary_data = []
        
        for country, country_data in results.raw_results.items():
            # Extract method-specific result
            result_key = f"{method_name.lower().replace(' ', '_').replace('-', '_')}_result"
            country_result = None
            
            if isinstance(country_data, dict):
                # Try different possible keys
                for key in [result_key, 'sc_result', 'did_result', 'sdid_result', 'gsc_result', 'ce2v_result']:
                    if key in country_data:
                        country_result = country_data[key]
                        break
            else:
                country_result = country_data
                
            if country_result is None:
                continue
            
            # Extract placebo results for p-values
            placebo_result = None
            if isinstance(country_data, dict):
                placebo_result = country_data.get('placebo_result')
            
            # Extract metrics
            att = country_result.get('att', np.nan)
            pre_rmse = country_result.get('pre_rmse', np.nan)
            post_rmse = country_result.get('post_rmse', np.nan)
            rmse_ratio = country_result.get('rmse_ratio', np.nan)
            treatment_year = country_result.get('treatment_year')
            
            # Extract p-value
            p_value = np.nan
            if placebo_result:
                p_value = placebo_result.get('p_value_ratio', 
                          placebo_result.get('p_value_att', np.nan))
            
            # Determine significance
            significant = 'No'
            if not np.isnan(p_value) and p_value <= 0.1:
                significant = 'Yes'
            
            summary_data.append({
                'country': country,
                'method': method_name,
                'treatment_year': treatment_year,
                'ATT': att,
                'RMSE_pre': pre_rmse,
                'RMSE_post': post_rmse,
                'RMSE_ratio': rmse_ratio,
                'p_value': p_value,
                'significant': significant
            })
        
        summary_df = pd.DataFrame(summary_data)
    
    # Ensure required columns exist
    required_columns = ['country', 'ATT', 'RMSE_pre', 'RMSE_post', 'RMSE_ratio', 'p_value', 'significant']
    for col in required_columns:
        if col not in summary_df.columns:
            summary_df[col] = np.nan if col != 'significant' else 'No'
    
    # Add method column if not present
    if 'method' not in summary_df.columns:
        summary_df['method'] = method_name
    
    # Sort by p-value
    if 'p_value' in summary_df.columns:
        summary_df = summary_df.sort_values('p_value', na_position='last')
    
    return summary_df


def combine_method_summaries(results_dict: Dict[str, CausalResults]) -> pd.DataFrame:
    """
    Combine summary tables from multiple methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
        
    Returns:
    --------
    pd.DataFrame : Combined summary table
    """
    
    all_summaries = []
    
    for method_name, results in results_dict.items():
        try:
            summary_df = create_standardized_summary_table(results, method_name)
            all_summaries.append(summary_df)
        except Exception as e:
            warnings.warn(f"Failed to create summary for {method_name}: {e}")
    
    if not all_summaries:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_summaries, ignore_index=True)
    
    # Sort by country and method
    combined_df = combined_df.sort_values(['country', 'method'])
    
    return combined_df


def create_method_comparison_table(results_dict: Dict[str, CausalResults],
                                  metric: str = 'ATT') -> pd.DataFrame:
    """
    Create comparison table across methods for a specific metric.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
    metric : str
        Metric to compare ('ATT', 'RMSE_ratio', 'p_value')
        
    Returns:
    --------
    pd.DataFrame : Comparison table with methods as columns
    """
    
    combined_df = combine_method_summaries(results_dict)
    
    if combined_df.empty or metric not in combined_df.columns:
        return pd.DataFrame()
    
    # Pivot table with countries as rows and methods as columns
    comparison_df = combined_df.pivot_table(
        index='country',
        columns='method',
        values=metric,
        aggfunc='first'
    )
    
    # Add summary statistics
    if metric in ['ATT', 'RMSE_ratio']:
        comparison_df.loc['Mean'] = comparison_df.mean()
        comparison_df.loc['Median'] = comparison_df.median()
        comparison_df.loc['Std'] = comparison_df.std()
    elif metric == 'p_value':
        comparison_df.loc['Mean'] = comparison_df.mean()
        comparison_df.loc['Significant (%)'] = (comparison_df <= 0.1).sum() / comparison_df.count() * 100
    
    return comparison_df


def create_significance_summary(results_dict: Dict[str, CausalResults]) -> pd.DataFrame:
    """
    Create summary of significant results across methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
        
    Returns:
    --------
    pd.DataFrame : Significance summary
    """
    
    combined_df = combine_method_summaries(results_dict)
    
    if combined_df.empty:
        return pd.DataFrame()
    
    # Calculate significance statistics by method
    significance_data = []
    
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        
        total_countries = len(method_df)
        significant_countries = sum(method_df['significant'] == 'Yes')
        significant_pct = (significant_countries / total_countries * 100) if total_countries > 0 else 0
        
        # Mean ATT for significant countries
        sig_att = method_df[method_df['significant'] == 'Yes']['ATT']
        mean_sig_att = sig_att.mean() if not sig_att.empty else np.nan
        
        # Mean p-value
        mean_p_value = method_df['p_value'].mean()
        
        significance_data.append({
            'method': method,
            'total_countries': total_countries,
            'significant_countries': significant_countries,
            'significant_percentage': significant_pct,
            'mean_significant_ATT': mean_sig_att,
            'mean_p_value': mean_p_value
        })
    
    significance_df = pd.DataFrame(significance_data)
    significance_df = significance_df.sort_values('significant_percentage', ascending=False)
    
    return significance_df


def create_robustness_table(results_dict: Dict[str, CausalResults]) -> pd.DataFrame:
    """
    Create robustness comparison table across methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
        
    Returns:
    --------
    pd.DataFrame : Robustness comparison table
    """
    
    combined_df = combine_method_summaries(results_dict)
    
    if combined_df.empty:
        return pd.DataFrame()
    
    # Calculate robustness metrics by method
    robustness_data = []
    
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        
        # Basic statistics
        total_countries = len(method_df)
        
        # RMSE statistics
        mean_rmse_ratio = method_df['RMSE_ratio'].mean()
        median_rmse_ratio = method_df['RMSE_ratio'].median()
        
        # Effect size statistics
        mean_att = method_df['ATT'].mean()
        median_att = method_df['ATT'].median()
        att_std = method_df['ATT'].std()
        
        # Significance statistics
        significant_pct = (method_df['significant'] == 'Yes').sum() / total_countries * 100
        mean_p_value = method_df['p_value'].mean()
        
        # Pre-treatment fit quality
        mean_pre_rmse = method_df['RMSE_pre'].mean()
        
        robustness_data.append({
            'method': method,
            'n_countries': total_countries,
            'mean_ATT': mean_att,
            'median_ATT': median_att,
            'ATT_std': att_std,
            'mean_RMSE_ratio': mean_rmse_ratio,
            'median_RMSE_ratio': median_rmse_ratio,
            'mean_pre_RMSE': mean_pre_rmse,
            'significant_pct': significant_pct,
            'mean_p_value': mean_p_value
        })
    
    robustness_df = pd.DataFrame(robustness_data)
    
    return robustness_df


def export_summary_tables(results_dict: Dict[str, CausalResults],
                         output_dir: str):
    """
    Export all summary tables to CSV files.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
    output_dir : str
        Output directory
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Individual method summaries
    for method_name, results in results_dict.items():
        try:
            summary_df = create_standardized_summary_table(results, method_name)
            filename = f"{method_name.lower().replace(' ', '_')}_summary.csv"
            summary_df.to_csv(os.path.join(output_dir, filename), index=False)
        except Exception as e:
            warnings.warn(f"Failed to export summary for {method_name}: {e}")
    
    # Combined summary
    try:
        combined_df = combine_method_summaries(results_dict)
        combined_df.to_csv(os.path.join(output_dir, "combined_summary.csv"), index=False)
    except Exception as e:
        warnings.warn(f"Failed to export combined summary: {e}")
    
    # Method comparisons
    try:
        for metric in ['ATT', 'RMSE_ratio', 'p_value']:
            comparison_df = create_method_comparison_table(results_dict, metric)
            if not comparison_df.empty:
                filename = f"method_comparison_{metric.lower()}.csv"
                comparison_df.to_csv(os.path.join(output_dir, filename))
    except Exception as e:
        warnings.warn(f"Failed to export method comparisons: {e}")
    
    # Significance summary
    try:
        significance_df = create_significance_summary(results_dict)
        significance_df.to_csv(os.path.join(output_dir, "significance_summary.csv"), index=False)
    except Exception as e:
        warnings.warn(f"Failed to export significance summary: {e}")
    
    # Robustness table
    try:
        robustness_df = create_robustness_table(results_dict)
        robustness_df.to_csv(os.path.join(output_dir, "robustness_comparison.csv"), index=False)
    except Exception as e:
        warnings.warn(f"Failed to export robustness table: {e}")
    
    print(f"Summary tables exported to {output_dir}")


def format_summary_for_latex(summary_df: pd.DataFrame,
                            significant_only: bool = False) -> str:
    """
    Format summary table for LaTeX output.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary table
    significant_only : bool
        Whether to include only significant results
        
    Returns:
    --------
    str : LaTeX table code
    """
    
    if significant_only:
        summary_df = summary_df[summary_df['significant'] == 'Yes']
    
    # Round numeric columns
    numeric_cols = ['ATT', 'RMSE_pre', 'RMSE_post', 'RMSE_ratio', 'p_value']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(3)
    
    # Convert to LaTeX
    latex_str = summary_df.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * (len(summary_df.columns) - 1),
        caption="Summary of Causal Inference Results",
        label="tab:summary"
    )
    
    return latex_str
