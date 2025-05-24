"""
Unified placebo testing framework for causal inference methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
import warnings

from ..core.base import CausalResults, calculate_treatment_effect


class PlaceboTest:
    """Standardized placebo test container."""
    
    def __init__(self, country: str, treatment_year: int, method: str):
        self.country = country
        self.treatment_year = treatment_year
        self.method = method
        self.metrics = {}
        self.placebo_results = []
        self.p_values = {}
    
    def add_metric(self, name: str, value: float):
        """Add a metric for the treated unit."""
        self.metrics[name] = value
    
    def add_placebo_result(self, placebo_country: str, metrics: Dict[str, float]):
        """Add results from a placebo test."""
        self.placebo_results.append({
            'country': placebo_country,
            'treatment_year': self.treatment_year,
            'method': self.method,
            **metrics
        })
    
    def calculate_p_values(self, metric_names: Optional[List[str]] = None):
        """Calculate p-values for specified metrics."""
        if metric_names is None:
            metric_names = list(self.metrics.keys())
        
        for metric_name in metric_names:
            if metric_name not in self.metrics:
                self.p_values[metric_name] = np.nan
                continue
            
            treated_value = self.metrics[metric_name]
            
            # Handle special cases for different metrics
            if metric_name in ['rmse_ratio', 'post_rmse']:
                # Higher is worse for these metrics
                placebo_values = [p.get(metric_name, np.nan) for p in self.placebo_results]
                valid_values = [v for v in placebo_values if not np.isnan(v)]
                
                if valid_values and not np.isnan(treated_value):
                    p_value = sum(1 for v in valid_values if v >= treated_value) / len(valid_values)
                else:
                    p_value = np.nan
                    
            elif metric_name in ['att', 'treatment_effect']:
                # Use absolute value for treatment effects
                placebo_values = [abs(p.get(metric_name, np.nan)) for p in self.placebo_results]
                valid_values = [v for v in placebo_values if not np.isnan(v)]
                treated_abs = abs(treated_value)
                
                if valid_values and not np.isnan(treated_abs):
                    p_value = sum(1 for v in valid_values if v >= treated_abs) / len(valid_values)
                else:
                    p_value = np.nan
            else:
                # Default case - higher is worse
                placebo_values = [p.get(metric_name, np.nan) for p in self.placebo_results]
                valid_values = [v for v in placebo_values if not np.isnan(v)]
                
                if valid_values and not np.isnan(treated_value):
                    p_value = sum(1 for v in valid_values if v >= treated_value) / len(valid_values)
                else:
                    p_value = np.nan
            
            self.p_values[metric_name] = p_value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of placebo test results."""
        return {
            'country': self.country,
            'method': self.method,
            'treatment_year': self.treatment_year,
            'n_placebos': len(self.placebo_results),
            'metrics': self.metrics,
            'p_values': self.p_values,
            'significant': {name: p_val <= 0.1 for name, p_val in self.p_values.items() if not np.isnan(p_val)}
        }


def run_unified_placebo_test(method_name: str,
                           country: str, 
                           country_result: Dict,
                           data_dict: Dict,
                           donor_pool: List[str],
                           placebo_function: Callable,
                           n_placebos: int = 20,
                           **kwargs) -> PlaceboTest:
    """
    Run unified placebo test for any causal inference method.
    
    Parameters:
    -----------
    method_name : str
        Name of the causal inference method
    country : str
        Treated country name
    country_result : dict
        Results from the main analysis for the treated country
    data_dict : dict
        Data dictionary
    donor_pool : list
        List of donor countries
    placebo_function : callable
        Method-specific placebo function
    n_placebos : int
        Number of placebo tests to run
    **kwargs : dict
        Additional arguments for the placebo function
        
    Returns:
    --------
    PlaceboTest : Unified placebo test results
    """
    
    # Extract treatment year
    treatment_year = country_result.get('treatment_year')
    if treatment_year is None:
        raise ValueError(f"No treatment year found for {country}")
    
    # Initialize placebo test
    placebo_test = PlaceboTest(country, treatment_year, method_name)
    
    # Add treated unit metrics
    standard_metrics = ['att', 'pre_rmse', 'post_rmse', 'rmse_ratio', 'treatment_effect']
    for metric in standard_metrics:
        if metric in country_result:
            placebo_test.add_metric(metric, country_result[metric])
    
    # Run method-specific placebo function
    try:
        placebo_result = placebo_function(
            country=country,
            country_result=country_result,
            data_dict=data_dict,
            n_placebos=n_placebos,
            **kwargs
        )
        
        # Extract placebo results in standardized format
        if 'placebos' in placebo_result:
            for placebo in placebo_result['placebos']:
                placebo_metrics = {}
                for metric in standard_metrics:
                    if metric in placebo:
                        placebo_metrics[metric] = placebo[metric]
                
                placebo_country_name = placebo.get('country', 'Unknown')
                placebo_test.add_placebo_result(placebo_country_name, placebo_metrics)
        
        # Calculate p-values
        placebo_test.calculate_p_values()
        
    except Exception as e:
        warnings.warn(f"Placebo test failed for {country} using {method_name}: {e}")
    
    return placebo_test


def run_placebo_battery(results: CausalResults,
                       data_dict: Dict,
                       donor_pools: Dict[str, List[str]],
                       n_placebos: int = 20) -> Dict[str, PlaceboTest]:
    """
    Run placebo tests for all countries in a CausalResults object.
    
    Parameters:
    -----------
    results : CausalResults
        Results from causal analysis
    data_dict : dict
        Data dictionary
    donor_pools : dict
        Donor pools for each country
    n_placebos : int
        Number of placebo tests per country
        
    Returns:
    --------
    dict : Dictionary mapping country names to PlaceboTest objects
    """
    
    placebo_tests = {}
    method_name = results.method_name
    
    # Import method-specific placebo functions
    if method_name == 'Difference-in-Differences':
        from ..models.did import run_placebo_tests_did as placebo_func
    elif method_name == 'Synthetic Control':
        from ..models.synthetic_control import run_placebo_test_sc as placebo_func
    elif method_name == 'Synthetic Difference-in-Differences':
        from ..models.sdid import run_placebo_test_sdid as placebo_func
    elif method_name == 'Generalized Synthetic Control':
        from ..models.gsc import run_placebo_test_gsc as placebo_func
    elif method_name == 'Causal Economic2Vec':
        from ..models.ce2v import run_gemini_placebo_test_ce2v as placebo_func
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Run placebo tests for each country
    for country in results.country_results.keys():
        if country not in donor_pools:
            warnings.warn(f"No donor pool for {country}, skipping placebo test")
            continue
            
        country_result = results.raw_results.get(country, {})
        if not country_result:
            warnings.warn(f"No raw results for {country}, skipping placebo test")
            continue
        
        # Extract the method-specific result
        if method_name == 'Difference-in-Differences':
            method_result = country_result.get('did_result')
            extra_kwargs = {'donor_pool': donor_pools[country], 'data_dict': data_dict}
        elif method_name == 'Synthetic Control':
            method_result = country_result.get('sc_result')
            extra_kwargs = {'data_dict': data_dict}
        elif method_name == 'Synthetic Difference-in-Differences':
            method_result = country_result.get('sdid_result')
            extra_kwargs = {'data_dict': data_dict}
        elif method_name == 'Generalized Synthetic Control':
            method_result = country_result.get('gsc_result')
            extra_kwargs = {'data_dict': data_dict, 'factor_model': results.factor_model}
        elif method_name == 'Causal Economic2Vec':
            method_result = country_result.get('ce2v_result')
            extra_kwargs = {
                'data_dict': data_dict,
                'model': results.model,
                'scaler': results.scaler,
                'feature_cols': results.feature_cols
            }
        else:
            continue
        
        if method_result is None:
            continue
        
        try:
            placebo_test = run_unified_placebo_test(
                method_name=method_name,
                country=country,
                country_result=method_result,
                placebo_function=placebo_func,
                n_placebos=n_placebos,
                **extra_kwargs
            )
            placebo_tests[country] = placebo_test
            
        except Exception as e:
            warnings.warn(f"Failed to run placebo test for {country}: {e}")
    
    return placebo_tests


def create_placebo_summary_table(placebo_tests: Dict[str, PlaceboTest]) -> pd.DataFrame:
    """
    Create summary table of placebo test results.
    
    Parameters:
    -----------
    placebo_tests : dict
        Dictionary mapping countries to PlaceboTest objects
        
    Returns:
    --------
    pd.DataFrame : Summary table of placebo results
    """
    
    summary_data = []
    
    for country, placebo_test in placebo_tests.items():
        summary = placebo_test.get_summary()
        
        row = {
            'country': country,
            'method': summary['method'],
            'n_placebos': summary['n_placebos']
        }
        
        # Add metrics
        for metric_name, value in summary['metrics'].items():
            row[metric_name] = value
        
        # Add p-values
        for metric_name, p_value in summary['p_values'].items():
            row[f'p_val_{metric_name}'] = p_value
        
        # Add significance indicators
        for metric_name, is_sig in summary['significant'].items():
            row[f'significant_{metric_name}'] = 'Yes' if is_sig else 'No'
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by primary p-value (typically rmse_ratio or att)
    if 'p_val_rmse_ratio' in summary_df.columns:
        summary_df = summary_df.sort_values('p_val_rmse_ratio')
    elif 'p_val_att' in summary_df.columns:
        summary_df = summary_df.sort_values('p_val_att')
    
    return summary_df


def compare_placebo_performance(placebo_tests: Dict[str, PlaceboTest],
                              metric: str = 'rmse_ratio') -> pd.DataFrame:
    """
    Compare placebo test performance across countries.
    
    Parameters:
    -----------
    placebo_tests : dict
        Dictionary of placebo tests
    metric : str
        Metric to compare ('rmse_ratio', 'att', etc.)
        
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    
    comparison_data = []
    
    for country, placebo_test in placebo_tests.items():
        summary = placebo_test.get_summary()
        
        if metric in summary['metrics'] and metric in summary['p_values']:
            comparison_data.append({
                'country': country,
                'method': summary['method'],
                f'{metric}_value': summary['metrics'][metric],
                f'{metric}_p_value': summary['p_values'][metric],
                'n_placebos': summary['n_placebos'],
                'rank': None  # Will be filled after sorting
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        # Rank by p-value (lower is better)
        comparison_df = comparison_df.sort_values(f'{metric}_p_value')
        comparison_df['rank'] = range(1, len(comparison_df) + 1)
    
    return comparison_df
