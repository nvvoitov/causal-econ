"""
Base classes and common utilities for causal economic analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    
    # Common parameters
    n_donors: int = 20
    n_placebos: int = 20
    random_state: int = 42
    
    # Model-specific parameters (will be extended by subclasses)
    def update(self, **kwargs):
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}")


class CausalResults:
    """
    Standardized results container for causal analysis methods.
    
    This class provides a common interface while allowing each method
    to store its unique outputs.
    """
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.summary_table = None
        self.plots = {}
        self.raw_results = {}
        self.logs = ""
        
        # Standard metrics available across methods
        self.country_results = {}
        self.placebo_results = {}
        
    def add_country_result(self, country: str, result_dict: Dict):
        """Add results for a specific country."""
        self.country_results[country] = result_dict
        
    def add_placebo_result(self, placebo_dict: Dict):
        """Add placebo test results."""
        self.placebo_results.update(placebo_dict)
        
    def get_summary_metrics(self) -> pd.DataFrame:
        """Extract summary metrics across all countries."""
        if self.summary_table is not None:
            return self.summary_table
            
        # Generate summary from country results
        rows = []
        for country, result in self.country_results.items():
            if isinstance(result, dict) and 'att' in result:
                rows.append({
                    'country': country,
                    'method': self.method_name,
                    'ATT': result.get('att', np.nan),
                    'RMSE_pre': result.get('pre_rmse', np.nan),
                    'RMSE_post': result.get('post_rmse', np.nan),
                    'RMSE_ratio': result.get('rmse_ratio', np.nan),
                    'p_value': result.get('p_value', np.nan),
                    'significant': result.get('significant', False)
                })
        
        return pd.DataFrame(rows)


# Common utility functions

def ensure_numpy_array(data: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
    """Convert various data types to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to numpy array")


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray, 
                   mask: Optional[np.ndarray] = None) -> float:
    """Calculate Root Mean Square Error with optional masking."""
    actual = ensure_numpy_array(actual)
    predicted = ensure_numpy_array(predicted)
    
    if mask is not None:
        mask = ensure_numpy_array(mask).astype(bool)
        actual = actual[mask]
        predicted = predicted[mask]
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
    if not np.any(valid_mask):
        return np.nan
        
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]
    
    return np.sqrt(np.mean((actual - predicted)**2))


def calculate_treatment_effect(actual: np.ndarray, counterfactual: np.ndarray,
                              treatment_mask: np.ndarray) -> Dict[str, float]:
    """Calculate treatment effect metrics."""
    actual = ensure_numpy_array(actual)
    counterfactual = ensure_numpy_array(counterfactual)
    treatment_mask = ensure_numpy_array(treatment_mask).astype(bool)
    
    # Pre-treatment RMSE
    pre_mask = ~treatment_mask
    pre_rmse = calculate_rmse(actual, counterfactual, pre_mask)
    
    # Post-treatment RMSE and ATT
    post_mask = treatment_mask
    post_rmse = calculate_rmse(actual, counterfactual, post_mask)
    
    # Average Treatment Effect on Treated
    if np.any(post_mask):
        post_effects = actual[post_mask] - counterfactual[post_mask]
        # Remove NaN values
        valid_effects = post_effects[~np.isnan(post_effects)]
        att = np.mean(valid_effects) if len(valid_effects) > 0 else np.nan
    else:
        att = np.nan
    
    # RMSE ratio
    rmse_ratio = post_rmse / pre_rmse if pre_rmse > 0 else np.nan
    
    return {
        'att': att,
        'pre_rmse': pre_rmse,
        'post_rmse': post_rmse,
        'rmse_ratio': rmse_ratio
    }


def validate_panel_data(panel: pd.DataFrame, 
                       treated_countries: List[str],
                       treatment_years: Dict[str, int]) -> bool:
    """Validate panel data structure for causal analysis."""
    
    # Check if treated countries exist in panel
    missing_countries = [c for c in treated_countries if c not in panel.columns]
    if missing_countries:
        raise ValueError(f"Treated countries not in panel: {missing_countries}")
    
    # Check if treatment years are within panel range
    panel_years = panel.index
    min_year, max_year = panel_years.min(), panel_years.max()
    
    invalid_years = {c: y for c, y in treatment_years.items() 
                    if not (min_year <= y <= max_year)}
    if invalid_years:
        raise ValueError(f"Treatment years outside panel range: {invalid_years}")
    
    # Check for minimum data requirements
    for country in treated_countries:
        if country in treatment_years:
            treatment_year = treatment_years[country]
            pre_periods = sum(panel.index < treatment_year)
            post_periods = sum(panel.index >= treatment_year)
            
            if pre_periods < 2:
                warnings.warn(f"Country {country} has only {pre_periods} pre-treatment periods")
            if post_periods < 1:
                warnings.warn(f"Country {country} has no post-treatment periods")
    
    return True


def country_name_mapper() -> Dict[str, str]:
    """Return standard country name mapping for consistency."""
    return {
        'Korea, Rep.': 'Korea',
        'Russian Federation': 'Russia'
    }


def get_reverse_country_mapping() -> Dict[str, str]:
    """Return reverse country name mapping."""
    mapping = country_name_mapper()
    return {v: k for k, v in mapping.items()}


def standardize_country_names(countries: List[str], 
                             direction: str = 'to_standard') -> List[str]:
    """
    Standardize country names for consistency across datasets.
    
    Parameters:
    -----------
    countries : list
        List of country names
    direction : str
        'to_standard' or 'from_standard'
    """
    if direction == 'to_standard':
        mapping = country_name_mapper()
    else:
        mapping = get_reverse_country_mapping()
    
    return [mapping.get(country, country) for country in countries]
