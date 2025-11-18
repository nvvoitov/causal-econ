"""
Base classes and common utilities for causal economic analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
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


def calculate_causal_metrics(effect_dict: Dict[int, float],
                            treatment_year: int,
                            years: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Calculate standardized causal effect metrics from effect values.

    This function eliminates ~100 lines of duplicate metric calculation code
    across all causal methods by providing a centralized implementation.

    Parameters:
    -----------
    effect_dict : dict
        Dictionary mapping years to effect values (actual - counterfactual)
    treatment_year : int
        Year when treatment began
    years : list, optional
        List of all years to consider. If None, derived from effect_dict keys.

    Returns:
    --------
    dict : Standardized metrics containing:
        - 'pre_rmse': Root mean square error in pre-treatment period
        - 'post_rmse': Root mean square error in post-treatment period
        - 'rmse_ratio': Ratio of post/pre RMSE (quality indicator)
        - 'att': Average Treatment Effect on Treated
        - 'n_pre_periods': Number of valid pre-treatment periods
        - 'n_post_periods': Number of valid post-treatment periods

    Notes:
    ------
    - NaN values in effect_dict are automatically filtered out
    - Returns NaN for metrics if insufficient data is available
    - RMSE ratio > 1 suggests poor fit (post-treatment error exceeds pre-treatment)
    - RMSE ratio < 1 suggests good fit

    Example:
    --------
    >>> effect_dict = {2000: 0.1, 2001: 0.2, 2002: 1.5, 2003: 1.8}
    >>> metrics = calculate_causal_metrics(effect_dict, treatment_year=2002)
    >>> print(metrics['att'])  # Average of 2002 and 2003 effects
    1.65
    """
    if years is None:
        years = sorted(effect_dict.keys())

    # Split into pre and post treatment periods
    pre_years = [y for y in years if y < treatment_year and y in effect_dict]
    post_years = [y for y in years if y >= treatment_year and y in effect_dict]

    # Extract effects, filtering out None and NaN values
    pre_effects = [effect_dict[y] for y in pre_years
                   if effect_dict[y] is not None and not np.isnan(effect_dict[y])]
    post_effects = [effect_dict[y] for y in post_years
                    if effect_dict[y] is not None and not np.isnan(effect_dict[y])]

    # Calculate pre-treatment RMSE
    pre_rmse = np.sqrt(np.mean([e**2 for e in pre_effects])) if pre_effects else np.nan

    # Calculate post-treatment RMSE
    post_rmse = np.sqrt(np.mean([e**2 for e in post_effects])) if post_effects else np.nan

    # Calculate RMSE ratio
    rmse_ratio = post_rmse / pre_rmse if pre_rmse and pre_rmse > 0 else np.nan

    # Calculate Average Treatment Effect on Treated
    att = np.mean(post_effects) if post_effects else np.nan

    return {
        'pre_rmse': pre_rmse,
        'post_rmse': post_rmse,
        'rmse_ratio': rmse_ratio,
        'att': att,
        'n_pre_periods': len(pre_effects),
        'n_post_periods': len(post_effects)
    }


def split_pre_post_periods(years: List[int],
                           treatment_year: int) -> tuple:
    """
    Split years into pre and post treatment periods.

    This is a simple utility that eliminates ~20 duplicate list comprehensions
    across the codebase.

    Parameters:
    -----------
    years : list
        List of year values
    treatment_year : int
        Year when treatment began

    Returns:
    --------
    tuple : (pre_years, post_years)
        - pre_years: List of years before treatment
        - post_years: List of years at or after treatment

    Example:
    --------
    >>> years = [2000, 2001, 2002, 2003, 2004]
    >>> pre, post = split_pre_post_periods(years, treatment_year=2002)
    >>> print(pre)
    [2000, 2001]
    >>> print(post)
    [2002, 2003, 2004]
    """
    pre_years = [y for y in years if y < treatment_year]
    post_years = [y for y in years if y >= treatment_year]
    return pre_years, post_years


class DataValidator:
    """
    Centralized data validation for causal analysis.

    This class eliminates ~50 lines of duplicate validation code across all
    methods by providing a single source of truth for data quality checks.

    Attributes:
    -----------
    data_dict : dict
        Dictionary containing panel data and metadata
    panel : pd.DataFrame
        Panel data matrix (years x countries)
    treatment_years : dict
        Dictionary mapping country names to treatment years
    treated_countries : list
        List of countries receiving treatment

    Example:
    --------
    >>> validator = DataValidator(data_dict)
    >>> is_valid, error_msg = validator.validate_for_analysis('USA', donor_pool)
    >>> if not is_valid:
    ...     print(f"Validation failed: {error_msg}")
    """

    def __init__(self, data_dict: Dict):
        """
        Initialize the data validator.

        Parameters:
        -----------
        data_dict : dict
            Dictionary with keys:
                - 'panel': DataFrame with countries as columns, years as index
                - 'treatment_years': Dict mapping countries to treatment years
                - 'treated_countries': List of treated country names (optional)
        """
        self.data_dict = data_dict
        self.panel = data_dict.get('panel')
        self.treatment_years = data_dict.get('treatment_years', {})
        self.treated_countries = data_dict.get('treated_countries', [])

    def validate_country(self, country: str, raise_error: bool = False) -> bool:
        """
        Check if country exists in panel data.

        Parameters:
        -----------
        country : str
            Country name to validate
        raise_error : bool, optional
            If True, raise ValueError instead of returning False (default: False)

        Returns:
        --------
        bool : True if country exists in panel, False otherwise

        Raises:
        -------
        ValueError : If raise_error=True and country not found
        """
        if self.panel is None:
            if raise_error:
                raise ValueError("Panel data not available in data_dict")
            return False

        valid = country in self.panel.columns

        if not valid and raise_error:
            raise ValueError(f"Country '{country}' not found in panel data")

        return valid

    def validate_treatment_year(self, country: str, raise_error: bool = False) -> bool:
        """
        Check if country has treatment year defined.

        Parameters:
        -----------
        country : str
            Country name to validate
        raise_error : bool, optional
            If True, raise ValueError instead of returning False (default: False)

        Returns:
        --------
        bool : True if treatment year exists, False otherwise

        Raises:
        -------
        ValueError : If raise_error=True and treatment year not found
        """
        valid = country in self.treatment_years

        if not valid and raise_error:
            raise ValueError(f"No treatment year found for '{country}'")

        return valid

    def validate_donor_pool(self,
                           donor_pool: List[str],
                           min_donors: int = 2,
                           raise_error: bool = False) -> bool:
        """
        Check if donor pool is adequate for analysis.

        Parameters:
        -----------
        donor_pool : list
            List of donor country names
        min_donors : int, optional
            Minimum number of donors required (default: 2)
        raise_error : bool, optional
            If True, raise ValueError instead of returning False (default: False)

        Returns:
        --------
        bool : True if donor pool is adequate, False otherwise

        Raises:
        -------
        ValueError : If raise_error=True and donor pool insufficient
        """
        valid = len(donor_pool) >= min_donors

        if not valid and raise_error:
            raise ValueError(
                f"Insufficient donors: {len(donor_pool)} < {min_donors}"
            )

        return valid

    def validate_data_availability(self,
                                   country: str,
                                   raise_error: bool = False) -> bool:
        """
        Check if country has any valid (non-NaN) data.

        Parameters:
        -----------
        country : str
            Country name to validate
        raise_error : bool, optional
            If True, raise ValueError instead of returning False (default: False)

        Returns:
        --------
        bool : True if country has valid data, False otherwise

        Raises:
        -------
        ValueError : If raise_error=True and no valid data found
        """
        if not self.validate_country(country):
            if raise_error:
                raise ValueError(f"Country '{country}' not in panel")
            return False

        country_data = self.panel[country]
        valid = not country_data.isna().all()

        if not valid and raise_error:
            raise ValueError(f"No valid data for country '{country}'")

        return valid

    def validate_for_analysis(self,
                             country: str,
                             donor_pool: List[str],
                             min_donors: int = 2,
                             check_data: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive validation for running causal analysis.

        This method performs all standard validation checks and returns
        a clear error message if validation fails.

        Parameters:
        -----------
        country : str
            Country name to validate
        donor_pool : list
            List of donor countries
        min_donors : int, optional
            Minimum number of donors required (default: 2)
        check_data : bool, optional
            If True, also check for data availability (default: True)

        Returns:
        --------
        tuple : (is_valid, error_message)
            - is_valid: True if all validation checks pass
            - error_message: Description of validation failure (None if valid)

        Example:
        --------
        >>> validator = DataValidator(data_dict)
        >>> valid, error = validator.validate_for_analysis('USA', ['Canada', 'UK'])
        >>> if not valid:
        ...     print(f"Cannot run analysis: {error}")
        """
        # Check if country exists in panel
        if not self.validate_country(country):
            return False, f"Country '{country}' not found in panel data"

        # Check if treatment year is defined
        if not self.validate_treatment_year(country):
            return False, f"No treatment year found for '{country}'"

        # Check donor pool size
        if not self.validate_donor_pool(donor_pool, min_donors):
            return False, f"Insufficient donors: {len(donor_pool)} < {min_donors}"

        # Check data availability if requested
        if check_data and not self.validate_data_availability(country):
            return False, f"No valid data for country '{country}'"

        return True, None

    def get_pre_post_data_counts(self, country: str) -> Dict[str, int]:
        """
        Get counts of available pre and post treatment periods.

        Parameters:
        -----------
        country : str
            Country name

        Returns:
        --------
        dict : Dictionary containing:
            - 'pre_periods': Number of pre-treatment periods with data
            - 'post_periods': Number of post-treatment periods with data
            - 'total_periods': Total periods with data

        Example:
        --------
        >>> validator = DataValidator(data_dict)
        >>> counts = validator.get_pre_post_data_counts('USA')
        >>> print(f"Pre-treatment periods: {counts['pre_periods']}")
        """
        if not self.validate_country(country) or not self.validate_treatment_year(country):
            return {'pre_periods': 0, 'post_periods': 0, 'total_periods': 0}

        treatment_year = self.treatment_years[country]
        country_data = self.panel[country]

        pre_periods = sum((self.panel.index < treatment_year) & ~country_data.isna())
        post_periods = sum((self.panel.index >= treatment_year) & ~country_data.isna())

        return {
            'pre_periods': int(pre_periods),
            'post_periods': int(post_periods),
            'total_periods': int(pre_periods + post_periods)
        }
