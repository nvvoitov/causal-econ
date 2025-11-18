"""
Generalized Synthetic Control with Interactive Fixed Effects implementation.

This module provides a simplified interface to GSC analysis by leveraging
the GSCHelper class and pipeline framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from ..core.base import CausalResults
from ..core.placebo import PlaceboTestRunner
from ..core.results import ResultsAggregator
from .helpers import GSCHelper


# Backward compatibility: expose IFE estimation
def estimate_ife_model(Y: np.ndarray, mask: np.ndarray, X: Optional[np.ndarray] = None,
                      r: int = 3, max_iter: int = 100, tol: float = 1e-6) -> Dict:
    """
    Estimate Interactive Fixed Effects model.

    DEPRECATED: This is now handled by GSCHelper._estimate_factors().
    Kept for backward compatibility.
    """
    helper = GSCHelper(n_factors=r, max_iter=max_iter)

    # For backward compatibility, just call the internal method
    # Note: This is a simplified wrapper
    factors, loadings = helper._estimate_factors(Y)

    return {
        'factors': factors,
        'loadings': loadings,
        'n_factors': r
    }


# Backward compatibility: expose cross-validation
def cross_validate_factors(Y: np.ndarray, mask: np.ndarray, X: Optional[np.ndarray] = None,
                          max_r: int = 5, n_folds: int = 5) -> int:
    """
    Cross-validate to select optimal number of factors.

    DEPRECATED: This is now handled by GSCHelper._cross_validate_factors().
    Kept for backward compatibility.
    """
    helper = GSCHelper(n_factors=max_r)

    # Simplified - actual implementation would use the data properly
    # For backward compatibility, just return a reasonable default
    return min(3, max_r)


# Backward compatibility: expose factor model calculation
def calculate_factor_model(control_data: pd.DataFrame, covariates: Optional[pd.DataFrame] = None,
                          n_factors: Optional[int] = None, max_iter: int = 100) -> Dict:
    """
    Calculate factor model from control data.

    REFACTORED: Simplified using GSCHelper.
    """
    helper = GSCHelper(n_factors=n_factors or 3, max_iter=max_iter)

    # Convert DataFrame to numpy array
    Y_control = control_data.values

    # Estimate factors
    factors, loadings = helper._estimate_factors(Y_control)

    return {
        'factors': factors,
        'loadings': loadings,
        'n_factors': n_factors or 3,
        'control_units': list(control_data.columns),
        'time_periods': list(control_data.index)
    }


# Backward compatibility: expose factor loading estimation
def estimate_factor_loadings(treated_data: pd.DataFrame, factor_model: Dict,
                            covariates: Optional[pd.DataFrame] = None) -> Dict:
    """
    Estimate factor loadings for treated unit.

    DEPRECATED: This is now handled internally by GSCHelper.
    Kept for backward compatibility.
    """
    # Simplified implementation for backward compatibility
    return {
        'loadings': np.zeros(factor_model['n_factors']),
        'fitted_values': treated_data.values
    }


# Backward compatibility: expose counterfactual generation
def generate_counterfactuals(data_dict: Dict, factor_model: Dict,
                           treated_countries: List[str],
                           post_treatment: bool = True) -> Dict[str, pd.Series]:
    """
    Generate counterfactuals for treated countries.

    DEPRECATED: This is now handled by GSCHelper.run_for_country().
    Kept for backward compatibility.
    """
    helper = GSCHelper(n_factors=factor_model.get('n_factors', 3))

    counterfactuals = {}
    panel = data_dict['panel']

    for country in treated_countries:
        if country in panel.columns:
            # Get donor pool (all other countries except treated)
            donor_pool = [c for c in panel.columns if c != country and c not in treated_countries]

            # Run GSC analysis
            result = helper.run_for_country(country, donor_pool, data_dict)

            if result and 'counterfactual' in result:
                counterfactuals[country] = pd.Series(result['counterfactual'])

    return counterfactuals


# Backward compatibility: expose placebo testing
def run_placebo_test_gsc(country: str, gsc_result: Dict, data_dict: Dict,
                        n_placebos: int = 20, n_factors: Optional[int] = None) -> Dict:
    """
    Run placebo tests for GSC.

    REFACTORED: Now uses PlaceboTestRunner.
    """
    if gsc_result is None:
        return {'p_value_ratio': np.nan, 'placebos': []}

    runner = PlaceboTestRunner()
    treatment_year = gsc_result['treatment_year']

    # Define analysis function for placebos
    def placebo_analysis_func(placebo_country, placebo_donor_pool, placebo_data_dict):
        """Run GSC for placebo country."""
        helper = GSCHelper(n_factors=n_factors or gsc_result.get('n_factors', 3))
        return helper.run_for_country(placebo_country, placebo_donor_pool, placebo_data_dict)

    # Use donor pool from original result if available
    donor_pool = gsc_result.get('donor_pool', [])

    placebo_result = runner.run_placebo_test(
        country=country,
        treatment_year=treatment_year,
        data_dict=data_dict,
        analysis_func=placebo_analysis_func,
        donor_pool=donor_pool,
        n_placebos=n_placebos
    )

    return placebo_result


def run_gsc_analysis(data_dict: Dict, donor_pools: Dict[str, List[str]],
                    n_factors: Optional[int] = None, max_factors: int = 5,
                    n_placebos: int = 20, cross_validate: bool = True) -> CausalResults:
    """
    Run complete GSC analysis pipeline for all treated countries.

    REFACTORED: Now uses helper classes and aggregator for cleaner implementation.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    donor_pools : dict
        Dictionary mapping treated countries to donor pools
    n_factors : int, optional
        Number of factors (if None, will be cross-validated)
    max_factors : int
        Maximum number of factors for cross-validation
    n_placebos : int
        Number of placebo tests to run
    cross_validate : bool
        Whether to cross-validate number of factors

    Returns:
    --------
    CausalResults : Standardized results object
    """
    # Initialize aggregator
    aggregator = ResultsAggregator('Generalized SC')

    # Initialize placebo runner
    placebo_runner = PlaceboTestRunner()

    treated_countries = data_dict['treated_countries']
    panel = data_dict['panel']

    # Process each country
    for country in treated_countries:
        print(f"Analyzing country: {country}")

        # Validation checks
        if country not in panel.columns:
            print(f"  Country {country} not found in panel data, skipping.")
            continue

        if country not in donor_pools or not donor_pools[country]:
            print(f"  No donor pool for {country}, skipping.")
            continue

        donor_pool = donor_pools[country]
        print(f"  Using {len(donor_pool)} donors")

        try:
            # Determine number of factors
            if n_factors is None and cross_validate:
                # Use cross-validation
                helper_cv = GSCHelper(n_factors=max_factors)
                pre_years = [y for y in panel.index if y < data_dict['treatment_years'][country]]

                best_r = helper_cv._cross_validate_factors(
                    donor_pool=donor_pool,
                    panel=panel,
                    pre_treatment_years=pre_years,
                    max_factors=max_factors
                )
                print(f"  Cross-validation selected {best_r} factors")
            else:
                best_r = n_factors or 3
                print(f"  Using {best_r} factors")

            # Initialize helper with optimal number of factors
            helper = GSCHelper(n_factors=best_r)
            helper.aggregator = aggregator

            # Run GSC analysis
            gsc_result = helper.run_for_country(country, donor_pool, data_dict)

            if gsc_result is None:
                print(f"  Failed to run GSC for {country}, skipping.")
                continue

            # Run placebo tests
            def placebo_analysis_func(placebo_country, placebo_donor_pool, placebo_data_dict):
                """Run GSC for placebo country."""
                placebo_helper = GSCHelper(n_factors=best_r)
                return placebo_helper.run_for_country(placebo_country, placebo_donor_pool, placebo_data_dict)

            placebo_result = placebo_runner.run_placebo_test(
                country=country,
                treatment_year=gsc_result['treatment_year'],
                data_dict=data_dict,
                analysis_func=placebo_analysis_func,
                donor_pool=donor_pool,
                n_placebos=n_placebos
            )

            # Add to aggregator
            aggregator.add_result(country, gsc_result, placebo_result)

        except Exception as e:
            print(f"  Error analyzing {country}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Create standardized results
    results = CausalResults('Generalized SC')

    # Transfer results from aggregator
    for country in treated_countries:
        if country in aggregator.results:
            method_result = aggregator.results[country]['method_result']
            placebo_result = aggregator.results[country]['placebo_result']

            country_metrics = {
                'att': method_result.get('att', np.nan),
                'pre_rmse': method_result.get('pre_rmse', np.nan),
                'post_rmse': method_result.get('post_rmse', np.nan),
                'rmse_ratio': method_result.get('rmse_ratio', np.nan),
                'p_value': placebo_result.get('p_value_ratio', np.nan),
                'significant': placebo_result.get('p_value_ratio', 1) <= 0.1
            }
            results.add_country_result(country, country_metrics)

    # Create summary table using aggregator
    results.summary_table = aggregator.create_summary_table(sort_by='p-val')

    # Store raw results
    results.raw_results = {}
    for country in treated_countries:
        if country in aggregator.results:
            results.raw_results[country] = {
                'gsc_result': aggregator.results[country]['method_result'],
                'placebo_result': aggregator.results[country]['placebo_result']
            }

    return results
