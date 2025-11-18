"""
Synthetic Difference-in-Differences implementation.

This module provides a simplified interface to SDiD analysis by leveraging
the SDiDHelper class and pipeline framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ..core.base import CausalResults
from ..core.placebo import PlaceboTestRunner
from ..core.results import ResultsAggregator
from .helpers import SDiDHelper


# Backward compatibility: expose regularization calculation
def calculate_regularization(data: pd.DataFrame, outcome_col: str, year_col: str,
                           state_col: str, treat_col: str, post_col: str) -> float:
    """
    Calculate the regularization parameter ζ for unit weights according to SDiD methodology.

    DEPRECATED: This is now handled internally by SDiDHelper.
    Kept for backward compatibility.
    """
    helper = SDiDHelper()
    return helper._calculate_regularization(data, outcome_col, year_col, state_col, treat_col, post_col)


# Backward compatibility: expose unit weight calculation
def calculate_unit_weights(data: pd.DataFrame, outcome_col: str, year_col: str,
                         state_col: str, treat_col: str, post_col: str) -> Tuple[float, Dict[str, float]]:
    """
    Calculate unit weights for SDiD following the optimization problem.

    DEPRECATED: This is now handled by SDiDHelper._optimize_weights().
    Kept for backward compatibility.
    """
    helper = SDiDHelper()

    # Create minimal data_dict for helper
    panel = data.pivot(index=year_col, columns=state_col, values=outcome_col)
    treated_units = data[data[treat_col]][state_col].unique().tolist()
    control_units = data[~data[treat_col]][state_col].unique().tolist()

    # For backward compatibility, we'll just use the first treated unit
    if not treated_units:
        return 0.0, {}

    treated_country = treated_units[0]
    treatment_year = data[data[post_col]][year_col].min()

    data_dict = {
        'panel': panel,
        'treatment_years': {treated_country: treatment_year}
    }

    result = helper._optimize_weights(treated_country, control_units, data_dict)

    if result:
        return 0.0, result['unit_weights']  # omega_0 set to 0.0 for simplicity
    else:
        return 0.0, {}


# Backward compatibility: expose time weight calculation
def calculate_time_weights(data: pd.DataFrame, outcome_col: str, year_col: str,
                         state_col: str, treat_col: str, post_col: str,
                         unit_weights: Dict[str, float]) -> Tuple[float, Dict[int, float]]:
    """
    Calculate time weights for SDiD.

    DEPRECATED: This is now handled by SDiDHelper._optimize_weights().
    Kept for backward compatibility.
    """
    # Simplified - actual implementation is in helper
    helper = SDiDHelper()

    panel = data.pivot(index=year_col, columns=state_col, values=outcome_col)
    treated_units = data[data[treat_col]][state_col].unique().tolist()

    if not treated_units:
        return 0.0, {}

    treated_country = treated_units[0]
    treatment_year = data[data[post_col]][year_col].min()
    control_units = list(unit_weights.keys())

    data_dict = {
        'panel': panel,
        'treatment_years': {treated_country: treatment_year}
    }

    result = helper._optimize_weights(treated_country, control_units, data_dict)

    if result:
        return 0.0, result['time_weights']  # lambda_0 set to 0.0 for simplicity
    else:
        return 0.0, {}


# Backward compatibility: expose SDiD estimation
def run_sdid(data: pd.DataFrame, unit_weights: Dict[str, float], time_weights: Dict[int, float],
            outcome_col: str, year_col: str, state_col: str, treat_col: str,
            post_col: str, treatment_year: int) -> Dict:
    """
    Run SDiD estimation with given weights.

    DEPRECATED: Use SDiDHelper.run_for_country() instead.
    Kept for backward compatibility.
    """
    helper = SDiDHelper()

    # Build data_dict from dataframe
    panel = data.pivot(index=year_col, columns=state_col, values=outcome_col)
    treated_units = data[data[treat_col]][state_col].unique().tolist()

    if not treated_units:
        return None

    treated_country = treated_units[0]

    data_dict = {
        'panel': panel,
        'treatment_years': {treated_country: treatment_year}
    }

    control_units = list(unit_weights.keys())
    result = helper.run_for_country(treated_country, control_units, data_dict)

    return result


# Backward compatibility: expose single-country analysis
def run_sdid_analysis(data_dict: Dict, country: str, donor_pool: List[str],
                     regularization: Optional[float] = None) -> Optional[Dict]:
    """
    Run SDiD analysis for a single country.

    REFACTORED: Now uses SDiDHelper.run_for_country().
    """
    helper = SDiDHelper(regularization=regularization)
    return helper.run_for_country(country, donor_pool, data_dict)


# Backward compatibility: expose placebo testing
def run_placebo_test_sdid(country: str, sdid_result: Dict, data_dict: Dict,
                         n_placebos: int = 20, regularization: Optional[float] = None) -> Dict:
    """
    Run placebo tests for SDiD.

    REFACTORED: Now uses PlaceboTestRunner.
    """
    if sdid_result is None:
        return {'p_value_ratio': np.nan, 'placebos': []}

    runner = PlaceboTestRunner()
    treatment_year = sdid_result['treatment_year']

    # Define analysis function for placebos
    def placebo_analysis_func(placebo_country, placebo_donor_pool, placebo_data_dict):
        """Run SDiD for placebo country."""
        helper = SDiDHelper(regularization=regularization)
        return helper.run_for_country(placebo_country, placebo_donor_pool, placebo_data_dict)

    # Use donor pool from original result if available
    donor_pool = sdid_result.get('donor_pool', [])

    placebo_result = runner.run_placebo_test(
        country=country,
        treatment_year=treatment_year,
        data_dict=data_dict,
        analysis_func=placebo_analysis_func,
        donor_pool=donor_pool,
        n_placebos=n_placebos
    )

    return placebo_result


def run_sdid_pipeline(data_dict: Dict, donor_pools: Dict[str, List[str]],
                     n_placebos: int = 20, regularization: Optional[float] = None) -> CausalResults:
    """
    Run complete SDiD analysis pipeline for all treated countries.

    REFACTORED: Now uses helper classes and aggregator for cleaner implementation.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    donor_pools : dict
        Dictionary mapping treated countries to donor pools
    n_placebos : int
        Number of placebo tests to run
    regularization : float, optional
        Regularization parameter (if None, will be calculated automatically)

    Returns:
    --------
    CausalResults : Standardized results object
    """
    # Initialize helper and aggregator
    helper = SDiDHelper(regularization=regularization)
    aggregator = ResultsAggregator('Synthetic Diff-in-Diff')
    helper.aggregator = aggregator

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
            # Run SDiD analysis
            sdid_result = helper.run_for_country(country, donor_pool, data_dict)

            if sdid_result is None:
                print(f"  Failed to run SDiD for {country}, skipping.")
                continue

            # Run placebo tests
            def placebo_analysis_func(placebo_country, placebo_donor_pool, placebo_data_dict):
                """Run SDiD for placebo country."""
                placebo_helper = SDiDHelper(regularization=regularization)
                return placebo_helper.run_for_country(placebo_country, placebo_donor_pool, placebo_data_dict)

            placebo_result = placebo_runner.run_placebo_test(
                country=country,
                treatment_year=sdid_result['treatment_year'],
                data_dict=data_dict,
                analysis_func=placebo_analysis_func,
                donor_pool=donor_pool,
                n_placebos=n_placebos
            )

            # Add to aggregator
            aggregator.add_result(country, sdid_result, placebo_result)

        except Exception as e:
            print(f"  Error analyzing {country}: {str(e)}")
            continue

    # Create standardized results
    results = CausalResults('Synthetic Diff-in-Diff')

    # Transfer results from aggregator
    for country in treated_countries:
        if country in aggregator.results:
            method_result = aggregator.results[country]['method_result']
            placebo_result = aggregator.results[country]['placebo_result']

            country_metrics = {
                'att': method_result['att'],
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
                'sdid_result': aggregator.results[country]['method_result'],
                'placebo_result': aggregator.results[country]['placebo_result']
            }

    return results


def create_weights_tables(results: CausalResults, threshold: float = 0.01) -> Dict[str, pd.DataFrame]:
    """
    Create tables showing unit and time weights for all countries.

    Parameters:
    -----------
    results : CausalResults
        Results object from run_sdid_pipeline
    threshold : float
        Minimum weight to include (default: 0.01)

    Returns:
    --------
    dict : Dictionary with 'unit_weights' and 'time_weights' DataFrames
    """
    unit_weights_data = []
    time_weights_data = []

    for country, country_data in results.raw_results.items():
        sdid_result = country_data.get('sdid_result')

        if sdid_result:
            # Extract unit weights
            if 'weights' in sdid_result:
                for donor, weight in sdid_result['weights'].items():
                    if weight > threshold:
                        unit_weights_data.append({
                            'treated_country': country,
                            'donor_country': donor,
                            'weight': weight
                        })

            # Extract time weights
            if 'time_weights' in sdid_result:
                for year, weight in sdid_result['time_weights'].items():
                    if weight > threshold:
                        time_weights_data.append({
                            'treated_country': country,
                            'year': year,
                            'weight': weight
                        })

    # Create DataFrames
    unit_weights_df = pd.DataFrame(unit_weights_data)
    if not unit_weights_df.empty:
        unit_weights_df = unit_weights_df.sort_values(
            ['treated_country', 'weight'], ascending=[True, False]
        )

    time_weights_df = pd.DataFrame(time_weights_data)
    if not time_weights_df.empty:
        time_weights_df = time_weights_df.sort_values(
            ['treated_country', 'year'], ascending=[True, True]
        )

    return {
        'unit_weights': unit_weights_df,
        'time_weights': time_weights_df
    }
