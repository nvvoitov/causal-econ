"""
Synthetic Control Method implementation.

This module provides a simplified interface to Synthetic Control analysis
by leveraging the helper classes and pipeline framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from ..core.base import CausalResults
from ..core.pipeline import CausalAnalysisPipeline
from ..core.placebo import PlaceboTestRunner
from ..core.results import ResultsAggregator
from ..core.optimization import WeightOptimizer
from .helpers import SyntheticControlHelper


# Backward compatibility: expose weight optimization function
def get_sc_weights(treated: str, donors: List[str], panel: pd.DataFrame,
                   pre_treatment_years: List[int]):
    """
    Calculate synthetic control weights using optimization.

    DEPRECATED: Use WeightOptimizer.optimize_sc_weights() instead.
    Kept for backward compatibility.
    """
    optimizer = WeightOptimizer()
    weights, pre_rmse = optimizer.optimize_sc_weights(
        treated, donors, panel, pre_treatment_years
    )
    return weights, pre_rmse


# Backward compatibility: expose single-country analysis
def run_synthetic_control(country: str, donor_pool: List[str], data_dict: Dict) -> Optional[Dict]:
    """
    Run synthetic control analysis for a single country.

    DEPRECATED: Use SyntheticControlHelper.run_for_country() instead.
    Kept for backward compatibility.
    """
    helper = SyntheticControlHelper()
    return helper.run_for_country(country, donor_pool, data_dict)


# Backward compatibility: expose placebo testing
def run_placebo_test_sc(country: str, sc_result: Dict, data_dict: Dict,
                       n_placebos: int = 20) -> Dict:
    """
    Run placebo tests for synthetic control.

    DEPRECATED: Use PlaceboTestRunner.run_placebo_test() instead.
    Kept for backward compatibility.
    """
    if sc_result is None:
        return {'p_value_ratio': np.nan, 'placebos': []}

    runner = PlaceboTestRunner()

    # Extract required info from sc_result
    weighted_donors = [donor for donor, weight in sc_result['weights'].items()
                     if weight > 0.01]

    placebo_result = runner.run_placebo_test_with_gemini(
        country=country,
        treatment_year=sc_result['treatment_year'],
        data_dict=data_dict,
        analysis_func=run_synthetic_control,
        donor_pool=sc_result['donor_pool'],
        weighted_donors=weighted_donors,
        n_placebos=n_placebos
    )

    return placebo_result


def run_sc_analysis(data_dict: Dict, donor_pools: Dict[str, List[str]],
                   n_placebos: int = 20) -> CausalResults:
    """
    Run synthetic control analysis for all treated countries.

    This is the main entry point for SC analysis. It now uses the pipeline
    framework for cleaner, more maintainable code.

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
    # Initialize helper and aggregator
    helper = SyntheticControlHelper()
    aggregator = ResultsAggregator('Synthetic Control')
    helper.aggregator = aggregator

    # Initialize placebo runner
    placebo_runner = PlaceboTestRunner()

    treated_countries = data_dict['treated_countries']

    # Process each country
    for country in treated_countries:
        print(f"Analyzing country: {country}")

        if country not in donor_pools or not donor_pools[country]:
            print(f"  No donor pool for {country}, skipping.")
            continue

        donor_pool = donor_pools[country]
        print(f"  Using {len(donor_pool)} donors")

        try:
            # Run SC analysis
            sc_result = helper.run_for_country(country, donor_pool, data_dict)

            if sc_result is None:
                print(f"  Failed to run SC for {country}, skipping.")
                continue

            # Run placebo test
            weighted_donors = [donor for donor, weight in sc_result['weights'].items()
                             if weight > 0.01]

            placebo_result = placebo_runner.run_placebo_test_with_gemini(
                country=country,
                treatment_year=sc_result['treatment_year'],
                data_dict=data_dict,
                analysis_func=lambda c, dp, dd: helper.run_for_country(c, dp, dd),
                donor_pool=donor_pool,
                weighted_donors=weighted_donors,
                n_placebos=n_placebos
            )

            # Add to aggregator
            aggregator.add_result(country, sc_result, placebo_result)

        except Exception as e:
            print(f"  Error analyzing {country}: {e}")
            continue

    # Create standardized results
    results = CausalResults('Synthetic Control')

    # Transfer results from aggregator
    for country in treated_countries:
        if country in aggregator.results:
            method_result = aggregator.results[country]['method_result']
            placebo_result = aggregator.results[country]['placebo_result']

            country_metrics = {
                'att': method_result['att'],
                'pre_rmse': method_result['pre_rmse'],
                'post_rmse': method_result['post_rmse'],
                'rmse_ratio': method_result['rmse_ratio'],
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
                'sc_result': aggregator.results[country]['method_result'],
                'placebo_result': aggregator.results[country]['placebo_result']
            }

    return results


def create_weights_table(results: CausalResults, threshold: float = 0.01) -> pd.DataFrame:
    """
    Create consolidated weights dataframe for all countries.

    Parameters:
    -----------
    results : CausalResults
        Results object from run_sc_analysis
    threshold : float
        Minimum weight to include (default: 0.01)

    Returns:
    --------
    DataFrame : Weights table with treated_country, donor_country, weight columns
    """
    # Use ResultsAggregator if we can extract it, otherwise use legacy code
    all_weights = []

    for country, country_data in results.raw_results.items():
        sc_result = country_data.get('sc_result')
        if sc_result and 'weights' in sc_result:
            weights = sc_result['weights']
            for donor, weight in weights.items():
                if weight > threshold:
                    all_weights.append({
                        'treated_country': country,
                        'donor_country': donor,
                        'weight': weight
                    })

    weights_df = pd.DataFrame(all_weights).sort_values(
        ['treated_country', 'weight'], ascending=[True, False]
    )
    return weights_df
