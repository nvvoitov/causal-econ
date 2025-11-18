"""
Difference-in-Differences with Propensity Score Matching implementation.

This module provides a simplified interface to DiD analysis by leveraging
the helper classes and pipeline framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ..core.base import CausalResults
from ..core.placebo import PlaceboTestRunner
from ..core.results import ResultsAggregator
from .helpers import DifferenceInDifferencesHelper


# Backward compatibility: expose PSM calculation
def calculate_propensity_scores(data_dict: Dict, treated_country: str, donor_pool: List[str],
                               feature_cols: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate propensity scores for potential matches using logistic regression.

    DEPRECATED: Now uses DifferenceInDifferencesHelper._calculate_propensity_scores().
    Kept for backward compatibility.
    """
    helper = DifferenceInDifferencesHelper()
    return helper._calculate_propensity_scores(data_dict, treated_country, donor_pool, feature_cols)


# Backward compatibility: expose match selection
def select_best_match(data_dict: Dict, treated_country: str, donor_pool: List[str],
                     propensity_scores: Optional[Dict[str, float]] = None,
                     feature_cols: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[float], Dict]:
    """
    Select the country with highest propensity score as the match.

    DEPRECATED: Now uses DifferenceInDifferencesHelper._select_best_match().
    Kept for backward compatibility.
    """
    helper = DifferenceInDifferencesHelper()
    return helper._select_best_match(data_dict, treated_country, donor_pool, propensity_scores, feature_cols)


# Backward compatibility: expose parallel shift calculation
def calculate_parallel_shift(data_dict: Dict, treated_country: str, synth_match: str,
                            treatment_year: Optional[int] = None) -> float:
    """
    Calculate parallel shift constant to align pre-treatment trends.

    DEPRECATED: Now uses DifferenceInDifferencesHelper._calculate_parallel_shift().
    Kept for backward compatibility.
    """
    helper = DifferenceInDifferencesHelper()
    return helper._calculate_parallel_shift(treated_country, synth_match, data_dict['panel'],
                                           treatment_year or data_dict['treatment_years'][treated_country])


# Backward compatibility: expose single-country DiD analysis
def run_did_analysis(data_dict: Dict, treated_country: str, synth_match: str) -> Dict:
    """
    Run DiD analysis for a treated country and its synthetic match.

    DEPRECATED: This functionality is now in DifferenceInDifferencesHelper.
    Kept for backward compatibility.
    """
    helper = DifferenceInDifferencesHelper()
    treatment_year = data_dict['treatment_years'][treated_country]
    match_info = {'match': synth_match, 'score': 1.0}  # Score not used in basic call

    result = helper.run_for_country(treated_country, match_info, data_dict)

    if result is None:
        raise ValueError(f"DiD analysis failed for {treated_country}")

    return result


# Backward compatibility: expose placebo testing
def run_placebo_tests_did(data_dict: Dict, treated_country: str, donor_pool: List[str],
                         did_result: Dict, n_placebos: int = 20,
                         feature_cols: Optional[List[str]] = None) -> Dict:
    """
    Run placebo tests using donor countries as placebo treated units.

    REFACTORED: Now uses PlaceboTestRunner for cleaner implementation.
    """
    if did_result is None:
        return {'p_value_ratio': np.nan, 'placebos': []}

    runner = PlaceboTestRunner()
    treatment_year = did_result['treatment_year']

    # Define analysis function for placebos
    def placebo_analysis_func(placebo_country, placebo_donor_pool, placebo_data_dict):
        """Run DiD for placebo country."""
        helper = DifferenceInDifferencesHelper()

        # Calculate propensity scores
        propensity_scores = helper._calculate_propensity_scores(
            placebo_data_dict, placebo_country, placebo_donor_pool, feature_cols
        )

        if not propensity_scores:
            return None

        # Select best match
        best_match, score, _ = helper._select_best_match(
            placebo_data_dict, placebo_country, placebo_donor_pool,
            propensity_scores, feature_cols
        )

        if best_match is None:
            return None

        # Run DiD analysis
        match_info = {'match': best_match, 'score': score}
        return helper.run_for_country(placebo_country, match_info, placebo_data_dict)

    # Run placebo test
    placebo_result = runner.run_placebo_test(
        country=treated_country,
        treatment_year=treatment_year,
        data_dict=data_dict,
        analysis_func=placebo_analysis_func,
        donor_pool=donor_pool,
        n_placebos=n_placebos
    )

    return placebo_result


def run_did_analysis_pipeline(data_dict: Dict, donor_pools: Dict[str, List[str]],
                             n_placebos: int = 20, feature_cols: Optional[List[str]] = None) -> CausalResults:
    """
    Run complete DiD analysis pipeline for all treated countries.

    REFACTORED: Now uses helper classes and aggregator for cleaner implementation.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with preprocessed data
    donor_pools : dict
        Dictionary mapping treated countries to donor pools
    n_placebos : int
        Number of placebo tests to run
    feature_cols : list, optional
        List of feature columns to use for propensity score matching

    Returns:
    --------
    CausalResults : Standardized results object
    """
    # Initialize helper and aggregator
    helper = DifferenceInDifferencesHelper()
    aggregator = ResultsAggregator('Difference-in-Differences')
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
        if len(donor_pool) < 2:
            print(f"  Not enough donors for {country}, skipping.")
            continue

        print(f"  Using {len(donor_pool)} donors")

        try:
            # Calculate propensity scores
            propensity_scores = helper._calculate_propensity_scores(
                data_dict, country, donor_pool, feature_cols
            )

            if not propensity_scores:
                print(f"  Failed to calculate propensity scores for {country}, skipping.")
                continue

            # Select best match
            best_match, score, _ = helper._select_best_match(
                data_dict, country, donor_pool, propensity_scores, feature_cols
            )

            if best_match is None:
                print(f"  Could not find suitable match for {country}, skipping.")
                continue

            print(f"  Best match: {best_match} (score: {score:.4f})")

            if best_match not in panel.columns:
                print(f"  Match {best_match} not found in panel data, skipping.")
                continue

            # Run DiD analysis
            match_info = {'match': best_match, 'score': score}
            did_result = helper.run_for_country(country, match_info, data_dict)

            if did_result is None:
                print(f"  Failed to run DiD for {country}, skipping.")
                continue

            # Run placebo tests
            def placebo_analysis_func(placebo_country, placebo_donor_pool, placebo_data_dict):
                """Run DiD for placebo country."""
                # Calculate propensity scores for placebo
                placebo_scores = helper._calculate_propensity_scores(
                    placebo_data_dict, placebo_country, placebo_donor_pool, feature_cols
                )
                if not placebo_scores:
                    return None

                # Select best match for placebo
                placebo_match, _, _ = helper._select_best_match(
                    placebo_data_dict, placebo_country, placebo_donor_pool,
                    placebo_scores, feature_cols
                )
                if placebo_match is None:
                    return None

                # Run DiD for placebo
                placebo_match_info = {'match': placebo_match, 'score': placebo_scores.get(placebo_match, 0)}
                return helper.run_for_country(placebo_country, placebo_match_info, placebo_data_dict)

            placebo_result = placebo_runner.run_placebo_test(
                country=country,
                treatment_year=did_result['treatment_year'],
                data_dict=data_dict,
                analysis_func=placebo_analysis_func,
                donor_pool=donor_pool,
                n_placebos=n_placebos
            )

            # Add to aggregator with match info
            aggregator.add_result(country, did_result, placebo_result)

            # Store match info separately
            if country in aggregator.results:
                aggregator.results[country]['match_info'] = {
                    'best_match': best_match,
                    'score': score,
                    'propensity_scores': propensity_scores
                }

        except Exception as e:
            print(f"  Error analyzing {country}: {str(e)}")
            continue

    # Create standardized results
    results = CausalResults('Difference-in-Differences')

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

    # Store raw results with match info
    results.raw_results = {}
    for country in treated_countries:
        if country in aggregator.results:
            results.raw_results[country] = {
                'did_result': aggregator.results[country]['method_result'],
                'placebo_result': aggregator.results[country]['placebo_result'],
                'match_info': aggregator.results[country].get('match_info', {})
            }

    return results


def create_match_table(results: CausalResults) -> pd.DataFrame:
    """
    Create table showing propensity scores and selected matches.

    Parameters:
    -----------
    results : CausalResults
        Results object from run_did_analysis_pipeline

    Returns:
    --------
    DataFrame : Match table with propensity scores and selections
    """
    match_data = []

    for country, country_data in results.raw_results.items():
        did_result = country_data.get('did_result')
        match_info = country_data.get('match_info', {})

        if did_result:
            match_data.append({
                'treated_country': country,
                'synth_match': did_result.get('synth_match'),
                'propensity_score': match_info.get('score', np.nan),
                'treatment_year': did_result.get('treatment_year'),
                'parallel_shift': did_result.get('shift')
            })

    match_df = pd.DataFrame(match_data)
    if not match_df.empty:
        match_df = match_df.sort_values('propensity_score', ascending=False)

    return match_df
