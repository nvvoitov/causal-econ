"""
Result aggregation and formatting utilities for causal inference methods.

This module provides unified result handling that eliminates duplicate code
for creating summary tables, formatting results, and managing outputs.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from .base import CausalResults


class ResultsAggregator:
    """
    Unified result aggregation and formatting for causal methods.

    This class eliminates ~200 lines of duplicate result handling code by
    providing standardized methods for:
    - Building result dictionaries
    - Creating summary tables
    - Formatting weight tables
    - Extracting and organizing metrics

    Example:
    --------
    >>> aggregator = ResultsAggregator(method_name='Synthetic Control')
    >>> aggregator.add_result(country='USA', method_result=sc_result,
    ...                       placebo_result=placebo_result)
    >>> summary_table = aggregator.create_summary_table()
    """

    def __init__(self, method_name: str):
        """
        Initialize the results aggregator.

        Parameters:
        -----------
        method_name : str
            Name of the causal method (e.g., 'Synthetic Control', 'DiD')
        """
        self.method_name = method_name
        self.results = CausalResults(method_name)
        self.raw_results = {}

    def add_result(self,
                   country: str,
                   method_result: Dict,
                   placebo_result: Optional[Dict] = None,
                   additional_data: Optional[Dict] = None) -> None:
        """
        Add results for a single country.

        Parameters:
        -----------
        country : str
            Country name
        method_result : dict
            Results from the causal method analysis
        placebo_result : dict, optional
            Results from placebo testing
        additional_data : dict, optional
            Any additional method-specific data to store

        Example:
        --------
        >>> aggregator.add_result(
        ...     country='USA',
        ...     method_result={'att': 1.5, 'rmse_ratio': 1.2, ...},
        ...     placebo_result={'p_value_ratio': 0.05, ...}
        ... )
        """
        # Store raw results
        self.raw_results[country] = {
            'method_result': method_result,
            'placebo_result': placebo_result or {},
            'additional_data': additional_data or {}
        }

        # Add standardized metrics to CausalResults object
        country_metrics = self._extract_standard_metrics(
            method_result, placebo_result
        )
        self.results.add_country_result(country, country_metrics)

    def _extract_standard_metrics(self,
                                  method_result: Dict,
                                  placebo_result: Optional[Dict]) -> Dict:
        """
        Extract standardized metrics from method and placebo results.

        Parameters:
        -----------
        method_result : dict
            Method-specific results
        placebo_result : dict, optional
            Placebo test results

        Returns:
        --------
        dict : Standardized metrics
        """
        metrics = {
            'att': method_result.get('att', np.nan),
            'pre_rmse': method_result.get('pre_rmse', np.nan),
            'post_rmse': method_result.get('post_rmse', np.nan),
            'rmse_ratio': method_result.get('rmse_ratio', np.nan)
        }

        # Add placebo results if available
        if placebo_result:
            p_value = placebo_result.get('p_value_ratio', np.nan)
            metrics['p_value'] = p_value
            metrics['significant'] = (
                p_value <= 0.1 if p_value is not None and not np.isnan(p_value) else False
            )
        else:
            metrics['p_value'] = np.nan
            metrics['significant'] = False

        return metrics

    def create_summary_table(self,
                            sort_by: str = 'p-val',
                            ascending: bool = True,
                            significance_threshold: float = 0.1) -> pd.DataFrame:
        """
        Create standardized summary table for all countries.

        This method eliminates the duplicate summary table creation code
        that appears in all 4 causal methods.

        Parameters:
        -----------
        sort_by : str, optional
            Column name to sort by (default: 'p-val')
        ascending : bool, optional
            Sort in ascending order (default: True)
        significance_threshold : float, optional
            Threshold for marking results as significant (default: 0.1)

        Returns:
        --------
        pd.DataFrame : Summary table with columns:
            - country: Country name
            - ATT: Average Treatment Effect on Treated
            - RMSE pre: Pre-treatment RMSE
            - RMSE post: Post-treatment RMSE
            - relation/RMSE ratio: Post/Pre RMSE ratio
            - p-val: P-value from placebo tests
            - significant: 'Yes' if p-value <= threshold

        Example:
        --------
        >>> summary = aggregator.create_summary_table(sort_by='p-val')
        >>> print(summary)
        """
        summary_data = []

        for country, country_data in self.raw_results.items():
            method_result = country_data['method_result']
            placebo_result = country_data['placebo_result']

            p_value = placebo_result.get('p_value_ratio', np.nan) if placebo_result else np.nan

            # Determine if significant
            if p_value is not None and not np.isnan(p_value):
                is_significant = 'Yes' if p_value <= significance_threshold else 'No'
            else:
                is_significant = 'No'

            summary_data.append({
                'country': country,
                'ATT': method_result.get('att', np.nan),
                'RMSE pre': method_result.get('pre_rmse', np.nan),
                'RMSE post': method_result.get('post_rmse', np.nan),
                'relation': method_result.get('rmse_ratio', np.nan),
                'p-val': p_value,
                'significant': is_significant
            })

        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Sort if not empty
        if not summary_df.empty and sort_by in summary_df.columns:
            summary_df = summary_df.sort_values(sort_by, ascending=ascending)

        return summary_df

    def create_weights_table(self,
                            weight_key: str = 'weights',
                            threshold: float = 0.01,
                            sort_by: str = 'weight',
                            ascending: bool = False) -> pd.DataFrame:
        """
        Create consolidated weights table (for SC and SDiD methods).

        Parameters:
        -----------
        weight_key : str, optional
            Key in method_result containing weights dict (default: 'weights')
        threshold : float, optional
            Only include weights above this threshold (default: 0.01)
        sort_by : str, optional
            Column to sort by (default: 'weight')
        ascending : bool, optional
            Sort in ascending order (default: False)

        Returns:
        --------
        pd.DataFrame : Weights table with columns:
            - treated_country: Country receiving treatment
            - donor_country: Donor country
            - weight: Weight value

        Example:
        --------
        >>> weights_table = aggregator.create_weights_table(threshold=0.05)
        """
        weights_data = []

        for country, country_data in self.raw_results.items():
            method_result = country_data['method_result']

            if weight_key in method_result:
                weights = method_result[weight_key]

                if isinstance(weights, dict):
                    for donor, weight in weights.items():
                        if weight > threshold:
                            weights_data.append({
                                'treated_country': country,
                                'donor_country': donor,
                                'weight': weight
                            })

        # Create DataFrame
        weights_df = pd.DataFrame(weights_data)

        # Sort if not empty
        if not weights_df.empty and sort_by in weights_df.columns:
            weights_df = weights_df.sort_values(
                ['treated_country', sort_by],
                ascending=[True, ascending]
            )

        return weights_df

    def create_time_weights_table(self,
                                  threshold: float = 0.01) -> pd.DataFrame:
        """
        Create time weights table (for SDiD method).

        Parameters:
        -----------
        threshold : float, optional
            Only include weights above this threshold (default: 0.01)

        Returns:
        --------
        pd.DataFrame : Time weights table with columns:
            - treated_country: Country receiving treatment
            - year: Pre-treatment year
            - weight: Time weight value

        Example:
        --------
        >>> time_weights = aggregator.create_time_weights_table()
        """
        time_weights_data = []

        for country, country_data in self.raw_results.items():
            method_result = country_data['method_result']

            if 'time_weights' in method_result:
                time_weights = method_result['time_weights']

                if isinstance(time_weights, dict):
                    for year, weight in time_weights.items():
                        if weight > threshold:
                            time_weights_data.append({
                                'treated_country': country,
                                'year': year,
                                'weight': weight
                            })

        # Create DataFrame
        time_weights_df = pd.DataFrame(time_weights_data)

        # Sort if not empty
        if not time_weights_df.empty:
            time_weights_df = time_weights_df.sort_values(['treated_country', 'year'])

        return time_weights_df

    def create_match_table(self) -> pd.DataFrame:
        """
        Create matching table (for DiD method).

        Returns:
        --------
        pd.DataFrame : Match table with columns:
            - treated_country: Country receiving treatment
            - synth_match: Matched control country
            - propensity_score: Propensity score
            - treatment_year: Treatment year
            - parallel_shift: Parallel shift constant

        Example:
        --------
        >>> match_table = aggregator.create_match_table()
        """
        match_data = []

        for country, country_data in self.raw_results.items():
            method_result = country_data['method_result']
            additional_data = country_data.get('additional_data', {})

            # Get match info (could be in method_result or additional_data)
            match_info = additional_data.get('match_info', {})

            match_data.append({
                'treated_country': country,
                'synth_match': method_result.get('synth_match', np.nan),
                'propensity_score': match_info.get('score', np.nan),
                'treatment_year': method_result.get('treatment_year', np.nan),
                'parallel_shift': method_result.get('shift', np.nan)
            })

        # Create DataFrame
        match_df = pd.DataFrame(match_data)

        # Sort by propensity score if not empty
        if not match_df.empty and 'propensity_score' in match_df.columns:
            match_df = match_df.sort_values('propensity_score', ascending=False)

        return match_df

    def get_results(self) -> CausalResults:
        """
        Get the CausalResults object with all aggregated results.

        Returns:
        --------
        CausalResults : Results object with summary table and raw results
        """
        # Create and attach summary table
        self.results.summary_table = self.create_summary_table()
        self.results.raw_results = self.raw_results

        return self.results

    def get_significant_countries(self,
                                 threshold: float = 0.1) -> List[str]:
        """
        Get list of countries with significant treatment effects.

        Parameters:
        -----------
        threshold : float, optional
            P-value threshold for significance (default: 0.1)

        Returns:
        --------
        list : Country names with p-value <= threshold

        Example:
        --------
        >>> significant = aggregator.get_significant_countries(threshold=0.05)
        >>> print(f"Significant results: {significant}")
        """
        significant = []

        for country, country_data in self.raw_results.items():
            placebo_result = country_data.get('placebo_result', {})
            p_value = placebo_result.get('p_value_ratio')

            if p_value is not None and not np.isnan(p_value) and p_value <= threshold:
                significant.append(country)

        return significant

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all countries.

        Returns:
        --------
        dict : Summary statistics including:
            - n_countries: Number of countries analyzed
            - n_significant: Number with significant effects (p <= 0.1)
            - mean_att: Mean ATT across countries
            - median_att: Median ATT
            - mean_rmse_ratio: Mean RMSE ratio
            - n_good_fit: Number with RMSE ratio < 2

        Example:
        --------
        >>> stats = aggregator.get_summary_statistics()
        >>> print(f"Analyzed {stats['n_countries']} countries")
        """
        if not self.raw_results:
            return {
                'n_countries': 0,
                'n_significant': 0,
                'mean_att': np.nan,
                'median_att': np.nan,
                'mean_rmse_ratio': np.nan,
                'n_good_fit': 0
            }

        att_values = []
        rmse_ratios = []
        n_significant = 0
        n_good_fit = 0

        for country, country_data in self.raw_results.items():
            method_result = country_data['method_result']
            placebo_result = country_data.get('placebo_result', {})

            # ATT
            att = method_result.get('att')
            if att is not None and not np.isnan(att):
                att_values.append(att)

            # RMSE ratio
            rmse_ratio = method_result.get('rmse_ratio')
            if rmse_ratio is not None and not np.isnan(rmse_ratio):
                rmse_ratios.append(rmse_ratio)
                if rmse_ratio < 2:
                    n_good_fit += 1

            # Significance
            p_value = placebo_result.get('p_value_ratio')
            if p_value is not None and not np.isnan(p_value) and p_value <= 0.1:
                n_significant += 1

        return {
            'n_countries': len(self.raw_results),
            'n_significant': n_significant,
            'mean_att': np.mean(att_values) if att_values else np.nan,
            'median_att': np.median(att_values) if att_values else np.nan,
            'mean_rmse_ratio': np.mean(rmse_ratios) if rmse_ratios else np.nan,
            'n_good_fit': n_good_fit
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all results to a dictionary for serialization.

        Returns:
        --------
        dict : All results in dictionary format

        Example:
        --------
        >>> results_dict = aggregator.export_to_dict()
        >>> import json
        >>> json.dump(results_dict, open('results.json', 'w'))
        """
        return {
            'method_name': self.method_name,
            'summary_table': self.create_summary_table().to_dict('records'),
            'raw_results': self.raw_results,
            'summary_statistics': self.get_summary_statistics()
        }

    @staticmethod
    def build_method_result(country: str,
                           treatment_year: int,
                           effect: Dict[int, float],
                           **kwargs) -> Dict:
        """
        Build a standardized method result dictionary.

        This helper eliminates duplicate result dictionary construction.

        Parameters:
        -----------
        country : str
            Country name
        treatment_year : int
            Treatment year
        effect : dict
            Dictionary mapping years to effect values
        **kwargs
            Additional method-specific fields

        Returns:
        --------
        dict : Standardized result dictionary

        Example:
        --------
        >>> result = ResultsAggregator.build_method_result(
        ...     country='USA',
        ...     treatment_year=2002,
        ...     effect={2000: 0.1, 2001: 0.2, 2002: 1.5},
        ...     weights={'Canada': 0.7, 'UK': 0.3}
        ... )
        """
        # Import here to avoid circular import
        from .base import calculate_causal_metrics

        # Calculate standard metrics from effect
        metrics = calculate_causal_metrics(effect, treatment_year)

        # Build result dictionary
        result = {
            'country': country,
            'treatment_year': treatment_year,
            'effect': effect,
            'att': metrics['att'],
            'pre_rmse': metrics['pre_rmse'],
            'post_rmse': metrics['post_rmse'],
            'rmse_ratio': metrics['rmse_ratio']
        }

        # Add any additional method-specific fields
        result.update(kwargs)

        return result
