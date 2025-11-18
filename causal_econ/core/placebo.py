"""
Unified placebo testing framework for all causal inference methods.

This module provides a generic PlaceboTestRunner class that eliminates
~450 lines of duplicate placebo testing code across all causal methods.
"""

from typing import Dict, List, Callable, Optional, Any
import numpy as np
import pandas as pd


class PlaceboTestRunner:
    """
    Unified placebo testing framework for all causal methods.

    This class provides a standardized approach to placebo testing that works
    across Synthetic Control, DiD, SDiD, and GSC methods. It eliminates code
    duplication by abstracting the common placebo testing logic.

    Attributes:
    -----------
    data_dict : dict
        Dictionary containing panel data and metadata
    n_placebos : int
        Maximum number of placebo tests to run
    panel : pd.DataFrame
        Panel data matrix (years x countries)
    treated_countries : list
        List of countries receiving treatment

    Example:
    --------
    >>> runner = PlaceboTestRunner(data_dict, n_placebos=20)
    >>> placebo_result = runner.run_placebo_test(
    ...     country='USA',
    ...     result=sc_result,
    ...     donor_pool=['Canada', 'UK', 'Germany'],
    ...     analysis_func=run_synthetic_control
    ... )
    """

    def __init__(self, data_dict: Dict, n_placebos: int = 20):
        """
        Initialize the placebo test runner.

        Parameters:
        -----------
        data_dict : dict
            Dictionary with keys:
                - 'panel': DataFrame with countries as columns, years as index
                - 'treated_countries': List of treated country names
                - 'treatment_years': Dict mapping countries to treatment years
        n_placebos : int, optional
            Maximum number of placebo tests to run (default: 20)
        """
        self.data_dict = data_dict
        self.n_placebos = n_placebos
        self.panel = data_dict['panel']
        self.treated_countries = data_dict['treated_countries']
        self.treatment_years = data_dict['treatment_years']

    def run_placebo_test(self,
                        country: str,
                        result: Dict,
                        donor_pool: List[str],
                        analysis_func: Callable,
                        metric_name: str = 'rmse_ratio',
                        use_weighted_donors: bool = True,
                        custom_placebo_selector: Optional[Callable] = None) -> Dict:
        """
        Run placebo test for a given country using specified analysis method.

        This method implements the standard placebo testing procedure:
        1. Select placebo countries from the donor pool
        2. For each placebo, temporarily assign it as "treated"
        3. Run the same analysis treating placebo as if it were treated
        4. Calculate p-value by comparing true result to placebo distribution

        Parameters:
        -----------
        country : str
            Name of the actually treated country
        result : dict
            Analysis results for the treated country, must contain:
                - 'treatment_year': Treatment year
                - metric_name: The metric to compare (e.g., 'rmse_ratio')
                - Optional: 'weights' dict for weighted donor selection
        donor_pool : list
            List of potential placebo countries
        analysis_func : callable
            Function that runs the causal method with signature:
            func(country, donor_pool, data_dict) -> dict
        metric_name : str, optional
            Name of metric to use for p-value calculation (default: 'rmse_ratio')
        use_weighted_donors : bool, optional
            If True and result contains 'weights', prioritize weighted donors
            (default: True)
        custom_placebo_selector : callable, optional
            Custom function to select placebo countries with signature:
            func(donor_pool, result, n_placebos) -> List[str]

        Returns:
        --------
        dict : Placebo test results containing:
            - 'p_value_ratio': P-value from placebo distribution
            - 'placebos': List of placebo result dictionaries
            - 'placebo_countries': List of placebo country names

        Notes:
        ------
        - The p-value represents the proportion of placebos with a metric value
          greater than or equal to the treated unit's metric value
        - A low p-value (e.g., < 0.1) suggests the treatment effect is significant
        - If no valid placebo results are obtained, returns NaN p-value
        """
        if result is None:
            return {'p_value_ratio': np.nan, 'placebos': [], 'placebo_countries': []}

        treatment_year = result['treatment_year']

        # Select placebo countries
        if custom_placebo_selector:
            placebo_countries = custom_placebo_selector(donor_pool, result, self.n_placebos)
        else:
            placebo_countries = self._select_placebo_countries(
                donor_pool,
                result.get('weights', {}) if use_weighted_donors else {},
                result.get('unit_weights', {}) if use_weighted_donors else {}
            )

        print(f"Using {len(placebo_countries)} placebo countries for {country}")

        # Run placebo tests
        placebo_results = []
        for placebo_country in placebo_countries:
            placebo_result = self._run_single_placebo(
                placebo_country=placebo_country,
                treated_country=country,
                donor_pool=donor_pool,
                treatment_year=treatment_year,
                analysis_func=analysis_func,
                metric_name=metric_name
            )

            if placebo_result is not None:
                placebo_results.append(placebo_result)

        # Calculate p-values
        return self._calculate_p_values(result, placebo_results, metric_name)

    def _select_placebo_countries(self,
                                  donor_pool: List[str],
                                  weights: Dict[str, float],
                                  unit_weights: Dict[str, float]) -> List[str]:
        """
        Select placebo countries, prioritizing weighted donors if available.

        Strategy:
        1. If weights exist, use donors with weight > 0.01 first
        2. Fill remaining slots from donor pool
        3. Exclude treated countries

        Parameters:
        -----------
        donor_pool : list
            Full donor pool
        weights : dict
            Weights from SC method (if applicable)
        unit_weights : dict
            Unit weights from SDiD method (if applicable)

        Returns:
        --------
        list : Selected placebo country names
        """
        # Combine weights from different sources
        combined_weights = {**weights, **unit_weights}

        if combined_weights:
            # Use weighted donors first (weight > 0.01 threshold)
            weighted_donors = [d for d, w in combined_weights.items() if w > 0.01]
            placebo_countries = [d for d in weighted_donors
                               if d not in self.treated_countries][:self.n_placebos]

            # Add more from donor pool if needed
            if len(placebo_countries) < self.n_placebos:
                additional = [d for d in donor_pool
                            if d not in placebo_countries
                            and d not in self.treated_countries]
                placebo_countries.extend(
                    additional[:self.n_placebos - len(placebo_countries)]
                )
        else:
            # No weights available, use donor pool directly
            placebo_countries = [d for d in donor_pool
                               if d not in self.treated_countries][:self.n_placebos]

        return placebo_countries

    def _run_single_placebo(self,
                           placebo_country: str,
                           treated_country: str,
                           donor_pool: List[str],
                           treatment_year: int,
                           analysis_func: Callable,
                           metric_name: str) -> Optional[Dict]:
        """
        Run analysis on a single placebo country.

        Parameters:
        -----------
        placebo_country : str
            Country to treat as placebo
        treated_country : str
            Actually treated country (excluded from placebo's donor pool)
        donor_pool : list
            Original donor pool
        treatment_year : int
            Treatment year to assign to placebo
        analysis_func : callable
            Analysis function to run
        metric_name : str
            Metric name to validate in result

        Returns:
        --------
        dict or None : Placebo result if successful, None otherwise
        """
        # Create donor pool for placebo (exclude treated and placebo itself)
        placebo_donors = self._get_placebo_donors(
            placebo_country, treated_country, donor_pool
        )

        if len(placebo_donors) < 2:
            return None

        try:
            # Create temporary data dict with placebo as treated
            temp_data_dict = self._create_placebo_data_dict(
                placebo_country, treatment_year
            )

            # Run analysis on placebo
            placebo_result = analysis_func(
                placebo_country,
                placebo_donors,
                temp_data_dict
            )

            # Validate result
            if placebo_result and not np.isnan(placebo_result.get(metric_name, np.nan)):
                return placebo_result

        except Exception as e:
            # Silently continue on errors (expected for some placebos)
            pass

        return None

    def _get_placebo_donors(self,
                           placebo_country: str,
                           treated_country: str,
                           donor_pool: List[str]) -> List[str]:
        """
        Get donor pool for a placebo country.

        Excludes:
        - The placebo country itself
        - The actually treated country
        - All other treated countries

        Parameters:
        -----------
        placebo_country : str
            Country being used as placebo
        treated_country : str
            Actually treated country
        donor_pool : list
            Original donor pool

        Returns:
        --------
        list : Donor pool for placebo analysis
        """
        return [d for d in donor_pool
                if d != placebo_country
                and d != treated_country
                and d not in self.treated_countries]

    def _create_placebo_data_dict(self,
                                  placebo_country: str,
                                  treatment_year: int) -> Dict:
        """
        Create modified data dictionary for placebo testing.

        Creates a shallow copy of data_dict with modified treatment_years
        that includes the placebo country.

        Parameters:
        -----------
        placebo_country : str
            Country to mark as treated
        treatment_year : int
            Treatment year to assign

        Returns:
        --------
        dict : Modified data dictionary
        """
        temp_data_dict = self.data_dict.copy()
        temp_treatment_years = self.treatment_years.copy()
        temp_treatment_years[placebo_country] = treatment_year
        temp_data_dict['treatment_years'] = temp_treatment_years
        return temp_data_dict

    def _calculate_p_values(self,
                           result: Dict,
                           placebo_results: List[Dict],
                           metric_name: str) -> Dict:
        """
        Calculate p-values from placebo distribution.

        The p-value is calculated as the proportion of placebo results with
        a metric value greater than or equal to the treated unit's value.

        Formula: p = (# placebos with metric >= treated metric) / (# placebos)

        Parameters:
        -----------
        result : dict
            Result for treated country
        placebo_results : list
            List of placebo result dictionaries
        metric_name : str
            Name of metric to compare

        Returns:
        --------
        dict : Dictionary containing:
            - 'p_value_ratio': Calculated p-value (or NaN if invalid)
            - 'placebos': List of placebo results
            - 'placebo_countries': List of placebo country names
        """
        if not placebo_results:
            return {
                'p_value_ratio': np.nan,
                'placebos': [],
                'placebo_countries': []
            }

        metric_value = result.get(metric_name)

        if metric_value is None or np.isnan(metric_value):
            p_value = np.nan
        else:
            # Count placebos with metric >= treated metric
            count = sum(1 for p in placebo_results
                       if p.get(metric_name) is not None
                       and p[metric_name] >= metric_value)
            p_value = count / len(placebo_results)

        return {
            'p_value_ratio': p_value,
            'placebos': placebo_results,
            'placebo_countries': [p['country'] for p in placebo_results]
        }

    def run_placebo_test_with_gemini(self,
                                    country: str,
                                    result: Dict,
                                    analysis_func: Callable,
                                    metric_name: str = 'rmse_ratio') -> Dict:
        """
        Run placebo test using gemini-based donor selection.

        This is a specialized version for methods that use gemini similarity
        data to select country-specific donor pools (like the original SC implementation).

        Parameters:
        -----------
        country : str
            Treated country name
        result : dict
            Analysis results
        analysis_func : callable
            Analysis function
        metric_name : str
            Metric for p-value calculation

        Returns:
        --------
        dict : Placebo test results
        """
        if result is None:
            return {'p_value_ratio': np.nan, 'placebos': [], 'placebo_countries': []}

        treatment_year = result['treatment_year']
        donor_pool = result.get('donor_pool', [])
        gemini_df = self.data_dict.get('gemini_df')

        # Get weighted donors as placebo candidates
        weighted_donors = [donor for donor, weight in result.get('weights', {}).items()
                         if weight > 0.01]

        placebo_countries = [d for d in weighted_donors
                           if d not in self.treated_countries][:self.n_placebos]

        # Fill with additional donors if needed
        if len(placebo_countries) < min(self.n_placebos, len(donor_pool)):
            additional = [d for d in donor_pool
                        if d not in placebo_countries
                        and d not in self.treated_countries]
            placebo_countries.extend(additional[:self.n_placebos - len(placebo_countries)])

        print(f"Using {len(placebo_countries)} placebo countries for {country}")

        # Run placebos with gemini-based donor selection
        placebo_results = []

        if gemini_df is not None:
            placebo_results = self._run_gemini_placebos(
                placebo_countries=placebo_countries,
                treatment_year=treatment_year,
                gemini_df=gemini_df,
                analysis_func=analysis_func,
                metric_name=metric_name
            )

        return self._calculate_p_values(result, placebo_results, metric_name)

    def _run_gemini_placebos(self,
                            placebo_countries: List[str],
                            treatment_year: int,
                            gemini_df: pd.DataFrame,
                            analysis_func: Callable,
                            metric_name: str) -> List[Dict]:
        """
        Run placebo tests using gemini similarity for donor selection.

        Parameters:
        -----------
        placebo_countries : list
            Countries to use as placebos
        treatment_year : int
            Treatment year
        gemini_df : pd.DataFrame
            Gemini similarity data
        analysis_func : callable
            Analysis function
        metric_name : str
            Metric name

        Returns:
        --------
        list : Placebo results
        """
        placebo_results = []

        # Check gemini data format
        is_top_geminis = 'Gemini' in gemini_df.columns and 'Cluster' not in gemini_df.columns

        # Country name mapping
        country_map = {'Korea, Rep.': 'Korea', 'Russian Federation': 'Russia'}
        reverse_map = {v: k for k, v in country_map.items()}

        for placebo_country in placebo_countries:
            # Get placebo's own donor pool from gemini
            placebo_gemini_country = country_map.get(placebo_country, placebo_country)

            # Get donors based on gemini format
            if is_top_geminis:
                gemini_matches = gemini_df[gemini_df['Country'] == placebo_gemini_country]['Gemini'].tolist()
                placebo_donors = [reverse_map.get(g, g) for g in gemini_matches
                                if reverse_map.get(g, g) in self.panel.columns]
            else:
                try:
                    row = gemini_df[gemini_df['Country'] == placebo_gemini_country]
                    if row.empty:
                        continue

                    cluster = row['Cluster'].values[0]
                    cluster_countries = gemini_df[gemini_df['Cluster'] == cluster]['Country'].tolist()
                    placebo_donors = [reverse_map.get(c, c) for c in cluster_countries
                                    if c != placebo_gemini_country
                                    and reverse_map.get(c, c) in self.panel.columns]
                except Exception:
                    continue

            # Remove treated countries and placebo itself
            placebo_donors = [d for d in placebo_donors
                            if d != placebo_country
                            and d not in self.treated_countries]

            if len(placebo_donors) < 2:
                continue

            # Run placebo analysis
            placebo_result = self._run_single_placebo(
                placebo_country=placebo_country,
                treated_country='',  # Not used in gemini case
                donor_pool=placebo_donors,
                treatment_year=treatment_year,
                analysis_func=analysis_func,
                metric_name=metric_name
            )

            if placebo_result:
                placebo_results.append(placebo_result)

        return placebo_results
