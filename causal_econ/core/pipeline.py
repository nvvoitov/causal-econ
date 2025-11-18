"""
Unified pipeline for running causal analysis methods.

This module provides a generic analysis pipeline that eliminates ~400 lines
of duplicate code across all causal methods by providing a standardized
workflow for running analyses, placebo tests, and aggregating results.
"""

from typing import Dict, List, Callable, Optional, Any
import numpy as np
import pandas as pd
from .base import CausalResults, DataValidator
from .placebo import PlaceboTestRunner
from .results import ResultsAggregator


class CausalAnalysisPipeline:
    """
    Generic pipeline for running causal analysis methods.

    This class eliminates ~400 lines of duplicate pipeline code by providing
    a unified workflow that works across all causal methods:
    - Synthetic Control
    - Difference-in-Differences
    - Synthetic Difference-in-Differences
    - Generalized Synthetic Control

    The pipeline handles:
    - Country iteration and validation
    - Method analysis execution
    - Placebo testing
    - Result aggregation
    - Summary table creation
    - Error handling and logging

    Example:
    --------
    >>> pipeline = CausalAnalysisPipeline(
    ...     method_name='Synthetic Control',
    ...     data_dict=data_dict,
    ...     donor_pools=donor_pools
    ... )
    >>> results = pipeline.run(
    ...     analysis_func=run_synthetic_control,
    ...     n_placebos=20
    ... )
    """

    def __init__(self,
                 method_name: str,
                 data_dict: Dict,
                 donor_pools: Dict[str, List[str]]):
        """
        Initialize the causal analysis pipeline.

        Parameters:
        -----------
        method_name : str
            Name of the causal method (e.g., 'Synthetic Control')
        data_dict : dict
            Dictionary containing:
                - 'panel': DataFrame with countries as columns, years as index
                - 'treated_countries': List of treated country names
                - 'treatment_years': Dict mapping countries to treatment years
                - Optional: 'df', 'gemini_df', etc.
        donor_pools : dict
            Dictionary mapping treated country names to lists of donor countries

        Example:
        --------
        >>> pipeline = CausalAnalysisPipeline(
        ...     method_name='Synthetic Control',
        ...     data_dict=data_dict,
        ...     donor_pools={'USA': ['Canada', 'UK', 'Germany']}
        ... )
        """
        self.method_name = method_name
        self.data_dict = data_dict
        self.donor_pools = donor_pools

        # Extract commonly used data
        self.treated_countries = data_dict['treated_countries']
        self.panel = data_dict['panel']
        self.treatment_years = data_dict['treatment_years']

        # Initialize helper objects
        self.validator = DataValidator(data_dict)
        self.aggregator = ResultsAggregator(method_name)

    def run(self,
            analysis_func: Callable,
            n_placebos: int = 20,
            placebo_func: Optional[Callable] = None,
            use_gemini_placebo: bool = False,
            pre_analysis_hook: Optional[Callable] = None,
            post_analysis_hook: Optional[Callable] = None,
            validation_func: Optional[Callable] = None,
            min_donors: int = 2,
            verbose: bool = True) -> CausalResults:
        """
        Run the complete analysis pipeline for all treated countries.

        This is the main entry point that orchestrates the entire analysis.

        Parameters:
        -----------
        analysis_func : callable
            Function that runs the method analysis with signature:
            func(country, donor_pool, data_dict) -> dict

            The returned dict must contain at minimum:
            - 'country': Country name
            - 'treatment_year': Treatment year
            - 'att': Average Treatment Effect on Treated
            - 'pre_rmse': Pre-treatment RMSE
            - 'post_rmse': Post-treatment RMSE
            - 'rmse_ratio': RMSE ratio

        n_placebos : int, optional
            Number of placebo tests to run (default: 20)

        placebo_func : callable, optional
            Custom placebo testing function with signature:
            func(country, result, data_dict, n_placebos) -> dict
            If None, uses PlaceboTestRunner (default: None)

        use_gemini_placebo : bool, optional
            Use gemini-based placebo testing (default: False)

        pre_analysis_hook : callable, optional
            Function called before analyzing each country with signature:
            func(country, donor_pool, data_dict) -> None
            Use for logging or data preparation (default: None)

        post_analysis_hook : callable, optional
            Function called after analyzing each country with signature:
            func(country, method_result, placebo_result) -> None
            Use for custom result processing (default: None)

        validation_func : callable, optional
            Additional validation function with signature:
            func(country, donor_pool, data_dict) -> (bool, str)
            Returns (is_valid, error_message) (default: None)

        min_donors : int, optional
            Minimum number of donors required (default: 2)

        verbose : bool, optional
            Print progress messages (default: True)

        Returns:
        --------
        CausalResults : Standardized results object containing:
            - summary_table: DataFrame with all countries' results
            - raw_results: Dict with detailed results for each country
            - country_results: Dict with standardized metrics
            - placebo_results: Dict with placebo test outcomes

        Example:
        --------
        >>> def my_analysis(country, donors, data):
        ...     # Your method implementation
        ...     return {'country': country, 'att': 1.5, ...}
        >>>
        >>> results = pipeline.run(
        ...     analysis_func=my_analysis,
        ...     n_placebos=20,
        ...     verbose=True
        ... )
        >>> print(results.summary_table)
        """
        # Initialize placebo test runner
        placebo_runner = PlaceboTestRunner(self.data_dict, n_placebos)

        # Track progress
        n_success = 0
        n_failed = 0
        n_skipped = 0

        # Iterate over treated countries
        for country in self.treated_countries:
            if verbose:
                print(f"Analyzing country: {country}")

            # Validate country
            is_valid, skip_reason = self._validate_country(
                country, validation_func, min_donors, verbose
            )

            if not is_valid:
                if verbose:
                    print(f"  {skip_reason}")
                n_skipped += 1
                continue

            donor_pool = self.donor_pools[country]
            if verbose:
                print(f"  Using {len(donor_pool)} donors")

            # Run pre-analysis hook
            if pre_analysis_hook:
                try:
                    pre_analysis_hook(country, donor_pool, self.data_dict)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Pre-analysis hook failed: {e}")

            # Run analysis
            try:
                method_result = self._run_analysis(
                    country, donor_pool, analysis_func, verbose
                )

                if method_result is None:
                    n_failed += 1
                    continue

                # Run placebo test
                placebo_result = self._run_placebo_test(
                    country, method_result, donor_pool,
                    analysis_func, placebo_func, placebo_runner,
                    use_gemini_placebo, verbose
                )

                # Run post-analysis hook
                if post_analysis_hook:
                    try:
                        post_analysis_hook(country, method_result, placebo_result)
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Post-analysis hook failed: {e}")

                # Store results
                self.aggregator.add_result(
                    country=country,
                    method_result=method_result,
                    placebo_result=placebo_result
                )

                n_success += 1

            except Exception as e:
                if verbose:
                    print(f"  Error analyzing {country}: {e}")
                n_failed += 1
                continue

        # Print summary if verbose
        if verbose:
            print(f"\nPipeline complete:")
            print(f"  Success: {n_success}")
            print(f"  Failed: {n_failed}")
            print(f"  Skipped: {n_skipped}")

        # Return results
        return self.aggregator.get_results()

    def _validate_country(self,
                         country: str,
                         validation_func: Optional[Callable],
                         min_donors: int,
                         verbose: bool) -> tuple:
        """
        Validate country for analysis.

        Parameters:
        -----------
        country : str
            Country name
        validation_func : callable, optional
            Additional validation function
        min_donors : int
            Minimum required donors
        verbose : bool
            Print messages

        Returns:
        --------
        tuple : (is_valid, error_message)
        """
        # Check if donor pool exists
        if country not in self.donor_pools or not self.donor_pools[country]:
            return False, f"No donor pool for {country}, skipping."

        donor_pool = self.donor_pools[country]

        # Use DataValidator for comprehensive validation
        is_valid, error = self.validator.validate_for_analysis(
            country, donor_pool, min_donors=min_donors
        )

        if not is_valid:
            return False, error

        # Run additional validation if provided
        if validation_func:
            try:
                is_valid, error = validation_func(country, donor_pool, self.data_dict)
                if not is_valid:
                    return False, error
            except Exception as e:
                if verbose:
                    print(f"  Warning: Validation function failed: {e}")

        return True, None

    def _run_analysis(self,
                     country: str,
                     donor_pool: List[str],
                     analysis_func: Callable,
                     verbose: bool) -> Optional[Dict]:
        """
        Run the method-specific analysis.

        Parameters:
        -----------
        country : str
            Country name
        donor_pool : list
            List of donors
        analysis_func : callable
            Analysis function
        verbose : bool
            Print messages

        Returns:
        --------
        dict or None : Analysis results
        """
        try:
            result = analysis_func(country, donor_pool, self.data_dict)

            if result is None:
                if verbose:
                    print(f"  Failed to run {self.method_name} for {country}, skipping.")
                return None

            # Validate result has required fields
            required_fields = ['country', 'treatment_year', 'att', 'rmse_ratio']
            missing_fields = [f for f in required_fields if f not in result]

            if missing_fields:
                if verbose:
                    print(f"  Warning: Result missing required fields: {missing_fields}")

            return result

        except Exception as e:
            if verbose:
                print(f"  Error in analysis: {e}")
            return None

    def _run_placebo_test(self,
                         country: str,
                         method_result: Dict,
                         donor_pool: List[str],
                         analysis_func: Callable,
                         placebo_func: Optional[Callable],
                         placebo_runner: PlaceboTestRunner,
                         use_gemini: bool,
                         verbose: bool) -> Dict:
        """
        Run placebo testing.

        Parameters:
        -----------
        country : str
            Country name
        method_result : dict
            Method analysis result
        donor_pool : list
            Donor pool
        analysis_func : callable
            Analysis function
        placebo_func : callable, optional
            Custom placebo function
        placebo_runner : PlaceboTestRunner
            Placebo test runner
        use_gemini : bool
            Use gemini-based placebo
        verbose : bool
            Print messages

        Returns:
        --------
        dict : Placebo test results
        """
        try:
            if placebo_func:
                # Use custom placebo function
                placebo_result = placebo_func(
                    country, method_result, self.data_dict, placebo_runner.n_placebos
                )
            elif use_gemini:
                # Use gemini-based placebo
                placebo_result = placebo_runner.run_placebo_test_with_gemini(
                    country, method_result, analysis_func
                )
            else:
                # Use standard placebo
                placebo_result = placebo_runner.run_placebo_test(
                    country, method_result, donor_pool, analysis_func
                )

            return placebo_result

        except Exception as e:
            if verbose:
                print(f"  Warning: Placebo test failed: {e}")
            return {'p_value_ratio': np.nan, 'placebos': [], 'placebo_countries': []}

    def run_batch(self,
                  countries: List[str],
                  analysis_func: Callable,
                  n_placebos: int = 20,
                  **kwargs) -> CausalResults:
        """
        Run analysis for a specific subset of countries.

        Parameters:
        -----------
        countries : list
            List of country names to analyze
        analysis_func : callable
            Analysis function
        n_placebos : int, optional
            Number of placebo tests (default: 20)
        **kwargs
            Additional arguments passed to run()

        Returns:
        --------
        CausalResults : Results for specified countries

        Example:
        --------
        >>> # Analyze only USA and China
        >>> results = pipeline.run_batch(
        ...     countries=['USA', 'China'],
        ...     analysis_func=run_synthetic_control
        ... )
        """
        # Temporarily filter treated countries
        original_treated = self.treated_countries
        self.treated_countries = [c for c in countries if c in original_treated]

        try:
            results = self.run(analysis_func, n_placebos, **kwargs)
        finally:
            # Restore original treated countries
            self.treated_countries = original_treated

        return results

    def run_parallel_methods(self,
                            method_configs: List[Dict],
                            combine_results: bool = True) -> Union[CausalResults, Dict[str, CausalResults]]:
        """
        Run multiple methods in sequence and optionally combine results.

        This is useful for comparing different causal methods on the same data.

        Parameters:
        -----------
        method_configs : list
            List of dicts, each containing:
                - 'name': Method name
                - 'analysis_func': Analysis function
                - 'n_placebos': Number of placebos (optional)
                - Additional run() parameters

        combine_results : bool, optional
            Combine results into single comparison table (default: True)

        Returns:
        --------
        CausalResults or dict : Combined results or dict of individual results

        Example:
        --------
        >>> configs = [
        ...     {'name': 'SC', 'analysis_func': run_synthetic_control},
        ...     {'name': 'DiD', 'analysis_func': run_did_analysis}
        ... ]
        >>> results = pipeline.run_parallel_methods(configs)
        """
        all_results = {}

        for config in method_configs:
            name = config.pop('name')
            analysis_func = config.pop('analysis_func')

            # Create new pipeline for this method
            method_pipeline = CausalAnalysisPipeline(
                method_name=name,
                data_dict=self.data_dict,
                donor_pools=self.donor_pools
            )

            # Run analysis
            results = method_pipeline.run(analysis_func, **config)
            all_results[name] = results

        if combine_results:
            return self._combine_method_results(all_results)
        else:
            return all_results

    def _combine_method_results(self, all_results: Dict[str, CausalResults]) -> pd.DataFrame:
        """
        Combine results from multiple methods into a comparison table.

        Parameters:
        -----------
        all_results : dict
            Dictionary mapping method names to CausalResults

        Returns:
        --------
        pd.DataFrame : Combined comparison table
        """
        combined_data = []

        for method_name, results in all_results.items():
            if results.summary_table is not None:
                summary = results.summary_table.copy()
                summary['method'] = method_name
                combined_data.append(summary)

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def get_aggregator(self) -> ResultsAggregator:
        """
        Get the ResultsAggregator object for custom result handling.

        Returns:
        --------
        ResultsAggregator : The aggregator object

        Example:
        --------
        >>> aggregator = pipeline.get_aggregator()
        >>> weights_table = aggregator.create_weights_table()
        """
        return self.aggregator

    def get_validator(self) -> DataValidator:
        """
        Get the DataValidator object for custom validation.

        Returns:
        --------
        DataValidator : The validator object

        Example:
        --------
        >>> validator = pipeline.get_validator()
        >>> counts = validator.get_pre_post_data_counts('USA')
        """
        return self.validator


class AnalysisRegistry:
    """
    Registry for storing and managing multiple analysis methods.

    This class allows you to register different analysis functions
    and run them easily without repetitive code.

    Example:
    --------
    >>> registry = AnalysisRegistry()
    >>> registry.register('SC', run_synthetic_control)
    >>> registry.register('DiD', run_did_analysis)
    >>> results = registry.run_all(data_dict, donor_pools)
    """

    def __init__(self):
        """Initialize the analysis registry."""
        self.methods = {}

    def register(self,
                 name: str,
                 analysis_func: Callable,
                 default_n_placebos: int = 20,
                 **default_kwargs):
        """
        Register an analysis method.

        Parameters:
        -----------
        name : str
            Method name
        analysis_func : callable
            Analysis function
        default_n_placebos : int, optional
            Default number of placebos (default: 20)
        **default_kwargs
            Default arguments for the method

        Example:
        --------
        >>> registry.register('SC', run_synthetic_control, n_placebos=20)
        """
        self.methods[name] = {
            'func': analysis_func,
            'n_placebos': default_n_placebos,
            'kwargs': default_kwargs
        }

    def run(self,
            name: str,
            data_dict: Dict,
            donor_pools: Dict[str, List[str]],
            **override_kwargs) -> CausalResults:
        """
        Run a registered method.

        Parameters:
        -----------
        name : str
            Method name
        data_dict : dict
            Data dictionary
        donor_pools : dict
            Donor pools
        **override_kwargs
            Override default kwargs

        Returns:
        --------
        CausalResults : Analysis results

        Example:
        --------
        >>> results = registry.run('SC', data_dict, donor_pools)
        """
        if name not in self.methods:
            raise ValueError(f"Method '{name}' not registered")

        method_config = self.methods[name]
        analysis_func = method_config['func']
        kwargs = {**method_config['kwargs'], **override_kwargs}
        n_placebos = kwargs.pop('n_placebos', method_config['n_placebos'])

        pipeline = CausalAnalysisPipeline(name, data_dict, donor_pools)
        return pipeline.run(analysis_func, n_placebos, **kwargs)

    def run_all(self,
                data_dict: Dict,
                donor_pools: Dict[str, List[str]],
                combine: bool = True) -> Union[pd.DataFrame, Dict[str, CausalResults]]:
        """
        Run all registered methods.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary
        donor_pools : dict
            Donor pools
        combine : bool, optional
            Combine into comparison table (default: True)

        Returns:
        --------
        pd.DataFrame or dict : Combined results or dict of individual results

        Example:
        --------
        >>> comparison_table = registry.run_all(data_dict, donor_pools)
        """
        all_results = {}

        for name in self.methods:
            print(f"\n{'='*60}")
            print(f"Running {name}")
            print(f"{'='*60}\n")

            results = self.run(name, data_dict, donor_pools)
            all_results[name] = results

        if combine:
            combined_data = []
            for name, results in all_results.items():
                if results.summary_table is not None:
                    summary = results.summary_table.copy()
                    summary['method'] = name
                    combined_data.append(summary)

            return pd.concat(combined_data, ignore_index=True) if combined_data else pd.DataFrame()
        else:
            return all_results
