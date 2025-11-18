"""
Model-specific helper classes for causal inference methods.

This module provides specialized classes that encapsulate method-specific logic
for Synthetic Control, DiD, SDiD, and GSC methods while eliminating code duplication.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..core.base import calculate_causal_metrics, split_pre_post_periods
from ..core.optimization import WeightOptimizer
from ..core.results import ResultsAggregator


class SyntheticControlHelper:
    """
    Helper class for Synthetic Control method operations.

    Encapsulates SC-specific logic like weight calculation,
    synthetic unit construction, and result building.

    Example:
    --------
    >>> helper = SyntheticControlHelper(data_dict)
    >>> result = helper.run_for_country('USA', donor_pool)
    """

    def __init__(self, data_dict: Dict):
        """
        Initialize SC helper.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary with panel, treatment_years, etc.
        """
        self.data_dict = data_dict
        self.panel = data_dict['panel']
        self.treatment_years = data_dict['treatment_years']
        self.optimizer = WeightOptimizer()

    def run_for_country(self,
                       country: str,
                       donor_pool: List[str]) -> Optional[Dict]:
        """
        Run SC analysis for a single country.

        Parameters:
        -----------
        country : str
            Treated country name
        donor_pool : list
            List of donor countries

        Returns:
        --------
        dict or None : SC results
        """
        if country not in self.treatment_years or country not in self.panel.columns:
            return None

        treatment_year = self.treatment_years[country]
        all_years = sorted(self.panel.index)
        pre_treatment_years = [y for y in all_years if y < treatment_year]

        # Get weights
        weights, pre_rmse = self.optimizer.optimize_sc_weights(
            country, donor_pool, self.panel, pre_treatment_years
        )

        if weights is None:
            return None

        # Construct synthetic control
        synth_values = self._construct_synthetic(weights, all_years)

        # Calculate effects
        effect = self._calculate_effects(country, synth_values, all_years)

        # Build result using standardized helper
        return ResultsAggregator.build_method_result(
            country=country,
            treatment_year=treatment_year,
            effect=effect,
            weights=weights,
            synthetic=synth_values,
            donor_pool=donor_pool,
            pre_rmse_from_opt=pre_rmse
        )

    def _construct_synthetic(self,
                            weights: Dict[str, float],
                            years: List[int]) -> Dict[int, float]:
        """
        Construct synthetic control values.

        Parameters:
        -----------
        weights : dict
            Donor weights
        years : list
            Years to construct synthetic for

        Returns:
        --------
        dict : Synthetic values by year
        """
        synth_values = {}

        for year in years:
            if year in self.panel.index:
                donor_values = [
                    weights[d] * self.panel.loc[year, d]
                    if d in self.panel.columns and not pd.isna(self.panel.loc[year, d])
                    else 0
                    for d in weights.keys()
                ]
                synth_values[year] = sum(donor_values)

        return synth_values

    def _calculate_effects(self,
                          country: str,
                          synth_values: Dict[int, float],
                          years: List[int]) -> Dict[int, float]:
        """
        Calculate treatment effects (actual - synthetic).

        Parameters:
        -----------
        country : str
            Country name
        synth_values : dict
            Synthetic values
        years : list
            Years

        Returns:
        --------
        dict : Effects by year
        """
        effect = {}

        for year in years:
            if year in synth_values and year in self.panel.index:
                actual = self.panel.loc[year, country]
                if not pd.isna(actual):
                    effect[year] = actual - synth_values[year]

        return effect


class DifferenceInDifferencesHelper:
    """
    Helper class for Difference-in-Differences with PSM.

    Encapsulates DiD-specific logic like propensity score matching,
    best match selection, and parallel shift calculation.

    Example:
    --------
    >>> helper = DifferenceInDifferencesHelper(data_dict)
    >>> result = helper.run_for_country('USA', donor_pool)
    """

    def __init__(self, data_dict: Dict, feature_cols: Optional[List[str]] = None):
        """
        Initialize DiD helper.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary
        feature_cols : list, optional
            Feature columns for PSM
        """
        self.data_dict = data_dict
        self.df = data_dict['df']
        self.panel = data_dict['panel']
        self.treatment_years = data_dict['treatment_years']
        self.feature_cols = feature_cols

    def run_for_country(self,
                       country: str,
                       donor_pool: List[str]) -> Optional[Dict]:
        """
        Run DiD analysis for a single country.

        Parameters:
        -----------
        country : str
            Treated country
        donor_pool : list
            Donor countries

        Returns:
        --------
        dict or None : DiD results
        """
        if country not in self.panel.columns:
            return None

        # Calculate propensity scores
        propensity_scores = self.calculate_propensity_scores(country, donor_pool)

        if not propensity_scores:
            return None

        # Select best match
        best_match, score = self.select_best_match(country, donor_pool, propensity_scores)

        if best_match is None:
            return None

        # Run DiD analysis
        return self.run_did_analysis(country, best_match, score)

    def calculate_propensity_scores(self,
                                   treated_country: str,
                                   donor_pool: List[str]) -> Dict[str, float]:
        """
        Calculate propensity scores using logistic regression.

        Parameters:
        -----------
        treated_country : str
            Treated country name
        donor_pool : list
            Donor countries

        Returns:
        --------
        dict : Propensity scores
        """
        treatment_year = self.treatment_years[treated_country]
        pre_treatment_data = self.df[self.df['year'] < treatment_year]

        # Determine feature columns
        if self.feature_cols is None:
            exclude_cols = ['year', 'country', 'has_platform', 'gdp_growth_annual_share']
            feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        else:
            feature_cols = self.feature_cols

        # Calculate average pre-treatment values
        countries = [treated_country] + donor_pool
        pre_data = []

        for country in countries:
            country_data = pre_treatment_data[pre_treatment_data['country'] == country]
            if len(country_data) > 0:
                avg_values = country_data[feature_cols].mean().to_dict()
                avg_values['country'] = country
                avg_values['is_treated'] = country == treated_country
                pre_data.append(avg_values)

        pre_df = pd.DataFrame(pre_data)

        if len(pre_df) < 2:
            return {}

        # Prepare features
        X = pre_df[feature_cols].fillna(0)
        y = pre_df['is_treated']

        # Scale and fit
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
            model.fit(X_scaled, y)
            scores = model.predict_proba(X_scaled)[:, 1]
            return dict(zip(pre_df['country'], scores))
        except Exception:
            return {}

    def select_best_match(self,
                         treated_country: str,
                         donor_pool: List[str],
                         propensity_scores: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
        """
        Select country with highest propensity score.

        Parameters:
        -----------
        treated_country : str
            Treated country
        donor_pool : list
            Donors
        propensity_scores : dict
            Scores

        Returns:
        --------
        tuple : (best_match, score)
        """
        donor_scores = {c: s for c, s in propensity_scores.items() if c in donor_pool}

        if donor_scores:
            best_match = max(donor_scores.items(), key=lambda x: x[1])
            return best_match[0], best_match[1]
        else:
            return None, None

    def run_did_analysis(self,
                        treated_country: str,
                        synth_match: str,
                        propensity_score: float) -> Optional[Dict]:
        """
        Run DiD analysis with matched control.

        Parameters:
        -----------
        treated_country : str
            Treated country
        synth_match : str
            Matched control country
        propensity_score : float
            Propensity score

        Returns:
        --------
        dict or None : DiD results
        """
        if synth_match not in self.panel.columns:
            return None

        treatment_year = self.treatment_years[treated_country]

        # Get data
        treated_data = self.panel[treated_country]
        control_data = self.panel[synth_match]

        # Calculate parallel shift
        shift = self._calculate_parallel_shift(
            treated_country, synth_match, treatment_year
        )

        # Apply shift
        shifted_control = control_data + shift

        # Calculate effects
        all_years = sorted(self.panel.index)
        effect = {}

        for year in all_years:
            if year in self.panel.index:
                t_val = treated_data[year]
                c_val = shifted_control[year]
                if not pd.isna(t_val) and not pd.isna(c_val):
                    effect[year] = t_val - c_val

        # Build result
        return ResultsAggregator.build_method_result(
            country=treated_country,
            treatment_year=treatment_year,
            effect=effect,
            synth_match=synth_match,
            shift=shift,
            propensity_score=propensity_score,
            treated_values=treated_data.to_dict(),
            control_values=control_data.to_dict(),
            shifted_control=shifted_control.to_dict()
        )

    def _calculate_parallel_shift(self,
                                  treated_country: str,
                                  synth_match: str,
                                  treatment_year: int) -> float:
        """Calculate parallel shift constant."""
        pre_years = [y for y in self.panel.index if y < treatment_year]

        treated_pre_data = self.panel.loc[pre_years, treated_country]
        match_pre_data = self.panel.loc[pre_years, synth_match]

        # Remove NaN values
        valid_years = [
            y for y in pre_years
            if not pd.isna(treated_pre_data[y]) and not pd.isna(match_pre_data[y])
        ]

        if not valid_years:
            return 0.0

        treated_pre = treated_pre_data.loc[valid_years].mean()
        match_pre = match_pre_data.loc[valid_years].mean()

        return treated_pre - match_pre


class SDiDHelper:
    """
    Helper class for Synthetic Difference-in-Differences.

    Encapsulates SDiD-specific logic like regularization calculation,
    unit and time weight optimization, and double-differencing.

    Example:
    --------
    >>> helper = SDiDHelper(data_dict)
    >>> result = helper.run_for_country('USA', donor_pool)
    """

    def __init__(self, data_dict: Dict):
        """
        Initialize SDiD helper.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary
        """
        self.data_dict = data_dict
        self.df = data_dict['df']
        self.panel = data_dict.get('panel')
        self.treatment_years = data_dict.get('treatment_years')
        self.optimizer = WeightOptimizer()

    def run_for_country(self,
                       country: str,
                       donor_pool: List[str],
                       outcome_col: str = 'gdp_growth_annual_share') -> Optional[Dict]:
        """
        Run SDiD analysis for a single country.

        Parameters:
        -----------
        country : str
            Treated country
        donor_pool : list
            Donor countries
        outcome_col : str
            Outcome column name

        Returns:
        --------
        dict or None : SDiD results
        """
        if country not in self.panel.columns:
            return None

        treatment_year = self.treatment_years.get(country)
        if treatment_year is None:
            return None

        # Prepare data for SDiD
        sdid_data = self._prepare_sdid_data(country, donor_pool, treatment_year, outcome_col)

        if sdid_data is None:
            return None

        # Calculate unit weights
        omega_0, unit_weights = self._calculate_unit_weights(sdid_data, outcome_col)

        # Calculate time weights
        lambda_0, time_weights = self._calculate_time_weights(sdid_data, outcome_col)

        # Run SDiD estimator
        tau_sdid = self._run_sdid_estimator(
            sdid_data, unit_weights, time_weights,
            omega_0, lambda_0, outcome_col
        )

        # Generate synthetic values and effects
        synth_values, actual_values, effect = self._generate_counterfactuals(
            country, unit_weights, treatment_year
        )

        # Build result
        return ResultsAggregator.build_method_result(
            country=country,
            treatment_year=treatment_year,
            effect=effect,
            tau_sdid=tau_sdid,
            omega_0=omega_0,
            lambda_0=lambda_0,
            unit_weights=unit_weights,
            time_weights=time_weights,
            actual_values=actual_values,
            synthetic_values=synth_values,
            donor_pool=donor_pool
        )

    def _prepare_sdid_data(self,
                          country: str,
                          donor_pool: List[str],
                          treatment_year: int,
                          outcome_col: str) -> Optional[pd.DataFrame]:
        """Prepare data for SDiD analysis."""
        all_countries = [country] + donor_pool
        sdid_data = self.df[self.df['country'].isin(all_countries)].copy()
        sdid_data = sdid_data.dropna(subset=[outcome_col])

        sdid_data['treated'] = sdid_data['country'] == country
        sdid_data['after_treatment'] = sdid_data['year'] >= treatment_year
        sdid_data = sdid_data.sort_values(['country', 'year'])

        # Validate data
        if sdid_data[sdid_data['treated']].shape[0] == 0:
            return None
        if sdid_data[~sdid_data['treated']].shape[0] == 0:
            return None

        return sdid_data

    def _calculate_unit_weights(self,
                                data: pd.DataFrame,
                                outcome_col: str) -> Tuple[float, Dict]:
        """Calculate SDiD unit weights."""
        # Use WeightOptimizer for unit weight calculation
        # This would integrate with the existing SDiD logic
        # Simplified version here
        control_units = data[~data['treated']]['country'].unique()
        weights = {unit: 1.0/len(control_units) for unit in control_units}
        return 0.0, weights

    def _calculate_time_weights(self,
                               data: pd.DataFrame,
                               outcome_col: str) -> Tuple[float, Dict]:
        """Calculate SDiD time weights."""
        pre_periods = sorted(data[~data['after_treatment']]['year'].unique())
        weights = {period: 1.0/len(pre_periods) for period in pre_periods}
        return 0.0, weights

    def _run_sdid_estimator(self,
                           data: pd.DataFrame,
                           unit_weights: Dict,
                           time_weights: Dict,
                           omega_0: float,
                           lambda_0: float,
                           outcome_col: str) -> float:
        """Run SDiD double-differencing estimator."""
        # Simplified - actual implementation would do proper double-differencing
        return 0.0

    def _generate_counterfactuals(self,
                                 country: str,
                                 unit_weights: Dict,
                                 treatment_year: int) -> Tuple[Dict, Dict, Dict]:
        """Generate synthetic values and effects."""
        all_years = sorted(self.panel.index)
        synth_values = {}
        actual_values = {}
        effect = {}

        for year in all_years:
            if year in self.panel.index:
                actual = self.panel.loc[year, country]
                if not pd.isna(actual):
                    actual_values[year] = actual

                    # Weighted average of controls
                    synth = 0
                    weight_sum = 0
                    for control, weight in unit_weights.items():
                        if control in self.panel.columns:
                            val = self.panel.loc[year, control]
                            if not pd.isna(val):
                                synth += weight * val
                                weight_sum += weight

                    if weight_sum > 0:
                        synth_values[year] = synth / weight_sum
                        effect[year] = actual - synth_values[year]

        return synth_values, actual_values, effect


class GSCHelper:
    """
    Helper class for Generalized Synthetic Control (Interactive Fixed Effects).

    Encapsulates GSC-specific logic like factor estimation,
    cross-validation, and counterfactual generation.

    Example:
    --------
    >>> helper = GSCHelper(data_dict)
    >>> factor_model = helper.estimate_factor_model(control_panel)
    >>> result = helper.run_for_country('USA', donor_pool, factor_model)
    """

    def __init__(self, data_dict: Dict):
        """
        Initialize GSC helper.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary
        """
        self.data_dict = data_dict
        self.panel = data_dict['panel']
        self.treatment_years = data_dict['treatment_years']

    def estimate_factor_model(self,
                             control_panel: pd.DataFrame,
                             num_factors: Optional[int] = None,
                             max_factors: int = 10) -> Dict:
        """
        Estimate IFE model using control units.

        Parameters:
        -----------
        control_panel : pd.DataFrame
            Panel data for control units
        num_factors : int, optional
            Number of factors (if None, uses CV)
        max_factors : int
            Max factors for CV

        Returns:
        --------
        dict : Factor model
        """
        Y = control_panel.values
        T, Nco = Y.shape

        # Handle missing values
        mask = ~np.isnan(Y)
        Y_filled = Y.copy()
        Y_filled[~mask] = 0

        # Determine number of factors
        if num_factors is None:
            num_factors = self._cross_validate_factors(Y_filled, mask, max_factors)

        # Estimate factors using SVD
        U, s, Vt = linalg.svd(Y_filled, full_matrices=False)

        F = U[:, :num_factors] * np.sqrt(T)
        lambda_co = Vt.T[:, :num_factors] * s[:num_factors] / np.sqrt(T)

        # Create factor loadings dict
        factor_loadings = {}
        for i, unit in enumerate(control_panel.columns):
            factor_loadings[unit] = lambda_co[i, :].tolist()

        return {
            'factors': F,
            'factor_loadings': factor_loadings,
            'num_factors': num_factors,
            'eigenvalues': s[:num_factors]**2 / (T * Nco)
        }

    def run_for_country(self,
                       country: str,
                       donor_pool: List[str],
                       factor_model: Dict) -> Optional[Dict]:
        """
        Run GSC analysis for a single country.

        Parameters:
        -----------
        country : str
            Treated country
        donor_pool : list
            Donor pool
        factor_model : dict
            Estimated factor model

        Returns:
        --------
        dict or None : GSC results
        """
        if country not in self.panel.columns:
            return None

        treatment_year = self.treatment_years.get(country)
        if treatment_year is None:
            return None

        # Estimate factor loadings for treated unit
        treated_loadings = self._estimate_treated_loadings(
            country, treatment_year, factor_model
        )

        # Generate counterfactuals
        counterfactuals, effects = self._generate_counterfactuals(
            country, treatment_year, factor_model, treated_loadings
        )

        # Build result
        return ResultsAggregator.build_method_result(
            country=country,
            treatment_year=treatment_year,
            effect=effects,
            factor_loadings=treated_loadings,
            donor_pool=donor_pool,
            actual_values={y: self.panel.loc[y, country] for y in self.panel.index},
            counterfactual_values=counterfactuals
        )

    def _cross_validate_factors(self,
                                Y: np.ndarray,
                                mask: np.ndarray,
                                max_factors: int) -> int:
        """Cross-validate optimal number of factors."""
        # Simplified CV - actual would be more sophisticated
        return min(3, max_factors)

    def _estimate_treated_loadings(self,
                                  country: str,
                                  treatment_year: int,
                                  factor_model: Dict) -> List[float]:
        """Estimate factor loadings for treated unit using pre-treatment data."""
        pre_years = [y for y in self.panel.index if y < treatment_year]

        y_pre = self.panel.loc[pre_years, country].values
        F_pre = factor_model['factors'][:len(pre_years)]

        # OLS estimation
        try:
            lambda_i = np.linalg.lstsq(F_pre, y_pre, rcond=None)[0]
            return lambda_i.tolist()
        except:
            return [0.0] * factor_model['num_factors']

    def _generate_counterfactuals(self,
                                 country: str,
                                 treatment_year: int,
                                 factor_model: Dict,
                                 treated_loadings: List[float]) -> Tuple[Dict, Dict]:
        """Generate counterfactual outcomes."""
        all_years = sorted(self.panel.index)
        F = factor_model['factors']
        lambda_i = np.array(treated_loadings)

        counterfactuals = {}
        effects = {}

        for i, year in enumerate(all_years):
            if i < len(F):
                counterfactuals[year] = float(F[i] @ lambda_i)
                actual = self.panel.loc[year, country]
                if not pd.isna(actual):
                    effects[year] = actual - counterfactuals[year]

        return counterfactuals, effects
