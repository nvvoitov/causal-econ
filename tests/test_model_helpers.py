"""
Unit tests for model helper classes.

Tests SyntheticControlHelper, DifferenceInDifferencesHelper,
SDiDHelper, and GSCHelper classes.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from causal_econ.models.helpers import (
    SyntheticControlHelper,
    DifferenceInDifferencesHelper,
    SDiDHelper,
    GSCHelper
)
from causal_econ.core.optimization import WeightOptimizer
from causal_econ.core.results import ResultsAggregator


@pytest.fixture
def sample_panel_data():
    """Create sample panel data for testing."""
    years = list(range(2000, 2011))
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France']

    np.random.seed(42)
    data = {}
    for country in countries:
        # Create realistic GDP growth patterns
        base = np.random.randn() * 2 + 3  # Mean around 3%
        noise = np.random.randn(len(years)) * 1.5
        data[country] = base + noise

    panel = pd.DataFrame(data, index=years)
    return panel


@pytest.fixture
def sample_data_dict(sample_panel_data):
    """Create sample data dictionary for testing."""
    return {
        'panel': sample_panel_data,
        'treated_countries': ['USA'],
        'treatment_years': {'USA': 2005},
        'df': pd.DataFrame()  # Simplified for tests
    }


# ==============================================================================
# SyntheticControlHelper Tests
# ==============================================================================

class TestSyntheticControlHelper:
    """Test SyntheticControlHelper class."""

    def test_initialization(self):
        """Test helper initialization."""
        helper = SyntheticControlHelper()
        assert helper.optimizer is not None
        assert isinstance(helper.optimizer, WeightOptimizer)
        assert helper.aggregator is None

    def test_initialization_with_params(self):
        """Test helper initialization with custom parameters."""
        optimizer = WeightOptimizer(method='L-BFGS-B', max_iter=500)
        helper = SyntheticControlHelper(optimizer=optimizer)
        assert helper.optimizer is optimizer
        assert helper.optimizer.method == 'L-BFGS-B'
        assert helper.optimizer.max_iter == 500

    def test_construct_synthetic(self, sample_panel_data):
        """Test synthetic control construction."""
        helper = SyntheticControlHelper()

        weights = {'Canada': 0.5, 'UK': 0.3, 'Germany': 0.2}
        years = list(range(2000, 2011))

        synthetic = helper._construct_synthetic(
            weights, sample_panel_data, years
        )

        assert len(synthetic) == len(years)
        assert all(year in synthetic for year in years)

        # Check values are weighted sums
        for year in years:
            expected = (
                0.5 * sample_panel_data.loc[year, 'Canada'] +
                0.3 * sample_panel_data.loc[year, 'UK'] +
                0.2 * sample_panel_data.loc[year, 'Germany']
            )
            assert np.isclose(synthetic[year], expected)

    def test_calculate_effects(self, sample_panel_data):
        """Test effect calculation."""
        helper = SyntheticControlHelper()

        country = 'USA'
        years = list(range(2000, 2011))

        # Create synthetic values
        synthetic_values = {year: sample_panel_data.loc[year, country] - 1
                          for year in years}

        effects = helper._calculate_effects(
            country, synthetic_values, sample_panel_data, years
        )

        assert len(effects) == len(years)
        for year in years:
            actual = sample_panel_data.loc[year, country]
            expected_effect = actual - synthetic_values[year]
            assert np.isclose(effects[year], expected_effect)

    def test_run_for_country_success(self, sample_data_dict):
        """Test successful SC analysis for a country."""
        helper = SyntheticControlHelper()
        helper.initialize_aggregator('Synthetic Control')

        donor_pool = ['Canada', 'UK', 'Germany', 'France']

        result = helper.run_for_country(
            country='USA',
            donor_pool=donor_pool,
            data_dict=sample_data_dict
        )

        assert result is not None
        assert 'country' in result
        assert result['country'] == 'USA'
        assert 'treatment_year' in result
        assert result['treatment_year'] == 2005
        assert 'att' in result
        assert 'pre_rmse' in result
        assert 'post_rmse' in result
        assert 'rmse_ratio' in result
        assert 'weights' in result
        assert 'effect' in result

    def test_run_for_country_missing_data(self, sample_data_dict):
        """Test SC analysis with missing country data."""
        helper = SyntheticControlHelper()
        helper.initialize_aggregator('Synthetic Control')

        # Remove USA from panel
        sample_data_dict['panel'] = sample_data_dict['panel'].drop('USA', axis=1)

        result = helper.run_for_country(
            country='USA',
            donor_pool=['Canada', 'UK'],
            data_dict=sample_data_dict
        )

        assert result is None

    def test_run_for_country_no_valid_donors(self, sample_data_dict):
        """Test SC analysis with no valid donors."""
        helper = SyntheticControlHelper()
        helper.initialize_aggregator('Synthetic Control')

        # Use empty donor pool
        result = helper.run_for_country(
            country='USA',
            donor_pool=[],
            data_dict=sample_data_dict
        )

        assert result is None


# ==============================================================================
# DifferenceInDifferencesHelper Tests
# ==============================================================================

class TestDifferenceInDifferencesHelper:
    """Test DifferenceInDifferencesHelper class."""

    def test_initialization(self):
        """Test helper initialization."""
        helper = DifferenceInDifferencesHelper()
        assert helper.aggregator is None

    def test_calculate_parallel_shift(self):
        """Test parallel shift calculation."""
        helper = DifferenceInDifferencesHelper()

        treated_pre = np.array([1, 2, 3, 4, 5])
        control_pre = np.array([2, 3, 4, 5, 6])

        shift = helper._calculate_parallel_shift(treated_pre, control_pre)

        # Shift should be mean difference
        expected_shift = np.mean(treated_pre - control_pre)
        assert np.isclose(shift, expected_shift)

    def test_calculate_parallel_shift_with_nans(self):
        """Test parallel shift with NaN values."""
        helper = DifferenceInDifferencesHelper()

        treated_pre = np.array([1, 2, np.nan, 4, 5])
        control_pre = np.array([2, 3, 4, np.nan, 6])

        shift = helper._calculate_parallel_shift(treated_pre, control_pre)

        # Should handle NaNs gracefully
        assert not np.isnan(shift)

    def test_calculate_did_effect(self):
        """Test DiD effect calculation."""
        helper = DifferenceInDifferencesHelper()

        # Pre-treatment: treated=5, control=3, shift=2
        # Post-treatment: treated=8, control=4
        # DiD = (8-5) - (4-3) - 2 = 3 - 1 - 2 = 0... wait that's wrong
        # Actually: DiD = (treated_post - control_post - shift) - (treated_pre_avg - control_pre_avg)
        # But shift IS (treated_pre_avg - control_pre_avg), so:
        # DiD = treated_post - control_post - shift

        shift = 2.0
        treated_post = 8.0
        control_post = 4.0

        effect = helper._calculate_did_effect(treated_post, control_post, shift)

        expected = treated_post - control_post - shift
        assert np.isclose(effect, expected)

    def test_run_for_country_success(self, sample_data_dict):
        """Test successful DiD analysis."""
        helper = DifferenceInDifferencesHelper()
        helper.initialize_aggregator('Difference-in-Differences')

        match_info = {
            'match': 'Canada',
            'score': 0.85
        }

        result = helper.run_for_country(
            country='USA',
            match_info=match_info,
            data_dict=sample_data_dict
        )

        assert result is not None
        assert 'country' in result
        assert result['country'] == 'USA'
        assert 'synth_match' in result
        assert result['synth_match'] == 'Canada'
        assert 'shift' in result
        assert 'att' in result
        assert 'effect' in result


# ==============================================================================
# SDiDHelper Tests
# ==============================================================================

class TestSDiDHelper:
    """Test SDiDHelper class."""

    def test_initialization(self):
        """Test helper initialization."""
        helper = SDiDHelper()
        assert helper.optimizer is not None
        assert helper.regularization == 1e-6

    def test_initialization_with_regularization(self):
        """Test initialization with custom regularization."""
        helper = SDiDHelper(regularization=0.01)
        assert helper.regularization == 0.01

    def test_optimize_weights(self, sample_data_dict):
        """Test weight optimization."""
        helper = SDiDHelper()

        result = helper._optimize_weights(
            country='USA',
            donor_pool=['Canada', 'UK', 'Germany'],
            data_dict=sample_data_dict
        )

        assert result is not None
        assert 'unit_weights' in result
        assert 'time_weights' in result

        # Check unit weights sum to 1
        unit_weights = result['unit_weights']
        assert np.isclose(sum(unit_weights.values()), 1.0)

        # Check time weights sum to 1
        time_weights = result['time_weights']
        assert np.isclose(sum(time_weights.values()), 1.0)

    def test_compute_double_diff(self, sample_data_dict):
        """Test double differencing."""
        helper = SDiDHelper()

        unit_weights = {'Canada': 0.5, 'UK': 0.5}
        time_weights = {2000: 0.2, 2001: 0.3, 2002: 0.3, 2003: 0.2}

        result = helper._compute_double_diff(
            country='USA',
            unit_weights=unit_weights,
            time_weights=time_weights,
            data_dict=sample_data_dict
        )

        assert 'synthetic_values' in result
        assert 'effects' in result

    def test_run_for_country_success(self, sample_data_dict):
        """Test successful SDiD analysis."""
        helper = SDiDHelper()
        helper.initialize_aggregator('Synthetic Diff-in-Diff')

        result = helper.run_for_country(
            country='USA',
            donor_pool=['Canada', 'UK', 'Germany'],
            data_dict=sample_data_dict
        )

        assert result is not None
        assert 'weights' in result  # unit weights
        assert 'time_weights' in result


# ==============================================================================
# GSCHelper Tests
# ==============================================================================

class TestGSCHelper:
    """Test GSCHelper class."""

    def test_initialization(self):
        """Test helper initialization."""
        helper = GSCHelper()
        assert helper.n_factors == 3
        assert helper.max_iter == 100

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        helper = GSCHelper(n_factors=5, max_iter=200)
        assert helper.n_factors == 5
        assert helper.max_iter == 200

    def test_estimate_factors_basic(self):
        """Test basic factor estimation."""
        helper = GSCHelper(n_factors=2)

        # Create simple data matrix
        np.random.seed(42)
        Y_control = np.random.randn(10, 5)  # 10 units, 5 time periods

        factors, loadings = helper._estimate_factors(Y_control)

        assert factors.shape == (5, 2)  # T x r
        assert loadings.shape == (10, 2)  # N x r

    def test_cross_validate_factors(self, sample_data_dict):
        """Test factor cross-validation."""
        helper = GSCHelper()

        donor_pool = ['Canada', 'UK', 'Germany']
        pre_years = [2000, 2001, 2002, 2003, 2004]

        best_r = helper._cross_validate_factors(
            donor_pool=donor_pool,
            panel=sample_data_dict['panel'],
            pre_treatment_years=pre_years,
            max_factors=3
        )

        assert isinstance(best_r, int)
        assert 1 <= best_r <= 3

    def test_run_for_country_success(self, sample_data_dict):
        """Test successful GSC analysis."""
        helper = GSCHelper(n_factors=2)
        helper.initialize_aggregator('Generalized SC')

        result = helper.run_for_country(
            country='USA',
            donor_pool=['Canada', 'UK', 'Germany'],
            data_dict=sample_data_dict
        )

        assert result is not None
        assert 'n_factors' in result
        assert 'att' in result
        assert 'effect' in result


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestHelperIntegration:
    """Test integration between helpers and aggregators."""

    def test_sc_with_aggregator(self, sample_data_dict):
        """Test SC helper with result aggregation."""
        helper = SyntheticControlHelper()
        aggregator = ResultsAggregator('Synthetic Control')
        helper.aggregator = aggregator

        result = helper.run_for_country(
            country='USA',
            donor_pool=['Canada', 'UK'],
            data_dict=sample_data_dict
        )

        if result:
            aggregator.add_result('USA', result)
            summary = aggregator.create_summary_table()

            assert len(summary) == 1
            assert summary.loc[0, 'country'] == 'USA'

    def test_multiple_helpers_same_aggregator(self, sample_data_dict):
        """Test using same aggregator for multiple helpers."""
        aggregator = ResultsAggregator('Mixed Methods')

        sc_helper = SyntheticControlHelper()
        sc_helper.aggregator = aggregator

        did_helper = DifferenceInDifferencesHelper()
        did_helper.aggregator = aggregator

        # Run SC
        sc_result = sc_helper.run_for_country(
            'USA', ['Canada', 'UK'], sample_data_dict
        )
        if sc_result:
            aggregator.add_result('USA_SC', sc_result)

        # Run DiD
        match_info = {'match': 'Canada', 'score': 0.9}
        did_result = did_helper.run_for_country(
            'USA', match_info, sample_data_dict
        )
        if did_result:
            aggregator.add_result('USA_DiD', did_result)

        summary = aggregator.create_summary_table()
        assert len(summary) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
