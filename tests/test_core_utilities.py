"""
Unit tests for core utilities: PlaceboTestRunner, DataValidator, and causal metrics.

This test suite ensures that the new centralized utilities work correctly
and maintain backward compatibility with existing code.
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List

from causal_econ.core.placebo import PlaceboTestRunner
from causal_econ.core.base import (
    DataValidator,
    calculate_causal_metrics,
    split_pre_post_periods
)


class TestCausalMetrics(unittest.TestCase):
    """Test suite for calculate_causal_metrics function."""

    def test_basic_metrics_calculation(self):
        """Test basic metric calculation with simple data."""
        effect_dict = {
            2000: 0.1,
            2001: 0.2,
            2002: 1.5,  # Treatment year
            2003: 1.8
        }
        treatment_year = 2002

        metrics = calculate_causal_metrics(effect_dict, treatment_year)

        # Check all metrics are present
        self.assertIn('pre_rmse', metrics)
        self.assertIn('post_rmse', metrics)
        self.assertIn('rmse_ratio', metrics)
        self.assertIn('att', metrics)
        self.assertIn('n_pre_periods', metrics)
        self.assertIn('n_post_periods', metrics)

        # Check period counts
        self.assertEqual(metrics['n_pre_periods'], 2)
        self.assertEqual(metrics['n_post_periods'], 2)

        # Check ATT calculation
        expected_att = (1.5 + 1.8) / 2
        self.assertAlmostEqual(metrics['att'], expected_att)

        # Check RMSE calculations
        expected_pre_rmse = np.sqrt((0.1**2 + 0.2**2) / 2)
        self.assertAlmostEqual(metrics['pre_rmse'], expected_pre_rmse)

        expected_post_rmse = np.sqrt((1.5**2 + 1.8**2) / 2)
        self.assertAlmostEqual(metrics['post_rmse'], expected_post_rmse)

        # Check RMSE ratio
        expected_ratio = expected_post_rmse / expected_pre_rmse
        self.assertAlmostEqual(metrics['rmse_ratio'], expected_ratio)

    def test_metrics_with_nan_values(self):
        """Test that NaN values are properly filtered out."""
        effect_dict = {
            2000: 0.1,
            2001: np.nan,  # Should be filtered out
            2002: 1.5,
            2003: np.nan   # Should be filtered out
        }
        treatment_year = 2002

        metrics = calculate_causal_metrics(effect_dict, treatment_year)

        # Should only count non-NaN periods
        self.assertEqual(metrics['n_pre_periods'], 1)
        self.assertEqual(metrics['n_post_periods'], 1)

        # ATT should only use valid post period
        self.assertAlmostEqual(metrics['att'], 1.5)

    def test_metrics_with_none_values(self):
        """Test that None values are properly filtered out."""
        effect_dict = {
            2000: 0.1,
            2001: None,  # Should be filtered out
            2002: 1.5,
            2003: None   # Should be filtered out
        }
        treatment_year = 2002

        metrics = calculate_causal_metrics(effect_dict, treatment_year)

        self.assertEqual(metrics['n_pre_periods'], 1)
        self.assertEqual(metrics['n_post_periods'], 1)

    def test_metrics_with_no_valid_pre_periods(self):
        """Test behavior when no valid pre-treatment periods exist."""
        effect_dict = {
            2002: 1.5,
            2003: 1.8
        }
        treatment_year = 2002

        metrics = calculate_causal_metrics(effect_dict, treatment_year)

        self.assertEqual(metrics['n_pre_periods'], 0)
        self.assertTrue(np.isnan(metrics['pre_rmse']))
        self.assertTrue(np.isnan(metrics['rmse_ratio']))

    def test_metrics_with_no_valid_post_periods(self):
        """Test behavior when no valid post-treatment periods exist."""
        effect_dict = {
            2000: 0.1,
            2001: 0.2
        }
        treatment_year = 2002

        metrics = calculate_causal_metrics(effect_dict, treatment_year)

        self.assertEqual(metrics['n_post_periods'], 0)
        self.assertTrue(np.isnan(metrics['post_rmse']))
        self.assertTrue(np.isnan(metrics['att']))


class TestSplitPrePostPeriods(unittest.TestCase):
    """Test suite for split_pre_post_periods function."""

    def test_basic_split(self):
        """Test basic year splitting."""
        years = [2000, 2001, 2002, 2003, 2004]
        treatment_year = 2002

        pre, post = split_pre_post_periods(years, treatment_year)

        self.assertEqual(pre, [2000, 2001])
        self.assertEqual(post, [2002, 2003, 2004])

    def test_treatment_at_start(self):
        """Test when treatment is at the start."""
        years = [2000, 2001, 2002, 2003]
        treatment_year = 2000

        pre, post = split_pre_post_periods(years, treatment_year)

        self.assertEqual(pre, [])
        self.assertEqual(post, [2000, 2001, 2002, 2003])

    def test_treatment_at_end(self):
        """Test when treatment is at the end."""
        years = [2000, 2001, 2002, 2003]
        treatment_year = 2004

        pre, post = split_pre_post_periods(years, treatment_year)

        self.assertEqual(pre, [2000, 2001, 2002, 2003])
        self.assertEqual(post, [])

    def test_empty_years(self):
        """Test with empty year list."""
        years = []
        treatment_year = 2002

        pre, post = split_pre_post_periods(years, treatment_year)

        self.assertEqual(pre, [])
        self.assertEqual(post, [])


class TestDataValidator(unittest.TestCase):
    """Test suite for DataValidator class."""

    def setUp(self):
        """Set up test data."""
        # Create sample panel data
        years = [2000, 2001, 2002, 2003, 2004]
        countries = ['USA', 'Canada', 'UK', 'Germany']

        panel = pd.DataFrame(
            np.random.randn(len(years), len(countries)),
            index=years,
            columns=countries
        )

        # Add some NaN values
        panel.loc[2000, 'Germany'] = np.nan

        self.data_dict = {
            'panel': panel,
            'treatment_years': {'USA': 2002, 'UK': 2003},
            'treated_countries': ['USA', 'UK']
        }

        self.validator = DataValidator(self.data_dict)

    def test_validate_country_exists(self):
        """Test validation of existing country."""
        self.assertTrue(self.validator.validate_country('USA'))
        self.assertTrue(self.validator.validate_country('Canada'))

    def test_validate_country_not_exists(self):
        """Test validation of non-existing country."""
        self.assertFalse(self.validator.validate_country('France'))

    def test_validate_country_raises_error(self):
        """Test that validation can raise errors."""
        with self.assertRaises(ValueError):
            self.validator.validate_country('France', raise_error=True)

    def test_validate_treatment_year_exists(self):
        """Test validation of treatment year."""
        self.assertTrue(self.validator.validate_treatment_year('USA'))
        self.assertFalse(self.validator.validate_treatment_year('Canada'))

    def test_validate_donor_pool_sufficient(self):
        """Test donor pool validation."""
        self.assertTrue(
            self.validator.validate_donor_pool(['Canada', 'Germany'], min_donors=2)
        )
        self.assertFalse(
            self.validator.validate_donor_pool(['Canada'], min_donors=2)
        )

    def test_validate_data_availability(self):
        """Test data availability check."""
        self.assertTrue(self.validator.validate_data_availability('USA'))

    def test_validate_for_analysis_success(self):
        """Test comprehensive validation - success case."""
        valid, error = self.validator.validate_for_analysis(
            'USA',
            ['Canada', 'Germany']
        )
        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_validate_for_analysis_missing_country(self):
        """Test comprehensive validation - missing country."""
        valid, error = self.validator.validate_for_analysis(
            'France',
            ['Canada', 'Germany']
        )
        self.assertFalse(valid)
        self.assertIn('not found', error)

    def test_validate_for_analysis_insufficient_donors(self):
        """Test comprehensive validation - insufficient donors."""
        valid, error = self.validator.validate_for_analysis(
            'USA',
            ['Canada']
        )
        self.assertFalse(valid)
        self.assertIn('Insufficient donors', error)

    def test_get_pre_post_data_counts(self):
        """Test pre/post period counting."""
        counts = self.validator.get_pre_post_data_counts('USA')

        self.assertEqual(counts['pre_periods'], 2)  # 2000, 2001
        self.assertEqual(counts['post_periods'], 3)  # 2002, 2003, 2004
        self.assertEqual(counts['total_periods'], 5)


class TestPlaceboTestRunner(unittest.TestCase):
    """Test suite for PlaceboTestRunner class."""

    def setUp(self):
        """Set up test data."""
        # Create sample panel data
        years = [2000, 2001, 2002, 2003, 2004]
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France']

        panel = pd.DataFrame(
            np.random.randn(len(years), len(countries)) * 2 + 5,
            index=years,
            columns=countries
        )

        self.data_dict = {
            'panel': panel,
            'treatment_years': {'USA': 2002},
            'treated_countries': ['USA']
        }

        self.runner = PlaceboTestRunner(self.data_dict, n_placebos=3)

    def test_initialization(self):
        """Test PlaceboTestRunner initialization."""
        self.assertEqual(self.runner.n_placebos, 3)
        self.assertIsNotNone(self.runner.panel)
        self.assertEqual(len(self.runner.treated_countries), 1)

    def test_select_placebo_countries_no_weights(self):
        """Test placebo selection without weights."""
        donor_pool = ['Canada', 'UK', 'Germany', 'France']

        placebos = self.runner._select_placebo_countries(
            donor_pool, weights={}, unit_weights={}
        )

        # Should select up to n_placebos
        self.assertLessEqual(len(placebos), 3)
        # Should not include treated countries
        self.assertNotIn('USA', placebos)

    def test_select_placebo_countries_with_weights(self):
        """Test placebo selection with weights."""
        donor_pool = ['Canada', 'UK', 'Germany', 'France']
        weights = {
            'Canada': 0.5,
            'UK': 0.3,
            'Germany': 0.2,
            'France': 0.005  # Below threshold
        }

        placebos = self.runner._select_placebo_countries(
            donor_pool, weights=weights, unit_weights={}
        )

        # Should prioritize weighted donors
        self.assertIn('Canada', placebos)
        self.assertIn('UK', placebos)
        # France has low weight, may or may not be included

    def test_get_placebo_donors(self):
        """Test donor pool creation for placebo."""
        donor_pool = ['Canada', 'UK', 'Germany', 'France']

        placebo_donors = self.runner._get_placebo_donors(
            placebo_country='Canada',
            treated_country='USA',
            donor_pool=donor_pool
        )

        # Should exclude both placebo and treated country
        self.assertNotIn('Canada', placebo_donors)
        self.assertNotIn('USA', placebo_donors)
        # Should include others
        self.assertIn('UK', placebo_donors)
        self.assertIn('Germany', placebo_donors)

    def test_create_placebo_data_dict(self):
        """Test creation of modified data dict for placebo."""
        temp_data = self.runner._create_placebo_data_dict(
            placebo_country='Canada',
            treatment_year=2002
        )

        # Should have Canada as treated
        self.assertIn('Canada', temp_data['treatment_years'])
        self.assertEqual(temp_data['treatment_years']['Canada'], 2002)

        # Original data should be unchanged
        self.assertNotIn('Canada', self.data_dict['treatment_years'])

    def test_calculate_p_values_empty_placebos(self):
        """Test p-value calculation with no placebo results."""
        result = {'rmse_ratio': 1.5}

        p_result = self.runner._calculate_p_values(
            result, placebo_results=[], metric_name='rmse_ratio'
        )

        self.assertTrue(np.isnan(p_result['p_value_ratio']))
        self.assertEqual(len(p_result['placebos']), 0)

    def test_calculate_p_values_with_placebos(self):
        """Test p-value calculation with placebo results."""
        result = {'rmse_ratio': 2.0, 'country': 'USA'}

        placebo_results = [
            {'country': 'Canada', 'rmse_ratio': 1.5},
            {'country': 'UK', 'rmse_ratio': 2.5},
            {'country': 'Germany', 'rmse_ratio': 1.0},
            {'country': 'France', 'rmse_ratio': 3.0}
        ]

        p_result = self.runner._calculate_p_values(
            result, placebo_results, metric_name='rmse_ratio'
        )

        # 2 out of 4 placebos have ratio >= 2.0 (UK=2.5, France=3.0)
        expected_p_value = 2 / 4
        self.assertAlmostEqual(p_result['p_value_ratio'], expected_p_value)
        self.assertEqual(len(p_result['placebos']), 4)

    def test_run_single_placebo_insufficient_donors(self):
        """Test that placebo returns None with insufficient donors."""
        # Mock analysis function
        def mock_analysis(country, donors, data):
            return {'country': country, 'rmse_ratio': 1.5}

        result = self.runner._run_single_placebo(
            placebo_country='Canada',
            treated_country='USA',
            donor_pool=['UK'],  # Only 1 donor, need at least 2
            treatment_year=2002,
            analysis_func=mock_analysis,
            metric_name='rmse_ratio'
        )

        self.assertIsNone(result)

    def test_run_single_placebo_success(self):
        """Test successful placebo execution."""
        # Mock analysis function
        def mock_analysis(country, donors, data):
            return {'country': country, 'rmse_ratio': 1.5}

        result = self.runner._run_single_placebo(
            placebo_country='Canada',
            treated_country='USA',
            donor_pool=['UK', 'Germany', 'France'],
            treatment_year=2002,
            analysis_func=mock_analysis,
            metric_name='rmse_ratio'
        )

        self.assertIsNotNone(result)
        self.assertEqual(result['country'], 'Canada')
        self.assertEqual(result['rmse_ratio'], 1.5)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCausalMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestSplitPrePostPeriods))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestPlaceboTestRunner))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    # Exit with error code if tests failed
    exit(0 if result.wasSuccessful() else 1)
