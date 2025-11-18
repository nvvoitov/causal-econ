"""
Unit tests for Phase 2 framework: Pipeline, Optimization, and Results.

This test suite ensures that the Pipeline, WeightOptimizer, ResultsAggregator,
and AnalysisRegistry classes work correctly and eliminate code duplication.
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List

from causal_econ.core.pipeline import CausalAnalysisPipeline, AnalysisRegistry
from causal_econ.core.optimization import WeightOptimizer
from causal_econ.core.results import ResultsAggregator


class TestWeightOptimizer(unittest.TestCase):
    """Test suite for WeightOptimizer class."""

    def setUp(self):
        """Set up test data."""
        self.optimizer = WeightOptimizer()

        # Create simple test data
        np.random.seed(42)
        self.X_target = np.array([1, 2, 3, 4, 5])
        self.X_donors = np.array([
            [1, 0, 0.5],
            [2, 1, 1.5],
            [3, 2, 2.5],
            [4, 3, 3.5],
            [5, 4, 4.5]
        ])

    def test_optimize_unit_weights_basic(self):
        """Test basic unit weight optimization."""
        result = self.optimizer.optimize_unit_weights(
            X_target=self.X_target,
            X_donors=self.X_donors,
            method='rmse',
            sum_to_one=True,
            non_negative=True
        )

        # Check result structure
        self.assertIn('weights', result)
        self.assertIn('objective_value', result)
        self.assertIn('success', result)

        # Check weights sum to 1
        weights = result['weights']
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)

        # Check weights are non-negative
        self.assertTrue(np.all(weights >= 0))

    def test_optimize_unit_weights_with_intercept(self):
        """Test weight optimization with intercept."""
        result = self.optimizer.optimize_unit_weights(
            X_target=self.X_target,
            X_donors=self.X_donors,
            intercept=True
        )

        # Should have both intercept and weights
        self.assertIn('intercept', result)
        self.assertIn('weights', result)

        # Weights should still sum to 1
        self.assertAlmostEqual(np.sum(result['weights']), 1.0, places=5)

    def test_optimize_unit_weights_with_regularization(self):
        """Test weight optimization with regularization."""
        result_no_reg = self.optimizer.optimize_unit_weights(
            X_target=self.X_target,
            X_donors=self.X_donors,
            regularization=0.0
        )

        result_with_reg = self.optimizer.optimize_unit_weights(
            X_target=self.X_target,
            X_donors=self.X_donors,
            regularization=0.1
        )

        # Regularization should change the weights
        self.assertFalse(np.allclose(
            result_no_reg['weights'],
            result_with_reg['weights']
        ))

    def test_optimize_time_weights(self):
        """Test time weight optimization."""
        np.random.seed(42)
        Y_pre = np.random.randn(10, 5)  # 10 units, 5 periods
        Y_post = np.random.randn(10)  # Average post outcome

        result = self.optimizer.optimize_time_weights(Y_pre, Y_post)

        # Check result structure
        self.assertIn('weights', result)
        self.assertIn('objective_value', result)

        # Check weights sum to 1
        self.assertAlmostEqual(np.sum(result['weights']), 1.0, places=5)

        # Check weights are non-negative
        self.assertTrue(np.all(result['weights'] >= 0))

    def test_optimize_sc_weights_interface(self):
        """Test Synthetic Control weights interface (backward compatibility)."""
        # Create panel data
        years = [2000, 2001, 2002, 2003]
        countries = ['USA', 'Canada', 'UK', 'Germany']

        panel = pd.DataFrame(
            np.random.randn(len(years), len(countries)) + 5,
            index=years,
            columns=countries
        )

        weights, rmse = self.optimizer.optimize_sc_weights(
            treated_country='USA',
            donor_countries=['Canada', 'UK', 'Germany'],
            panel=panel,
            pre_treatment_years=[2000, 2001]
        )

        # Check weights dictionary
        self.assertIsInstance(weights, dict)
        self.assertIn('Canada', weights)
        self.assertIn('UK', weights)
        self.assertIn('Germany', weights)

        # Check weights sum to 1
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)

        # Check RMSE is returned
        self.assertIsNotNone(rmse)
        self.assertGreater(rmse, 0)

    def test_create_weights_dict(self):
        """Test weights dictionary creation."""
        weights = np.array([0.5, 0.3, 0.15, 0.05])
        names = ['A', 'B', 'C', 'D']

        # No threshold
        weights_dict = WeightOptimizer.create_weights_dict(weights, names)
        self.assertEqual(len(weights_dict), 4)

        # With threshold
        weights_dict_filtered = WeightOptimizer.create_weights_dict(
            weights, names, threshold=0.1
        )
        self.assertEqual(len(weights_dict_filtered), 3)  # D filtered out
        self.assertNotIn('D', weights_dict_filtered)


class TestResultsAggregator(unittest.TestCase):
    """Test suite for ResultsAggregator class."""

    def setUp(self):
        """Set up test data."""
        self.aggregator = ResultsAggregator(method_name='Test Method')

    def test_add_result(self):
        """Test adding a result."""
        method_result = {
            'country': 'USA',
            'treatment_year': 2002,
            'att': 1.5,
            'pre_rmse': 0.5,
            'post_rmse': 0.6,
            'rmse_ratio': 1.2
        }

        placebo_result = {
            'p_value_ratio': 0.05,
            'placebos': [],
            'placebo_countries': []
        }

        self.aggregator.add_result('USA', method_result, placebo_result)

        # Check results were stored
        self.assertIn('USA', self.aggregator.raw_results)
        self.assertIn('USA', self.aggregator.results.country_results)

    def test_create_summary_table(self):
        """Test summary table creation."""
        # Add multiple results
        for country, att, p_value in [('USA', 1.5, 0.05), ('UK', 2.0, 0.15), ('France', -0.5, 0.08)]:
            method_result = {
                'country': country,
                'treatment_year': 2002,
                'att': att,
                'pre_rmse': 0.5,
                'post_rmse': 0.6,
                'rmse_ratio': 1.2
            }

            placebo_result = {
                'p_value_ratio': p_value
            }

            self.aggregator.add_result(country, method_result, placebo_result)

        # Create summary table
        summary = self.aggregator.create_summary_table()

        # Check structure
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 3)
        self.assertIn('country', summary.columns)
        self.assertIn('ATT', summary.columns)
        self.assertIn('p-val', summary.columns)
        self.assertIn('significant', summary.columns)

        # Check sorting (should be by p-val)
        self.assertTrue(summary['p-val'].is_monotonic_increasing)

        # Check significance
        significant_countries = summary[summary['significant'] == 'Yes']['country'].tolist()
        self.assertIn('USA', significant_countries)
        self.assertIn('France', significant_countries)
        self.assertNotIn('UK', significant_countries)  # p=0.15 > 0.1

    def test_create_weights_table(self):
        """Test weights table creation."""
        # Add results with weights
        method_result = {
            'country': 'USA',
            'treatment_year': 2002,
            'att': 1.5,
            'pre_rmse': 0.5,
            'post_rmse': 0.6,
            'rmse_ratio': 1.2,
            'weights': {
                'Canada': 0.5,
                'UK': 0.3,
                'Germany': 0.2
            }
        }

        self.aggregator.add_result('USA', method_result)

        # Create weights table
        weights_table = self.aggregator.create_weights_table(threshold=0.1)

        # Check structure
        self.assertIsInstance(weights_table, pd.DataFrame)
        self.assertEqual(len(weights_table), 3)
        self.assertIn('treated_country', weights_table.columns)
        self.assertIn('donor_country', weights_table.columns)
        self.assertIn('weight', weights_table.columns)

    def test_get_significant_countries(self):
        """Test getting significant countries."""
        # Add results with various p-values
        for country, p_value in [('USA', 0.05), ('UK', 0.15), ('France', 0.08)]:
            method_result = {
                'country': country,
                'att': 1.0,
                'pre_rmse': 0.5,
                'post_rmse': 0.6,
                'rmse_ratio': 1.2
            }
            placebo_result = {'p_value_ratio': p_value}

            self.aggregator.add_result(country, method_result, placebo_result)

        # Get significant countries
        significant = self.aggregator.get_significant_countries(threshold=0.1)

        # Check results
        self.assertEqual(len(significant), 2)
        self.assertIn('USA', significant)
        self.assertIn('France', significant)
        self.assertNotIn('UK', significant)

    def test_get_summary_statistics(self):
        """Test summary statistics calculation."""
        # Add some results
        for country, att, rmse_ratio in [('USA', 1.5, 1.2), ('UK', 2.0, 1.5), ('France', -0.5, 0.8)]:
            method_result = {
                'country': country,
                'att': att,
                'pre_rmse': 0.5,
                'post_rmse': rmse_ratio * 0.5,
                'rmse_ratio': rmse_ratio
            }

            self.aggregator.add_result(country, method_result)

        # Get statistics
        stats = self.aggregator.get_summary_statistics()

        # Check structure
        self.assertIn('n_countries', stats)
        self.assertIn('mean_att', stats)
        self.assertIn('median_att', stats)
        self.assertIn('n_good_fit', stats)

        # Check values
        self.assertEqual(stats['n_countries'], 3)
        self.assertAlmostEqual(stats['mean_att'], 1.0, places=5)  # (1.5 + 2.0 - 0.5) / 3
        self.assertEqual(stats['n_good_fit'], 1)  # Only France has ratio < 2

    def test_build_method_result(self):
        """Test building standardized method result."""
        effect_dict = {
            2000: 0.1,
            2001: 0.2,
            2002: 1.5,
            2003: 1.8
        }

        result = ResultsAggregator.build_method_result(
            country='USA',
            treatment_year=2002,
            effect=effect_dict,
            weights={'Canada': 0.7, 'UK': 0.3}
        )

        # Check required fields
        self.assertEqual(result['country'], 'USA')
        self.assertEqual(result['treatment_year'], 2002)
        self.assertIn('att', result)
        self.assertIn('pre_rmse', result)
        self.assertIn('rmse_ratio', result)
        self.assertIn('weights', result)

        # Check metrics were calculated
        self.assertGreater(result['att'], 0)


class TestCausalAnalysisPipeline(unittest.TestCase):
    """Test suite for CausalAnalysisPipeline class."""

    def setUp(self):
        """Set up test data."""
        # Create panel data
        years = [2000, 2001, 2002, 2003, 2004]
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France']

        panel = pd.DataFrame(
            np.random.randn(len(years), len(countries)) * 2 + 5,
            index=years,
            columns=countries
        )

        self.data_dict = {
            'panel': panel,
            'treatment_years': {'USA': 2002, 'UK': 2003},
            'treated_countries': ['USA', 'UK'],
            'df': pd.DataFrame()  # Placeholder
        }

        self.donor_pools = {
            'USA': ['Canada', 'Germany', 'France'],
            'UK': ['Canada', 'Germany', 'France']
        }

        self.pipeline = CausalAnalysisPipeline(
            method_name='Test Method',
            data_dict=self.data_dict,
            donor_pools=self.donor_pools
        )

    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.method_name, 'Test Method')
        self.assertIsNotNone(self.pipeline.validator)
        self.assertIsNotNone(self.pipeline.aggregator)
        self.assertEqual(len(self.pipeline.treated_countries), 2)

    def test_run_with_mock_analysis(self):
        """Test running pipeline with mock analysis function."""
        # Mock analysis function
        def mock_analysis(country, donor_pool, data_dict):
            effect = {2000: 0.1, 2001: 0.2, 2002: 1.5, 2003: 1.8, 2004: 2.0}
            treatment_year = data_dict['treatment_years'][country]

            return ResultsAggregator.build_method_result(
                country=country,
                treatment_year=treatment_year,
                effect=effect
            )

        # Run pipeline (no placebos for speed)
        results = self.pipeline.run(
            analysis_func=mock_analysis,
            n_placebos=0,
            verbose=False
        )

        # Check results
        self.assertIsNotNone(results.summary_table)
        self.assertEqual(len(results.country_results), 2)
        self.assertIn('USA', results.country_results)
        self.assertIn('UK', results.country_results)

    def test_run_batch(self):
        """Test running analysis for specific countries."""
        def mock_analysis(country, donor_pool, data_dict):
            effect = {2000: 0.1, 2002: 1.5, 2004: 2.0}
            return ResultsAggregator.build_method_result(
                country=country,
                treatment_year=2002,
                effect=effect
            )

        # Run only USA
        results = self.pipeline.run_batch(
            countries=['USA'],
            analysis_func=mock_analysis,
            n_placebos=0,
            verbose=False
        )

        # Should only have USA
        self.assertEqual(len(results.country_results), 1)
        self.assertIn('USA', results.country_results)
        self.assertNotIn('UK', results.country_results)

    def test_get_aggregator(self):
        """Test getting aggregator object."""
        aggregator = self.pipeline.get_aggregator()
        self.assertIsInstance(aggregator, ResultsAggregator)

    def test_get_validator(self):
        """Test getting validator object."""
        validator = self.pipeline.get_validator()
        from causal_econ.core.base import DataValidator
        self.assertIsInstance(validator, DataValidator)


class TestAnalysisRegistry(unittest.TestCase):
    """Test suite for AnalysisRegistry class."""

    def setUp(self):
        """Set up test data."""
        self.registry = AnalysisRegistry()

        # Create test data
        years = [2000, 2001, 2002, 2003, 2004]
        countries = ['USA', 'Canada', 'UK']

        panel = pd.DataFrame(
            np.random.randn(len(years), len(countries)) + 5,
            index=years,
            columns=countries
        )

        self.data_dict = {
            'panel': panel,
            'treatment_years': {'USA': 2002},
            'treated_countries': ['USA'],
            'df': pd.DataFrame()
        }

        self.donor_pools = {'USA': ['Canada', 'UK']}

    def test_register_method(self):
        """Test registering a method."""
        def mock_method(country, donors, data):
            return {'country': country, 'att': 1.0}

        self.registry.register('Mock', mock_method)

        # Check registration
        self.assertIn('Mock', self.registry.methods)

    def test_run_registered_method(self):
        """Test running a registered method."""
        def mock_method(country, donors, data):
            effect = {2000: 0.1, 2002: 1.5, 2004: 2.0}
            return ResultsAggregator.build_method_result(
                country=country,
                treatment_year=2002,
                effect=effect
            )

        self.registry.register('Mock', mock_method, default_n_placebos=0)

        # Run method
        results = self.registry.run('Mock', self.data_dict, self.donor_pools)

        # Check results
        self.assertIsNotNone(results)
        self.assertIn('USA', results.country_results)

    def test_run_unregistered_method_raises_error(self):
        """Test that running unregistered method raises error."""
        with self.assertRaises(ValueError):
            self.registry.run('Nonexistent', self.data_dict, self.donor_pools)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWeightOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestResultsAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalAnalysisPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisRegistry))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    exit(0 if result.wasSuccessful() else 1)
