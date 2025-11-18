"""
Unit tests for visualization plotter classes.

Tests CausalPlotter, CounterfactualExporter, and EffectDistributionPlotter classes.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from causal_econ.visualization.plotter import (
    CausalPlotter,
    CounterfactualExporter,
    EffectDistributionPlotter
)
from causal_econ.core.base import CausalResults


@pytest.fixture
def sample_time_series():
    """Create sample time series data."""
    years = list(range(2000, 2011))
    actual = {year: 3.0 + np.random.randn() * 0.5 for year in years}
    # Synthetic is similar but with treatment effect after 2005
    synthetic = {year: (3.0 + np.random.randn() * 0.5 if year < 2005
                       else 2.0 + np.random.randn() * 0.5)
                for year in years}
    return years, actual, synthetic


@pytest.fixture
def sample_results():
    """Create sample causal results."""
    results = CausalResults('Test Method')

    for i, country in enumerate(['USA', 'Canada', 'UK']):
        results.add_country_result(country, {
            'att': 1.0 + i * 0.5,
            'pre_rmse': 0.5,
            'post_rmse': 1.0,
            'rmse_ratio': 2.0,
            'p_value': 0.05 * (i + 1),
            'significant': i == 0
        })

    return results


# ==============================================================================
# CausalPlotter Tests
# ==============================================================================

class TestCausalPlotter:
    """Test CausalPlotter class."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        plotter = CausalPlotter()

        assert plotter.figsize == (12, 6)
        assert plotter.style == 'seaborn-v0_8'
        assert plotter.save_dir is None

    def test_initialization_custom(self, tmp_path):
        """Test custom initialization."""
        plotter = CausalPlotter(
            figsize=(10, 5),
            style='ggplot',
            save_dir=tmp_path
        )

        assert plotter.figsize == (10, 5)
        assert plotter.style == 'ggplot'
        assert plotter.save_dir == tmp_path

    def test_plot_counterfactual(self, sample_time_series):
        """Test counterfactual plotting."""
        years, actual, synthetic = sample_time_series

        plotter = CausalPlotter()
        fig, ax = plotter.plot_counterfactual(
            actual_values=actual,
            synthetic_values=synthetic,
            treatment_year=2005,
            country='USA'
        )

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) >= 2  # At least actual and synthetic lines

        plt.close(fig)

    def test_plot_counterfactual_with_ci(self, sample_time_series):
        """Test counterfactual plotting with confidence intervals."""
        years, actual, synthetic = sample_time_series

        # Create confidence intervals
        ci_lower = {year: synthetic[year] - 0.5 for year in years}
        ci_upper = {year: synthetic[year] + 0.5 for year in years}

        plotter = CausalPlotter()
        fig, ax = plotter.plot_counterfactual(
            actual_values=actual,
            synthetic_values=synthetic,
            treatment_year=2005,
            country='USA',
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_treatment_effects(self, sample_time_series):
        """Test treatment effects plotting."""
        years, actual, synthetic = sample_time_series

        # Calculate effects
        effects = {year: actual[year] - synthetic[year] for year in years}

        plotter = CausalPlotter()
        fig, ax = plotter.plot_treatment_effects(
            effects=effects,
            treatment_year=2005,
            country='USA'
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_plot_placebo_distribution(self):
        """Test placebo distribution plotting."""
        placebo_effects = np.random.randn(100) * 0.5
        true_effect = 1.5

        plotter = CausalPlotter()
        fig, ax = plotter.plot_placebo_distribution(
            placebo_effects=placebo_effects,
            true_effect=true_effect,
            country='USA',
            p_value=0.05
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_plot_weights_bar(self):
        """Test weight bar plotting."""
        weights = {
            'Canada': 0.4,
            'UK': 0.3,
            'Germany': 0.2,
            'France': 0.1
        }

        plotter = CausalPlotter()
        fig, ax = plotter.plot_weights(
            weights=weights,
            country='USA',
            plot_type='bar'
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_plot_weights_pie(self):
        """Test weight pie plotting."""
        weights = {
            'Canada': 0.4,
            'UK': 0.3,
            'Germany': 0.2,
            'France': 0.1
        }

        plotter = CausalPlotter()
        fig, ax = plotter.plot_weights(
            weights=weights,
            country='USA',
            plot_type='pie'
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_save_figure(self, tmp_path):
        """Test saving figure."""
        plotter = CausalPlotter(save_dir=tmp_path)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        save_path = plotter.save('test_plot.png', fig=fig)

        assert save_path.exists()
        assert save_path.suffix == '.png'

        plt.close(fig)

    def test_save_without_save_dir(self):
        """Test saving without save_dir set."""
        plotter = CausalPlotter()

        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="save_dir not set"):
            plotter.save('test.png', fig=fig)

        plt.close(fig)

    def test_close_all(self):
        """Test closing all figures."""
        plotter = CausalPlotter()

        # Create some figures
        for _ in range(3):
            fig, ax = plt.subplots()

        n_before = len(plt.get_fignums())
        assert n_before > 0

        plotter.close_all()

        n_after = len(plt.get_fignums())
        assert n_after == 0


# ==============================================================================
# CounterfactualExporter Tests
# ==============================================================================

class TestCounterfactualExporter:
    """Test CounterfactualExporter class."""

    def test_initialization(self):
        """Test exporter initialization."""
        exporter = CounterfactualExporter()

        assert exporter.plotter is not None
        assert isinstance(exporter.plotter, CausalPlotter)

    def test_initialization_custom(self, tmp_path):
        """Test custom initialization."""
        plotter = CausalPlotter(figsize=(10, 5), save_dir=tmp_path)
        exporter = CounterfactualExporter(plotter=plotter)

        assert exporter.plotter is plotter

    def test_export_single_country(self, tmp_path, sample_time_series):
        """Test exporting single country plot."""
        years, actual, synthetic = sample_time_series

        exporter = CounterfactualExporter(
            plotter=CausalPlotter(save_dir=tmp_path)
        )

        effects = {year: actual[year] - synthetic[year] for year in years}
        method_result = {
            'effect': effects,
            'treatment_year': 2005
        }

        saved_paths = exporter.export_country(
            country='USA',
            actual_values=actual,
            synthetic_values=synthetic,
            method_result=method_result
        )

        assert len(saved_paths) == 2  # Counterfactual + effects
        assert all(p.exists() for p in saved_paths)

    def test_export_all_countries(self, tmp_path):
        """Test exporting all countries."""
        results = CausalResults('Test')
        results.raw_results = {
            'USA': {
                'method_result': {
                    'effect': {2000: 0, 2001: 1, 2002: 2},
                    'treatment_year': 2001,
                    'actual_values': {2000: 3, 2001: 4, 2002: 5},
                    'synthetic_values': {2000: 3, 2001: 3, 2002: 3}
                }
            }
        }

        exporter = CounterfactualExporter(
            plotter=CausalPlotter(save_dir=tmp_path)
        )

        saved_files = exporter.export_all(
            results=results,
            actual_key='actual_values',
            synthetic_key='synthetic_values'
        )

        assert 'USA' in saved_files
        assert len(saved_files['USA']) > 0

    def test_create_summary_grid(self, tmp_path, sample_time_series):
        """Test creating summary grid."""
        years, actual, synthetic = sample_time_series

        countries_data = {
            'USA': {
                'actual': actual,
                'synthetic': synthetic,
                'treatment_year': 2005
            },
            'Canada': {
                'actual': actual,
                'synthetic': {y: v + 0.5 for y, v in synthetic.items()},
                'treatment_year': 2006
            }
        }

        exporter = CounterfactualExporter(
            plotter=CausalPlotter(save_dir=tmp_path)
        )

        fig = exporter.create_summary_grid(countries_data, ncols=2)

        assert fig is not None
        saved_path = tmp_path / 'summary_grid.png'
        fig.savefig(saved_path)
        assert saved_path.exists()

        plt.close(fig)


# ==============================================================================
# EffectDistributionPlotter Tests
# ==============================================================================

class TestEffectDistributionPlotter:
    """Test EffectDistributionPlotter class."""

    def test_initialization(self):
        """Test plotter initialization."""
        plotter = EffectDistributionPlotter()

        assert plotter.plotter is not None
        assert isinstance(plotter.plotter, CausalPlotter)

    def test_plot_att_distribution(self, sample_results):
        """Test ATT distribution plotting."""
        plotter = EffectDistributionPlotter()

        fig, ax = plotter.plot_att_distribution(
            results=sample_results,
            highlight_significant=True
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_plot_att_distribution_no_highlight(self, sample_results):
        """Test ATT distribution without highlighting."""
        plotter = EffectDistributionPlotter()

        fig, ax = plotter.plot_att_distribution(
            results=sample_results,
            highlight_significant=False
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_method_comparison(self):
        """Test method comparison plotting."""
        method_results = {
            'SC': CausalResults('SC'),
            'DiD': CausalResults('DiD')
        }

        # Add same country to both methods
        for method_name, results in method_results.items():
            results.add_country_result('USA', {
                'att': 1.0 + np.random.randn() * 0.2,
                'pre_rmse': 0.5,
                'post_rmse': 1.0,
                'rmse_ratio': 2.0,
                'p_value': 0.05,
                'significant': True
            })

        plotter = EffectDistributionPlotter()

        fig, ax = plotter.plot_method_comparison(
            method_results=method_results,
            country='USA'
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_plot_effect_over_time(self):
        """Test effect over time plotting."""
        effects_by_country = {
            'USA': {2000: 0.1, 2001: 0.5, 2002: 1.0, 2003: 1.5},
            'Canada': {2000: 0.0, 2001: 0.2, 2002: 0.5, 2003: 0.8}
        }
        treatment_years = {'USA': 2001, 'Canada': 2002}

        plotter = EffectDistributionPlotter()

        fig, ax = plotter.plot_effect_over_time(
            effects_by_country=effects_by_country,
            treatment_years=treatment_years
        )

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_plot_rmse_comparison(self, sample_results):
        """Test RMSE comparison plotting."""
        plotter = EffectDistributionPlotter()

        fig, ax = plotter.plot_rmse_comparison(results=sample_results)

        assert fig is not None
        assert ax is not None

        plt.close(fig)

    def test_create_diagnostic_grid(self, tmp_path):
        """Test creating diagnostic grid."""
        results = CausalResults('Test')

        # Add multiple countries
        for i, country in enumerate(['USA', 'Canada', 'UK', 'Germany']):
            results.add_country_result(country, {
                'att': 1.0 + i * 0.3,
                'pre_rmse': 0.5 + i * 0.1,
                'post_rmse': 1.0 + i * 0.2,
                'rmse_ratio': 2.0,
                'p_value': 0.05 * (i + 1),
                'significant': i < 2
            })

        plotter = EffectDistributionPlotter(
            plotter=CausalPlotter(save_dir=tmp_path)
        )

        fig = plotter.create_diagnostic_grid(results)

        assert fig is not None

        # Save and verify
        save_path = tmp_path / 'diagnostics.png'
        fig.savefig(save_path)
        assert save_path.exists()

        plt.close(fig)

    def test_export_distribution_plots(self, tmp_path, sample_results):
        """Test exporting all distribution plots."""
        plotter = EffectDistributionPlotter(
            plotter=CausalPlotter(save_dir=tmp_path)
        )

        saved_files = plotter.export_all_distributions(
            results=sample_results,
            prefix='test'
        )

        assert len(saved_files) > 0
        assert all(p.exists() for p in saved_files)


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestVisualizationIntegration:
    """Test integration between visualization components."""

    def test_full_export_workflow(self, tmp_path, sample_results):
        """Test complete export workflow."""
        # Setup plotter with save directory
        base_plotter = CausalPlotter(save_dir=tmp_path)

        # Export counterfactuals
        cf_exporter = CounterfactualExporter(plotter=base_plotter)

        # Export distributions
        dist_plotter = EffectDistributionPlotter(plotter=base_plotter)
        dist_files = dist_plotter.export_all_distributions(sample_results)

        assert len(dist_files) > 0

    def test_multiple_exporters_same_directory(self, tmp_path):
        """Test multiple exporters using same directory."""
        plotter = CausalPlotter(save_dir=tmp_path)

        exporter1 = CounterfactualExporter(plotter=plotter)
        exporter2 = EffectDistributionPlotter(plotter=plotter)

        # Both should be able to save to same directory
        assert exporter1.plotter.save_dir == tmp_path
        assert exporter2.plotter.save_dir == tmp_path

    def test_style_consistency(self):
        """Test that style is consistent across plotters."""
        base_plotter = CausalPlotter(style='ggplot')

        exporter = CounterfactualExporter(plotter=base_plotter)
        dist_plotter = EffectDistributionPlotter(plotter=base_plotter)

        assert exporter.plotter.style == 'ggplot'
        assert dist_plotter.plotter.style == 'ggplot'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
