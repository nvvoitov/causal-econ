"""
Unit tests for analysis utility classes.

Tests SummaryGenerator, ImpactAnalyzer, MethodComparator, and ResultExporter classes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from causal_econ.analysis.utilities import (
    SummaryGenerator,
    ImpactAnalyzer,
    MethodComparator,
    ResultExporter
)
from causal_econ.core.base import CausalResults


@pytest.fixture
def sample_results():
    """Create sample causal results."""
    results = CausalResults('Synthetic Control')

    countries_data = [
        ('USA', 1.5, 0.5, 1.0, 2.0, 0.03, True),
        ('Canada', -0.5, 0.4, 0.9, 2.25, 0.15, False),
        ('UK', 2.0, 0.6, 1.2, 2.0, 0.02, True),
        ('Germany', 0.2, 0.5, 1.1, 2.2, 0.20, False)
    ]

    for country, att, pre_rmse, post_rmse, rmse_ratio, p_val, sig in countries_data:
        results.add_country_result(country, {
            'att': att,
            'pre_rmse': pre_rmse,
            'post_rmse': post_rmse,
            'rmse_ratio': rmse_ratio,
            'p_value': p_val,
            'significant': sig
        })

    # Create summary table
    results.summary_table = pd.DataFrame([
        {
            'country': country,
            'ATT': att,
            'RMSE pre': pre_rmse,
            'RMSE post': post_rmse,
            'relation': rmse_ratio,
            'p-val': p_val,
            'significant': 'Yes' if sig else 'No'
        }
        for country, att, pre_rmse, post_rmse, rmse_ratio, p_val, sig in countries_data
    ])

    return results


@pytest.fixture
def multiple_method_results():
    """Create results from multiple methods."""
    methods = {}

    for method_name in ['SC', 'DiD', 'SDiD']:
        results = CausalResults(method_name)

        for country in ['USA', 'Canada', 'UK']:
            # Different ATT values for each method
            att = np.random.randn() * 0.5 + 1.0
            results.add_country_result(country, {
                'att': att,
                'pre_rmse': 0.5,
                'post_rmse': 1.0,
                'rmse_ratio': 2.0,
                'p_value': 0.05 if method_name == 'SC' else 0.15,
                'significant': method_name == 'SC'
            })

        methods[method_name] = results

    return methods


# ==============================================================================
# SummaryGenerator Tests
# ==============================================================================

class TestSummaryGenerator:
    """Test SummaryGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = SummaryGenerator()
        assert generator is not None

    def test_create_summary_table(self, sample_results):
        """Test creating summary table."""
        generator = SummaryGenerator()

        summary = generator.create_summary_table(sample_results)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 4
        assert 'country' in summary.columns
        assert 'ATT' in summary.columns
        assert 'p-val' in summary.columns

    def test_create_summary_with_sorting(self, sample_results):
        """Test summary table with sorting."""
        generator = SummaryGenerator()

        summary = generator.create_summary_table(
            sample_results,
            sort_by='ATT',
            ascending=False
        )

        # Check that ATT is sorted descending
        att_values = summary['ATT'].values
        assert all(att_values[i] >= att_values[i+1]
                  for i in range(len(att_values)-1))

    def test_format_summary_latex(self, sample_results):
        """Test LaTeX formatting."""
        generator = SummaryGenerator()

        latex_str = generator.format_summary(
            sample_results,
            format='latex'
        )

        assert isinstance(latex_str, str)
        assert '\\begin{tabular}' in latex_str
        assert '\\end{tabular}' in latex_str

    def test_format_summary_html(self, sample_results):
        """Test HTML formatting."""
        generator = SummaryGenerator()

        html_str = generator.format_summary(
            sample_results,
            format='html'
        )

        assert isinstance(html_str, str)
        assert '<table' in html_str
        assert '</table>' in html_str

    def test_export_summary_csv(self, tmp_path, sample_results):
        """Test exporting summary to CSV."""
        generator = SummaryGenerator()

        output_file = tmp_path / 'summary.csv'
        generator.export_summary(
            sample_results,
            output_path=output_file,
            format='csv'
        )

        assert output_file.exists()

        # Read and verify
        df = pd.read_csv(output_file)
        assert len(df) == 4

    def test_export_summary_excel(self, tmp_path, sample_results):
        """Test exporting summary to Excel."""
        generator = SummaryGenerator()

        output_file = tmp_path / 'summary.xlsx'
        generator.export_summary(
            sample_results,
            output_path=output_file,
            format='excel'
        )

        assert output_file.exists()

    def test_combine_method_summaries(self, multiple_method_results):
        """Test combining summaries from multiple methods."""
        generator = SummaryGenerator()

        combined = generator.combine_method_summaries(multiple_method_results)

        assert isinstance(combined, pd.DataFrame)
        assert 'method' in combined.columns
        assert 'country' in combined.columns

        # Should have results from all methods
        assert set(combined['method'].unique()) == {'SC', 'DiD', 'SDiD'}


# ==============================================================================
# ImpactAnalyzer Tests
# ==============================================================================

class TestImpactAnalyzer:
    """Test ImpactAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ImpactAnalyzer()
        assert analyzer.significance_threshold == 0.1

    def test_initialization_custom_threshold(self):
        """Test initialization with custom threshold."""
        analyzer = ImpactAnalyzer(significance_threshold=0.05)
        assert analyzer.significance_threshold == 0.05

    def test_categorize_impact(self):
        """Test impact categorization."""
        analyzer = ImpactAnalyzer()

        # Positive significant
        assert analyzer._categorize_impact(1.5, 0.03, True) == 'Positive Significant'

        # Negative significant
        assert analyzer._categorize_impact(-1.5, 0.03, True) == 'Negative Significant'

        # Positive insignificant
        assert analyzer._categorize_impact(0.5, 0.15, False) == 'Positive Insignificant'

        # Negative insignificant
        assert analyzer._categorize_impact(-0.5, 0.15, False) == 'Negative Insignificant'

        # Near zero
        assert analyzer._categorize_impact(0.01, 0.5, False) == 'No Effect'

    def test_create_impact_table(self, sample_results):
        """Test creating impact table."""
        analyzer = ImpactAnalyzer()

        impact_table = analyzer.create_impact_table(sample_results)

        assert isinstance(impact_table, pd.DataFrame)
        assert 'country' in impact_table.columns
        assert 'ATT' in impact_table.columns
        assert 'impact_category' in impact_table.columns

        # Check categorizations
        usa_row = impact_table[impact_table['country'] == 'USA'].iloc[0]
        assert usa_row['impact_category'] == 'Positive Significant'

        canada_row = impact_table[impact_table['country'] == 'Canada'].iloc[0]
        assert canada_row['impact_category'] == 'Negative Insignificant'

    def test_get_impact_summary(self, sample_results):
        """Test getting impact summary statistics."""
        analyzer = ImpactAnalyzer()

        summary = analyzer.get_impact_summary(sample_results)

        assert isinstance(summary, dict)
        assert 'n_positive_significant' in summary
        assert 'n_negative_significant' in summary
        assert 'n_insignificant' in summary
        assert 'mean_att_positive_sig' in summary

        # Check counts
        assert summary['n_positive_significant'] == 2  # USA, UK
        assert summary['n_negative_significant'] == 0

    def test_filter_by_impact(self, sample_results):
        """Test filtering results by impact category."""
        analyzer = ImpactAnalyzer()

        positive_sig = analyzer.filter_by_impact(
            sample_results,
            category='Positive Significant'
        )

        assert len(positive_sig) == 2
        assert set(positive_sig['country']) == {'USA', 'UK'}

    def test_export_impact_table(self, tmp_path, sample_results):
        """Test exporting impact table."""
        analyzer = ImpactAnalyzer()

        output_file = tmp_path / 'impact.csv'
        analyzer.export_impact_table(
            sample_results,
            output_path=output_file
        )

        assert output_file.exists()

        # Read and verify
        df = pd.read_csv(output_file)
        assert 'impact_category' in df.columns


# ==============================================================================
# MethodComparator Tests
# ==============================================================================

class TestMethodComparator:
    """Test MethodComparator class."""

    def test_initialization(self):
        """Test comparator initialization."""
        comparator = MethodComparator()
        assert comparator is not None

    def test_compare_methods(self, multiple_method_results):
        """Test comparing multiple methods."""
        comparator = MethodComparator()

        comparison = comparator.compare_methods(multiple_method_results)

        assert isinstance(comparison, pd.DataFrame)
        assert 'country' in comparison.columns

        # Should have ATT columns for each method
        assert 'SC_ATT' in comparison.columns
        assert 'DiD_ATT' in comparison.columns
        assert 'SDiD_ATT' in comparison.columns

    def test_compare_methods_include_significance(self, multiple_method_results):
        """Test comparison with significance indicators."""
        comparator = MethodComparator()

        comparison = comparator.compare_methods(
            multiple_method_results,
            include_significance=True
        )

        # Should have significance columns
        assert 'SC_significant' in comparison.columns
        assert 'DiD_significant' in comparison.columns

    def test_calculate_agreement(self, multiple_method_results):
        """Test calculating agreement between methods."""
        comparator = MethodComparator()

        agreement = comparator.calculate_agreement(multiple_method_results)

        assert isinstance(agreement, dict)
        assert 'pairwise_correlation' in agreement
        assert 'significance_agreement' in agreement

        # Check correlation structure
        corr = agreement['pairwise_correlation']
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (3, 3)  # 3 methods

    def test_rank_methods_by_fit(self, multiple_method_results):
        """Test ranking methods by fit quality."""
        comparator = MethodComparator()

        ranking = comparator.rank_methods_by_fit(
            multiple_method_results,
            country='USA'
        )

        assert isinstance(ranking, pd.DataFrame)
        assert 'method' in ranking.columns
        assert 'rmse_ratio' in ranking.columns
        assert len(ranking) == 3

    def test_identify_consensus(self, multiple_method_results):
        """Test identifying consensus results."""
        comparator = MethodComparator()

        consensus = comparator.identify_consensus(
            multiple_method_results,
            min_agreement=2
        )

        assert isinstance(consensus, list)
        # Countries where at least 2 methods agree on significance

    def test_export_comparison(self, tmp_path, multiple_method_results):
        """Test exporting comparison table."""
        comparator = MethodComparator()

        output_file = tmp_path / 'comparison.csv'
        comparator.export_comparison(
            multiple_method_results,
            output_path=output_file
        )

        assert output_file.exists()


# ==============================================================================
# ResultExporter Tests
# ==============================================================================

class TestResultExporter:
    """Test ResultExporter class."""

    def test_initialization(self, tmp_path):
        """Test exporter initialization."""
        exporter = ResultExporter(output_dir=tmp_path)
        assert exporter.output_dir == tmp_path

    def test_export_single_method(self, tmp_path, sample_results):
        """Test exporting single method results."""
        exporter = ResultExporter(output_dir=tmp_path)

        files = exporter.export_single_method(
            results=sample_results,
            method_name='SC'
        )

        assert isinstance(files, dict)
        assert 'summary_csv' in files
        assert 'impact_csv' in files

        # Verify files exist
        assert all(Path(f).exists() for f in files.values())

    def test_export_all_methods(self, tmp_path, multiple_method_results):
        """Test exporting all methods."""
        exporter = ResultExporter(output_dir=tmp_path)

        files = exporter.export_all_methods(multiple_method_results)

        assert isinstance(files, dict)
        assert 'SC' in files
        assert 'DiD' in files
        assert 'comparison' in files

    def test_create_markdown_report(self, tmp_path, sample_results):
        """Test creating markdown report."""
        exporter = ResultExporter(output_dir=tmp_path)

        report_path = exporter.create_markdown_report(
            results=sample_results,
            method_name='SC'
        )

        assert report_path.exists()
        assert report_path.suffix == '.md'

        # Read and verify content
        content = report_path.read_text()
        assert '# Synthetic Control Analysis Results' in content
        assert 'USA' in content

    def test_create_comprehensive_report(self, tmp_path, multiple_method_results):
        """Test creating comprehensive multi-method report."""
        exporter = ResultExporter(output_dir=tmp_path)

        report_path = exporter.create_comprehensive_report(
            method_results=multiple_method_results,
            title='Causal Analysis Report'
        )

        assert report_path.exists()

        # Read and verify
        content = report_path.read_text()
        assert 'Causal Analysis Report' in content
        assert 'Synthetic Control' in content or 'SC' in content

    def test_export_to_excel_workbook(self, tmp_path, multiple_method_results):
        """Test exporting to Excel workbook with multiple sheets."""
        exporter = ResultExporter(output_dir=tmp_path)

        excel_path = exporter.export_to_excel_workbook(
            method_results=multiple_method_results,
            filename='results.xlsx'
        )

        assert excel_path.exists()
        assert excel_path.suffix == '.xlsx'

        # Read and verify sheets
        xl_file = pd.ExcelFile(excel_path)
        assert 'SC' in xl_file.sheet_names
        assert 'Comparison' in xl_file.sheet_names


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestAnalysisUtilitiesIntegration:
    """Test integration of analysis utilities."""

    def test_full_analysis_workflow(self, tmp_path, multiple_method_results):
        """Test complete analysis workflow."""
        # Generate summaries
        summary_gen = SummaryGenerator()
        combined_summary = summary_gen.combine_method_summaries(
            multiple_method_results
        )
        assert len(combined_summary) > 0

        # Analyze impact
        impact_analyzer = ImpactAnalyzer()
        impact_summary = impact_analyzer.get_impact_summary(
            multiple_method_results['SC']
        )
        assert impact_summary is not None

        # Compare methods
        comparator = MethodComparator()
        comparison = comparator.compare_methods(multiple_method_results)
        assert len(comparison) > 0

        # Export everything
        exporter = ResultExporter(output_dir=tmp_path)
        files = exporter.export_all_methods(multiple_method_results)

        assert len(files) > 0

    def test_chained_analysis(self, sample_results):
        """Test chaining analysis operations."""
        # Generate summary
        summary_gen = SummaryGenerator()
        summary = summary_gen.create_summary_table(sample_results)

        # Analyze impact
        impact_analyzer = ImpactAnalyzer()
        impact_table = impact_analyzer.create_impact_table(sample_results)

        # Filter significant results
        significant = impact_analyzer.filter_by_impact(
            sample_results,
            category='Positive Significant'
        )

        assert len(significant) > 0
        assert all(significant['impact_category'] == 'Positive Significant')

    def test_multi_format_export(self, tmp_path, sample_results):
        """Test exporting in multiple formats."""
        summary_gen = SummaryGenerator()

        # Export as CSV
        csv_path = tmp_path / 'summary.csv'
        summary_gen.export_summary(sample_results, csv_path, format='csv')
        assert csv_path.exists()

        # Export as Excel
        excel_path = tmp_path / 'summary.xlsx'
        summary_gen.export_summary(sample_results, excel_path, format='excel')
        assert excel_path.exists()

        # Export as LaTeX
        latex_str = summary_gen.format_summary(sample_results, format='latex')
        assert isinstance(latex_str, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
