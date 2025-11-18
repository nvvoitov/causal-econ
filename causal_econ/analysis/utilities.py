"""
Analysis utility classes for causal inference results.

This module provides classes for summary generation, impact analysis,
and comparison across methods.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path


class SummaryGenerator:
    """
    Generate summary tables and statistics from causal analysis results.

    Example:
    --------
    >>> generator = SummaryGenerator()
    >>> summary = generator.create_standardized_summary(results)
    >>> generator.export_to_excel(summary, 'summary.xlsx')
    """

    def __init__(self):
        """Initialize summary generator."""
        pass

    def create_standardized_summary(self,
                                   results: 'CausalResults',
                                   sort_by: str = 'p_value',
                                   ascending: bool = True) -> pd.DataFrame:
        """
        Create standardized summary table.

        Parameters:
        -----------
        results : CausalResults
            Results object
        sort_by : str, optional
            Column to sort by (default: 'p_value')
        ascending : bool, optional
            Sort ascending (default: True)

        Returns:
        --------
        pd.DataFrame : Standardized summary table
        """
        if results.summary_table is not None:
            return results.summary_table

        # Build from country_results
        rows = []
        for country, metrics in results.country_results.items():
            rows.append({
                'country': country,
                'method': results.method_name,
                'ATT': metrics.get('att', np.nan),
                'RMSE_pre': metrics.get('pre_rmse', np.nan),
                'RMSE_post': metrics.get('post_rmse', np.nan),
                'RMSE_ratio': metrics.get('rmse_ratio', np.nan),
                'p_value': metrics.get('p_value', np.nan),
                'significant': metrics.get('significant', False)
            })

        df = pd.DataFrame(rows)

        if not df.empty and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

        return df

    def combine_method_summaries(self,
                                method_results: Dict[str, 'CausalResults'],
                                metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Combine summaries from multiple methods.

        Parameters:
        -----------
        method_results : dict
            Dictionary mapping method names to CausalResults
        metrics : list, optional
            Metrics to include (default: all)

        Returns:
        --------
        pd.DataFrame : Combined summary table

        Example:
        --------
        >>> combined = generator.combine_method_summaries({
        ...     'SC': sc_results,
        ...     'DiD': did_results,
        ...     'SDiD': sdid_results
        ... })
        """
        combined_data = []

        for method_name, results in method_results.items():
            summary = self.create_standardized_summary(results)
            summary['method'] = method_name
            combined_data.append(summary)

        if not combined_data:
            return pd.DataFrame()

        combined_df = pd.concat(combined_data, ignore_index=True)

        # Pivot for comparison
        if metrics is None:
            metrics = ['ATT', 'RMSE_ratio', 'p_value']

        pivot_tables = {}
        for metric in metrics:
            if metric in combined_df.columns:
                pivot = combined_df.pivot(
                    index='country',
                    columns='method',
                    values=metric
                )
                pivot_tables[metric] = pivot

        return combined_df, pivot_tables

    def export_to_excel(self,
                       summary: pd.DataFrame,
                       filename: Union[str, Path],
                       sheet_name: str = 'Summary'):
        """
        Export summary to Excel.

        Parameters:
        -----------
        summary : pd.DataFrame
            Summary table
        filename : str or Path
            Output filename
        sheet_name : str, optional
            Sheet name (default: 'Summary')
        """
        summary.to_excel(filename, sheet_name=sheet_name, index=False)
        print(f"Exported summary to {filename}")


class ImpactAnalyzer:
    """
    Analyze treatment impacts and categorize results.

    Example:
    --------
    >>> analyzer = ImpactAnalyzer()
    >>> impact_table = analyzer.create_impact_table(results)
    """

    def __init__(self,
                 significance_threshold: float = 0.1,
                 effect_threshold: float = 0.0):
        """
        Initialize impact analyzer.

        Parameters:
        -----------
        significance_threshold : float, optional
            P-value threshold for significance (default: 0.1)
        effect_threshold : float, optional
            Minimum effect size to consider (default: 0.0)
        """
        self.significance_threshold = significance_threshold
        self.effect_threshold = effect_threshold

    def create_impact_table(self,
                           results: 'CausalResults',
                           categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create impact table categorizing countries by effect type.

        Parameters:
        -----------
        results : CausalResults
            Results object
        categories : list, optional
            Custom categories (default: ['Positive Significant', 'Negative Significant', etc.])

        Returns:
        --------
        pd.DataFrame : Impact table

        Example:
        --------
        >>> impact = analyzer.create_impact_table(results)
        >>> print(impact.groupby('category').size())
        """
        rows = []

        for country, metrics in results.country_results.items():
            att = metrics.get('att', np.nan)
            p_value = metrics.get('p_value', np.nan)

            # Categorize
            if pd.isna(att) or pd.isna(p_value):
                category = 'Inconclusive'
            elif p_value <= self.significance_threshold:
                if att > self.effect_threshold:
                    category = 'Positive Significant'
                elif att < -self.effect_threshold:
                    category = 'Negative Significant'
                else:
                    category = 'No Effect (Significant)'
            else:
                if att > self.effect_threshold:
                    category = 'Positive Insignificant'
                elif att < -self.effect_threshold:
                    category = 'Negative Insignificant'
                else:
                    category = 'No Effect (Insignificant)'

            rows.append({
                'country': country,
                'category': category,
                'ATT': att,
                'p_value': p_value,
                'significant': p_value <= self.significance_threshold if not pd.isna(p_value) else False
            })

        impact_df = pd.DataFrame(rows)
        impact_df = impact_df.sort_values(['category', 'ATT'], ascending=[True, False])

        return impact_df

    def get_impact_summary(self,
                          impact_table: pd.DataFrame) -> Dict:
        """
        Get summary statistics from impact table.

        Parameters:
        -----------
        impact_table : pd.DataFrame
            Impact table from create_impact_table()

        Returns:
        --------
        dict : Summary statistics

        Example:
        --------
        >>> summary = analyzer.get_impact_summary(impact_table)
        >>> print(f"Positive significant: {summary['positive_significant']}")
        """
        category_counts = impact_table['category'].value_counts().to_dict()

        return {
            'total_countries': len(impact_table),
            'positive_significant': category_counts.get('Positive Significant', 0),
            'negative_significant': category_counts.get('Negative Significant', 0),
            'insignificant': (
                category_counts.get('Positive Insignificant', 0) +
                category_counts.get('Negative Insignificant', 0) +
                category_counts.get('No Effect (Insignificant)', 0)
            ),
            'inconclusive': category_counts.get('Inconclusive', 0),
            'mean_att_positive': impact_table[impact_table['ATT'] > 0]['ATT'].mean(),
            'mean_att_negative': impact_table[impact_table['ATT'] < 0]['ATT'].mean(),
            'category_breakdown': category_counts
        }


class MethodComparator:
    """
    Compare results across different causal methods.

    Example:
    --------
    >>> comparator = MethodComparator()
    >>> comparison = comparator.compare_methods({
    ...     'SC': sc_results,
    ...     'DiD': did_results,
    ...     'SDiD': sdid_results
    ... })
    """

    def __init__(self):
        """Initialize method comparator."""
        pass

    def compare_methods(self,
                       method_results: Dict[str, 'CausalResults'],
                       metrics: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Compare multiple methods side-by-side.

        Parameters:
        -----------
        method_results : dict
            Dictionary mapping method names to results
        metrics : list, optional
            Metrics to compare (default: ['ATT', 'p_value'])

        Returns:
        --------
        dict : Dictionary of comparison DataFrames by metric

        Example:
        --------
        >>> comparison = comparator.compare_methods(method_results)
        >>> print(comparison['ATT'])  # ATT comparison across methods
        """
        if metrics is None:
            metrics = ['ATT', 'p_value', 'RMSE_ratio']

        comparisons = {}

        for metric in metrics:
            # Extract metric for each method
            method_data = {}

            for method_name, results in method_results.items():
                values = {}
                for country, country_metrics in results.country_results.items():
                    val = country_metrics.get(metric.lower())
                    if val is not None:
                        values[country] = val
                method_data[method_name] = values

            # Create DataFrame
            comp_df = pd.DataFrame(method_data)
            comp_df.index.name = 'country'

            comparisons[metric] = comp_df

        return comparisons

    def calculate_agreement(self,
                           method_results: Dict[str, 'CausalResults'],
                           significance_threshold: float = 0.1) -> pd.DataFrame:
        """
        Calculate agreement on significance across methods.

        Parameters:
        -----------
        method_results : dict
            Method results
        significance_threshold : float, optional
            Threshold for significance (default: 0.1)

        Returns:
        --------
        pd.DataFrame : Agreement table

        Example:
        --------
        >>> agreement = comparator.calculate_agreement(method_results)
        >>> print(agreement['agreement_rate'])
        """
        # Get all countries
        all_countries = set()
        for results in method_results.values():
            all_countries.update(results.country_results.keys())

        rows = []

        for country in all_countries:
            significance_calls = []

            for method_name, results in method_results.items():
                if country in results.country_results:
                    p_value = results.country_results[country].get('p_value')
                    if p_value is not None and not np.isnan(p_value):
                        significance_calls.append(p_value <= significance_threshold)

            if significance_calls:
                agreement_rate = sum(significance_calls) / len(significance_calls)
                consensus = 'Yes' if agreement_rate >= 0.5 else 'No'

                rows.append({
                    'country': country,
                    'n_methods': len(significance_calls),
                    'n_significant': sum(significance_calls),
                    'agreement_rate': agreement_rate,
                    'consensus_significant': consensus
                })

        agreement_df = pd.DataFrame(rows)
        agreement_df = agreement_df.sort_values('agreement_rate', ascending=False)

        return agreement_df


class ResultExporter:
    """
    Export results in various formats.

    Example:
    --------
    >>> exporter = ResultExporter()
    >>> exporter.export_comprehensive_report(
    ...     results, output_dir='results/', format='excel'
    ... )
    """

    def __init__(self):
        """Initialize result exporter."""
        self.summary_generator = SummaryGenerator()
        self.impact_analyzer = ImpactAnalyzer()

    def export_comprehensive_report(self,
                                   results: 'CausalResults',
                                   output_dir: Union[str, Path],
                                   format: str = 'excel'):
        """
        Export comprehensive report with all tables.

        Parameters:
        -----------
        results : CausalResults
            Results object
        output_dir : str or Path
            Output directory
        format : str, optional
            Format: 'excel', 'csv' (default: 'excel')

        Exports:
        --------
        - Summary table
        - Impact table
        - Raw results (if available)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        method_name = results.method_name.replace(' ', '_').lower()

        # Summary table
        summary = self.summary_generator.create_standardized_summary(results)

        # Impact table
        impact = self.impact_analyzer.create_impact_table(results)

        if format == 'excel':
            # Single Excel file with multiple sheets
            filename = output_dir / f'{method_name}_report.xlsx'

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                summary.to_excel(writer, sheet_name='Summary', index=False)
                impact.to_excel(writer, sheet_name='Impact Analysis', index=False)

                # Add impact summary
                impact_summary = self.impact_analyzer.get_impact_summary(impact)
                impact_summary_df = pd.DataFrame([impact_summary])
                impact_summary_df.to_excel(writer, sheet_name='Impact Summary', index=False)

            print(f"Exported comprehensive Excel report to {filename}")

        elif format == 'csv':
            # Multiple CSV files
            summary.to_csv(output_dir / f'{method_name}_summary.csv', index=False)
            impact.to_csv(output_dir / f'{method_name}_impact.csv', index=False)

            print(f"Exported CSV reports to {output_dir}")

    def export_method_comparison(self,
                                method_results: Dict[str, 'CausalResults'],
                                output_file: Union[str, Path]):
        """
        Export comparison across methods.

        Parameters:
        -----------
        method_results : dict
            Dictionary of method results
        output_file : str or Path
            Output Excel file

        Example:
        --------
        >>> exporter.export_method_comparison(
        ...     {'SC': sc_results, 'DiD': did_results},
        ...     'method_comparison.xlsx'
        ... )
        """
        comparator = MethodComparator()
        comparisons = comparator.compare_methods(method_results)
        agreement = comparator.calculate_agreement(method_results)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Write each metric comparison
            for metric, comp_df in comparisons.items():
                comp_df.to_excel(writer, sheet_name=f'{metric}_Comparison')

            # Write agreement table
            agreement.to_excel(writer, sheet_name='Agreement', index=False)

            # Write combined summary
            generator = SummaryGenerator()
            combined, _ = generator.combine_method_summaries(method_results)
            combined.to_excel(writer, sheet_name='Combined_Summary', index=False)

        print(f"Exported method comparison to {output_file}")
