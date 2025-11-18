"""
Summary table generation and export utilities.

This module provides backward-compatible wrappers around SummaryGenerator.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

from ..core.base import CausalResults
from .utilities import SummaryGenerator


def combine_method_summaries(results_dict: Dict[str, CausalResults],
                            sort_by: str = 'p-val',
                            ascending: bool = True) -> pd.DataFrame:
    """
    Combine summary tables from multiple methods into one DataFrame.

    REFACTORED: Now uses SummaryGenerator.combine_method_summaries().

    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
    sort_by : str
        Column to sort by
    ascending : bool
        Sort order

    Returns:
    --------
    DataFrame : Combined summary table with method column
    """
    generator = SummaryGenerator()
    combined = generator.combine_method_summaries(results_dict)

    if sort_by in combined.columns:
        combined = combined.sort_values(sort_by, ascending=ascending)

    return combined


def export_summary_tables(results_dict: Dict[str, CausalResults],
                         output_dir: str,
                         format: str = 'csv') -> Dict[str, str]:
    """
    Export summary tables for all methods.

    REFACTORED: Uses SummaryGenerator for exports.

    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
    output_dir : str
        Output directory
    format : str
        Export format ('csv', 'excel', 'latex', 'html')

    Returns:
    --------
    dict : Dictionary mapping method names to file paths
    """
    generator = SummaryGenerator()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for method_name, results in results_dict.items():
        filename = f"{method_name.lower().replace(' ', '_')}_summary.{format}"
        file_path = output_path / filename

        generator.export_summary(
            results=results,
            output_path=file_path,
            format=format
        )

        saved_files[method_name] = str(file_path)

    # Also export combined summary
    combined = combine_method_summaries(results_dict)
    combined_filename = f"combined_summary.{format}"
    combined_path = output_path / combined_filename

    if format == 'csv':
        combined.to_csv(combined_path, index=False)
    elif format == 'excel':
        combined.to_excel(combined_path, index=False)
    else:
        combined.to_csv(combined_path, index=False)

    saved_files['combined'] = str(combined_path)

    return saved_files
