"""
Impact analysis and table generation for causal inference results.

This module provides backward-compatible wrappers around ImpactAnalyzer.
"""

import pandas as pd
from typing import Dict, Optional
from pathlib import Path

from ..core.base import CausalResults
from .utilities import ImpactAnalyzer


def create_impact_table(results: CausalResults,
                       method_name: str,
                       data_dict: Optional[Dict] = None,
                       significance_threshold: float = 0.1) -> pd.DataFrame:
    """
    Create impact table showing effects grouped by positive/negative ATT.

    REFACTORED: Now uses ImpactAnalyzer.create_impact_table().

    Parameters:
    -----------
    results : CausalResults
        Results from causal analysis
    method_name : str
        Name of the method
    data_dict : dict, optional
        Original data dictionary (not used in refactored version)
    significance_threshold : float
        Threshold for significance

    Returns:
    --------
    DataFrame : Impact table
    """
    analyzer = ImpactAnalyzer(significance_threshold=significance_threshold)
    return analyzer.create_impact_table(results)


def export_impact_tables(results_dict: Dict[str, CausalResults],
                        output_dir: str,
                        format: str = 'csv',
                        significance_threshold: float = 0.1) -> Dict[str, str]:
    """
    Export impact tables for all methods.

    REFACTORED: Uses ImpactAnalyzer for exports.

    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
    output_dir : str
        Output directory
    format : str
        Export format ('csv', 'excel')
    significance_threshold : float
        Threshold for significance

    Returns:
    --------
    dict : Dictionary mapping method names to file paths
    """
    analyzer = ImpactAnalyzer(significance_threshold=significance_threshold)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for method_name, results in results_dict.items():
        filename = f"{method_name.lower().replace(' ', '_')}_impact.{format}"
        file_path = output_path / filename

        analyzer.export_impact_table(
            results=results,
            output_path=file_path
        )

        saved_files[method_name] = str(file_path)

    return saved_files
