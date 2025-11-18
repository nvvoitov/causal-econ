"""
Distribution visualization and comparison tools.

This module provides backward-compatible wrappers around Effect DistributionPlotter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import warnings

from ..core.base import CausalResults
from .plotter import CausalPlotter, EffectDistributionPlotter


def prepare_analysis_data(results: CausalResults,
                         data_dict: Dict,
                         outcome_col: str = 'gdp_growth_annual_share') -> Dict[str, pd.DataFrame]:
    """
    Prepare data for distribution analysis and visualization.

    Parameters:
    -----------
    results : CausalResults
        Results object from analysis
    data_dict : dict
        Data dictionary with panel data
    outcome_col : str
        Name of outcome column

    Returns:
    --------
    dict : Dictionary with prepared DataFrames for visualization
    """
    panel = data_dict['panel']
    treatment_years = data_dict['treatment_years']

    # Prepare growth distribution dataframe
    growth_data = []
    for country in results.country_results.keys():
        if country not in panel.columns:
            continue

        treatment_year = treatment_years.get(country)
        for year, value in panel[country].items():
            if not pd.isna(value):
                growth_data.append({
                    'country': country,
                    'year': year,
                    'value': value,
                    'period': 'Pre' if year < treatment_year else 'Post',
                    'treatment_year': treatment_year
                })

    growth_df = pd.DataFrame(growth_data)

    # Prepare RMSE dataframe
    rmse_data = []
    for country, metrics in results.country_results.items():
        if 'pre_rmse' in metrics and 'post_rmse' in metrics:
            rmse_data.append({
                'country': country,
                'pre_rmse': metrics['pre_rmse'],
                'post_rmse': metrics['post_rmse'],
                'rmse_ratio': metrics.get('rmse_ratio', np.nan)
            })

    rmse_df = pd.DataFrame(rmse_data)

    return {
        'growth_df': growth_df,
        'rmse_df': rmse_df
    }


def create_gdp_distribution_plot(growth_df: pd.DataFrame,
                                 figsize: Tuple[int, int] = (14, 6),
                                 title: Optional[str] = None) -> plt.Figure:
    """
    Create distribution plots for GDP growth before and after treatment.

    SIMPLIFIED: Uses standard matplotlib patterns.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pre-treatment distribution
    pre_data = growth_df[growth_df['period'] == 'Pre']['value'].dropna()
    axes[0].hist(pre_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(pre_data.mean(), color='red', linestyle='--',
                   label=f'Mean: {pre_data.mean():.2f}')
    axes[0].set_xlabel('GDP Growth Rate')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Pre-Treatment Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Post-treatment distribution
    post_data = growth_df[growth_df['period'] == 'Post']['value'].dropna()
    axes[1].hist(post_data, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(post_data.mean(), color='red', linestyle='--',
                   label=f'Mean: {post_data.mean():.2f}')
    axes[1].set_xlabel('GDP Growth Rate')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Post-Treatment Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    return fig


def create_rmse_distribution_plot(rmse_df: pd.DataFrame,
                                  figsize: Tuple[int, int] = (14, 10),
                                  title: Optional[str] = None) -> plt.Figure:
    """
    Create RMSE distribution and comparison plots.

    REFACTORED: Uses EffectDistributionPlotter for consistent styling.
    """
    plotter = EffectDistributionPlotter(plotter=CausalPlotter(figsize=figsize))

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Pre-treatment RMSE
    axes[0, 0].hist(rmse_df['pre_rmse'].dropna(), bins=20, alpha=0.7,
                   color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Pre-treatment RMSE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Pre-Treatment RMSE Distribution')
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Post-treatment RMSE
    axes[0, 1].hist(rmse_df['post_rmse'].dropna(), bins=20, alpha=0.7,
                   color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Post-treatment RMSE')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Post-Treatment RMSE Distribution')
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: RMSE Ratio
    axes[1, 0].hist(rmse_df['rmse_ratio'].dropna(), bins=20, alpha=0.7,
                   color='purple', edgecolor='black')
    axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Ratio = 1')
    axes[1, 0].set_xlabel('RMSE Ratio (Post/Pre)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('RMSE Ratio Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Scatter plot
    axes[1, 1].scatter(rmse_df['pre_rmse'], rmse_df['post_rmse'], alpha=0.6)
    axes[1, 1].plot([0, rmse_df['pre_rmse'].max()],
                   [0, rmse_df['pre_rmse'].max()],
                   'r--', label='y=x')
    axes[1, 1].set_xlabel('Pre-treatment RMSE')
    axes[1, 1].set_ylabel('Post-treatment RMSE')
    axes[1, 1].set_title('RMSE Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, y=1.0)

    plt.tight_layout()
    return fig


def create_method_comparison_distributions(results_dict: Dict[str, CausalResults],
                                          figsize: Tuple[int, int] = (14, 8),
                                          metric: str = 'att') -> plt.Figure:
    """
    Create distribution comparison plots across different methods.

    REFACTORED: Uses EffectDistributionPlotter.
    """
    plotter = EffectDistributionPlotter(plotter=CausalPlotter(figsize=figsize))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Extract data for each method
    method_data = {}
    for method_name, results in results_dict.items():
        values = []
        for country, metrics in results.country_results.items():
            if metric in metrics and not np.isnan(metrics[metric]):
                values.append(metrics[metric])
        method_data[method_name] = values

    # Plot 1: Overlapping histograms
    for method_name, values in method_data.items():
        axes[0].hist(values, bins=20, alpha=0.5, label=method_name, edgecolor='black')

    axes[0].set_xlabel(metric.upper())
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{metric.upper()} Distribution by Method')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Box plots
    box_data = [values for values in method_data.values()]
    box_labels = list(method_data.keys())

    axes[1].boxplot(box_data, labels=box_labels)
    axes[1].set_ylabel(metric.upper())
    axes[1].set_title(f'{metric.upper()} Comparison Across Methods')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def export_distribution_plots(results: CausalResults,
                             data_dict: Dict,
                             output_dir: str,
                             method_name: str = "Method",
                             format: str = 'png') -> Dict[str, str]:
    """
    Export distribution plots to files.

    REFACTORED: Uses EffectDistributionPlotter for batch export.

    Parameters:
    -----------
    results : CausalResults
        Results object
    data_dict : dict
        Data dictionary
    output_dir : str
        Output directory
    method_name : str
        Method name for labeling
    format : str
        Output format

    Returns:
    --------
    dict : Dictionary mapping plot names to file paths
    """
    # Initialize plotter
    plotter = CausalPlotter(save_dir=output_dir)
    dist_plotter = EffectDistributionPlotter(plotter=plotter)

    # Export all distribution plots
    saved_files = dist_plotter.export_all_distributions(
        results=results,
        prefix=method_name.lower().replace(' ', '_')
    )

    # Also create growth and RMSE distributions
    analysis_data = prepare_analysis_data(results, data_dict)

    # GDP distribution
    gdp_fig = create_gdp_distribution_plot(
        analysis_data['growth_df'],
        title=f'{method_name}: GDP Growth Distribution'
    )
    gdp_path = plotter.save(f'{method_name.lower()}_gdp_distribution.{format}', fig=gdp_fig)
    saved_files.append(gdp_path)

    # RMSE distribution
    rmse_fig = create_rmse_distribution_plot(
        analysis_data['rmse_df'],
        title=f'{method_name}: RMSE Analysis'
    )
    rmse_path = plotter.save(f'{method_name.lower()}_rmse_distribution.{format}', fig=rmse_fig)
    saved_files.append(rmse_path)

    # Create dictionary mapping
    saved_dict = {
        'att_distribution': saved_files[0] if len(saved_files) > 0 else None,
        'rmse_comparison': saved_files[1] if len(saved_files) > 1 else None,
        'gdp_distribution': gdp_path,
        'rmse_distribution': rmse_path
    }

    return saved_dict
