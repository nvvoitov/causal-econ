"""
Standardized visualization tools for counterfactual analysis and treatment effects.

This module provides backward-compatible wrappers around the CausalPlotter
and CounterfactualExporter classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Union, Tuple
import warnings

from .plotter import CausalPlotter, CounterfactualExporter


def plot_counterfactual_comparison(country_result: Dict,
                                 placebo_result: Optional[Dict] = None,
                                 figsize: Tuple[int, int] = (14, 6),
                                 confidence_level: float = 0.95,
                                 method_name: str = "Method") -> plt.Figure:
    """
    Create standardized counterfactual comparison plot for any method.

    REFACTORED: Now uses CausalPlotter for cleaner implementation.
    """
    plotter = CausalPlotter(figsize=figsize)

    country = country_result['country']
    treatment_year = country_result['treatment_year']

    # Extract actual and counterfactual values
    actual_key = next((k for k in ['actual_values', 'treated_values'] if k in country_result), None)
    counterfactual_key = next((k for k in ['counterfactual_values', 'synthetic_values',
                                           'synthetic', 'shifted_control'] if k in country_result), None)

    if not actual_key or not counterfactual_key:
        raise ValueError("Could not find actual and counterfactual values in country_result")

    actual_values = country_result[actual_key]
    counterfactual_values = country_result[counterfactual_key]

    # Plot counterfactual
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Use plotter for main counterfactual plot
    plotter.plot_counterfactual(
        actual_values=actual_values,
        synthetic_values=counterfactual_values,
        treatment_year=treatment_year,
        country=country,
        ax=axes[0],
        title=f'{method_name}: {country}'
    )

    # Plot treatment effects on second axis
    if 'effect' in country_result:
        plotter.plot_treatment_effects(
            effects=country_result['effect'],
            treatment_year=treatment_year,
            country=country,
            ax=axes[1],
            title=f'Treatment Effects: {country}'
        )

        # Add ATT if available
        if 'att' in country_result:
            axes[1].axhline(y=country_result['att'], color='red', linestyle='--',
                          label=f'ATT: {country_result["att"]:.3f}')
            axes[1].legend()

    # Add p-value if placebo result provided
    if placebo_result and 'p_value_ratio' in placebo_result:
        p_val = placebo_result['p_value_ratio']
        if not np.isnan(p_val):
            fig.text(0.5, 0.02, f'Placebo p-value: {p_val:.3f}',
                    ha='center', fontsize=12)

    plt.tight_layout()
    return fig


def plot_placebo_comparison(country_result: Dict,
                           placebo_results: List[Dict],
                           figsize: Tuple[int, int] = (14, 10),
                           metric: str = 'rmse_ratio') -> plt.Figure:
    """
    Plot comparison of actual result with placebo test results.

    REFACTORED: Uses CausalPlotter for plotting.
    """
    plotter = CausalPlotter(figsize=figsize)

    country = country_result['country']

    # Extract metric values
    actual_metric = country_result.get(metric, np.nan)
    placebo_metrics = [p.get(metric, np.nan) for p in placebo_results
                      if not np.isnan(p.get(metric, np.nan))]

    if not placebo_metrics:
        warnings.warn(f"No valid placebo results for {metric}")
        return plt.figure(figsize=figsize)

    # Calculate p-value
    if not np.isnan(actual_metric):
        p_value = sum(1 for p in placebo_metrics if p >= actual_metric) / len(placebo_metrics)
    else:
        p_value = np.nan

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot 1: Placebo distribution
    plotter.plot_placebo_distribution(
        placebo_effects=placebo_metrics,
        true_effect=actual_metric,
        country=country,
        p_value=p_value,
        ax=ax1,
        title=f'Placebo Test: {country} ({metric})'
    )

    # Plot 2: Rank plot
    all_metrics = placebo_metrics + [actual_metric]
    sorted_metrics = sorted(all_metrics, reverse=True)
    ranks = list(range(1, len(sorted_metrics) + 1))

    colors = ['red' if m == actual_metric else 'gray' for m in sorted_metrics]
    ax2.bar(ranks, sorted_metrics, color=colors)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel(metric)
    ax2.set_title(f'Ranking Among Placebo Tests')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_interactive_counterfactual_plot(country_result: Dict,
                                          title: Optional[str] = None,
                                          method_name: str = "Method") -> go.Figure:
    """
    Create interactive Plotly counterfactual plot.

    Note: This function still uses Plotly directly as CausalPlotter
    focuses on matplotlib. Can be extended in future.
    """
    country = country_result['country']
    treatment_year = country_result['treatment_year']

    # Extract data
    actual_key = next((k for k in ['actual_values', 'treated_values'] if k in country_result), None)
    counterfactual_key = next((k for k in ['counterfactual_values', 'synthetic_values',
                                           'synthetic', 'shifted_control'] if k in country_result), None)

    if not actual_key or not counterfactual_key:
        raise ValueError("Could not find actual and counterfactual values")

    actual_values = country_result[actual_key]
    counterfactual_values = country_result[counterfactual_key]

    # Convert to lists
    if isinstance(actual_values, dict):
        years = sorted(actual_values.keys())
        actual_vals = [actual_values[y] for y in years]
        counterfactual_vals = [counterfactual_values[y] for y in years]
    else:
        years = list(range(len(actual_values)))
        actual_vals = list(actual_values)
        counterfactual_vals = list(counterfactual_values)

    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=years, y=actual_vals,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=years, y=counterfactual_vals,
        mode='lines+markers',
        name=f'{method_name} Counterfactual',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Add treatment line
    fig.add_vline(x=treatment_year, line_dash="dot", line_color="black",
                 annotation_text=f"Treatment ({treatment_year})")

    # Update layout
    fig.update_layout(
        title=title or f'{method_name} Analysis: {country}',
        xaxis_title='Year',
        yaxis_title='Outcome',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_treatment_effects_dashboard(results_dict: Dict[str, Dict],
                                       method_name: str = "Method",
                                       top_n: int = 10) -> go.Figure:
    """
    Create interactive dashboard with treatment effects for multiple countries.

    Simplified version using plotly directly.
    """
    # Extract countries and effects
    countries_data = []
    for country, data in results_dict.items():
        if 'att' in data and 'p_value' in data:
            countries_data.append({
                'country': country,
                'att': data['att'],
                'p_value': data['p_value'],
                'significant': data.get('significant', False)
            })

    df = pd.DataFrame(countries_data)
    if df.empty:
        return go.Figure()

    # Sort by absolute ATT
    df['abs_att'] = df['att'].abs()
    df = df.sort_values('abs_att', ascending=False).head(top_n)

    # Create figure
    fig = go.Figure()

    # Add bar chart
    colors = ['green' if sig else 'gray' for sig in df['significant']]

    fig.add_trace(go.Bar(
        x=df['country'],
        y=df['att'],
        marker_color=colors,
        text=df['p_value'].round(3),
        textposition='outside',
        name='ATT'
    ))

    fig.update_layout(
        title=f'{method_name}: Top {top_n} Countries by Treatment Effect',
        xaxis_title='Country',
        yaxis_title='Average Treatment Effect (ATT)',
        template='plotly_white',
        showlegend=False
    )

    return fig


def export_counterfactual_plots(results: Any,
                               output_dir: str,
                               method_name: str = "Method",
                               format: str = 'png',
                               actual_key: str = 'actual_values',
                               counterfactual_key: str = 'synthetic_values') -> Dict[str, List[str]]:
    """
    Export counterfactual plots for all countries in results.

    REFACTORED: Now uses CounterfactualExporter for batch export.

    Parameters:
    -----------
    results : CausalResults or dict
        Results object or dictionary with country results
    output_dir : str
        Output directory for plots
    method_name : str
        Method name for labeling
    format : str
        Output format (png, pdf, etc.)
    actual_key : str
        Key for actual values in results
    counterfactual_key : str
        Key for counterfactual values in results

    Returns:
    --------
    dict : Dictionary mapping country names to list of saved file paths
    """
    # Initialize exporter
    plotter = CausalPlotter(save_dir=output_dir)
    exporter = CounterfactualExporter(plotter=plotter)

    # Export all countries
    saved_files = exporter.export_all(
        results=results,
        actual_key=actual_key,
        synthetic_key=counterfactual_key
    )

    return saved_files
