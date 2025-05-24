"""
Standardized visualization tools for counterfactual analysis and treatment effects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Union, Tuple
import warnings


def plot_counterfactual_comparison(country_result: Dict,
                                 placebo_result: Optional[Dict] = None,
                                 figsize: Tuple[int, int] = (14, 6),
                                 confidence_level: float = 0.95,
                                 method_name: str = "Method") -> plt.Figure:
    """
    Create standardized counterfactual comparison plot for any method.
    
    Parameters:
    -----------
    country_result : dict
        Results for a specific country containing:
        - 'country': Country name
        - 'treatment_year': Treatment year
        - 'actual_values' or equivalent: Actual outcome values
        - 'counterfactual_values', 'synthetic_values', etc.: Counterfactual values
        - 'effect': Treatment effects
        - 'att': Average treatment effect
    placebo_result : dict, optional
        Placebo test results with p-values
    figsize : tuple
        Figure size
    confidence_level : float
        Confidence level for intervals
    method_name : str
        Name of the method for labeling
        
    Returns:
    --------
    plt.Figure : Matplotlib figure
    """
    country = country_result['country']
    treatment_year = country_result['treatment_year']
    
    # Extract data flexibly based on available keys
    actual_key = next((k for k in ['actual_values', 'treated_values'] if k in country_result), None)
    counterfactual_key = next((k for k in ['counterfactual_values', 'synthetic_values', 'synthetic', 'shifted_control'] if k in country_result), None)
    
    if not actual_key or not counterfactual_key:
        raise ValueError("Could not find actual and counterfactual values in country_result")
    
    actual_values = country_result[actual_key]
    counterfactual_values = country_result[counterfactual_key]
    
    # Handle different data formats
    if isinstance(actual_values, dict):
        years = sorted(actual_values.keys())
        actual_vals = [actual_values.get(year, np.nan) for year in years]
        counterfactual_vals = [counterfactual_values.get(year, np.nan) for year in years]
    else:
        # Assume array-like with years inferred
        years = list(range(len(actual_values)))
        actual_vals = actual_values
        counterfactual_vals = counterfactual_values
    
    # Calculate effects
    effect_values = [a - c if not (np.isnan(a) or np.isnan(c)) else np.nan 
                    for a, c in zip(actual_vals, counterfactual_vals)]
    
    # Create masks for pre and post treatment
    pre_treat_mask = np.array(years) < treatment_year
    post_treat_mask = np.array(years) >= treatment_year
    
    # Calculate cumulative effect
    cumulative_effect = np.zeros_like(years, dtype=float)
    if any(post_treat_mask):
        post_effects = [e for e, m in zip(effect_values, post_treat_mask) if m and not np.isnan(e)]
        cum_idx = np.where(post_treat_mask)[0]
        if len(post_effects) > 0 and len(cum_idx) > 0:
            cumulative_effect[cum_idx] = np.cumsum(post_effects)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Outcome Comparison
    ax1.plot(years, actual_vals, 'b-', linewidth=2, label='Actual')
    ax1.plot(years, counterfactual_vals, 'r--', linewidth=2, label=f'{method_name} Counterfactual')
    
    # Add treatment year line
    if treatment_year:
        ax1.axvline(x=treatment_year, color='black', linestyle=':', linewidth=2,
                   label=f'Treatment ({treatment_year})')
    
    # Calculate average effect after treatment
    if any(post_treat_mask):
        post_effect_values = [e for e, m in zip(effect_values, post_treat_mask) if m and not np.isnan(e)]
        if post_effect_values:
            avg_effect = np.mean(post_effect_values)
            ax1.set_title(f'Outcome for {country}\nAverage Effect: {avg_effect:.4f}')
        else:
            ax1.set_title(f'Outcome for {country}')
    else:
        ax1.set_title(f'Outcome for {country}')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Outcome')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Cumulative Effect with confidence intervals
    ax2.plot(years, cumulative_effect, 'g-', linewidth=2)
    
    # Add treatment year line
    if treatment_year:
        ax2.axvline(x=treatment_year, color='black', linestyle=':', linewidth=2,
                   label=f'Treatment ({treatment_year})')
    
    # Add confidence intervals if we have pre-treatment data
    if any(pre_treat_mask) and any(post_treat_mask):
        pre_effect_values = [e for e, m in zip(effect_values, pre_treat_mask) if m and not np.isnan(e)]
        if pre_effect_values:
            pre_treat_std = np.std(pre_effect_values)
            
            # For confidence interval
            z_score = 1.96 if confidence_level == 0.95 else 1.65
            
            # CI grows with the square root of time since treatment
            time_since_treatment = np.zeros_like(years, dtype=float)
            time_indices = np.arange(1, np.sum(post_treat_mask) + 1)
            time_since_treatment[post_treat_mask] = time_indices
            
            # Calculate confidence width
            conf_width = z_score * pre_treat_std * np.sqrt(time_since_treatment)
            
            # Calculate bounds
            upper_ci = cumulative_effect + conf_width
            lower_ci = cumulative_effect - conf_width
            
            # Plot confidence interval
            ax2.fill_between(years, lower_ci, upper_ci, color='g', alpha=0.2)
    
    # Add horizontal line at 0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add title with total effect and p-value
    if any(post_treat_mask):
        last_effect = cumulative_effect[-1] if len(cumulative_effect) > 0 else 0
        p_value = placebo_result.get('p_value_ratio', None) if placebo_result else None
        p_value_str = f", p-value: {p_value:.3f}" if p_value is not None else ""
        ax2.set_title(f'Cumulative Impact for {country}\nTotal Effect: {last_effect:.4f}{p_value_str}')
    else:
        ax2.set_title(f'Cumulative Impact for {country}')
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Effect')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_placebo_comparison(country_result: Dict,
                          placebo_result: Dict,
                          figsize: Tuple[int, int] = (14, 8),
                          method_name: str = "Method") -> Optional[plt.Figure]:
    """
    Create standardized placebo comparison plot.
    
    Parameters:
    -----------
    country_result : dict
        Results for the treated country
    placebo_result : dict
        Placebo test results
    figsize : tuple
        Figure size
    method_name : str
        Name of the method
        
    Returns:
    --------
    plt.Figure : Matplotlib figure or None if no placebo data
    """
    if not placebo_result or not placebo_result.get('placebos'):
        return None
    
    country = country_result['country']
    treatment_year = country_result['treatment_year']
    placebos = placebo_result['placebos']
    
    # Extract effect data for treated country
    effect_key = 'effect'
    if effect_key not in country_result:
        return None
    
    effect_data = country_result[effect_key]
    
    # Handle different data formats
    if isinstance(effect_data, dict):
        years = sorted(effect_data.keys())
        effect_values = [effect_data.get(year, np.nan) for year in years]
    else:
        years = list(range(len(effect_data)))
        effect_values = effect_data
    
    # Create mask for pre and post treatment
    pre_treat_mask = np.array(years) < treatment_year
    post_treat_mask = np.array(years) >= treatment_year
    
    # Calculate cumulative effect for treated unit
    cumulative_effect = np.zeros_like(years, dtype=float)
    if any(post_treat_mask):
        post_effects = [e for e, m in zip(effect_values, post_treat_mask) if m and not np.isnan(e)]
        cum_idx = np.where(post_treat_mask)[0]
        if len(post_effects) > 0 and len(cum_idx) > 0:
            cumulative_effect[cum_idx] = np.cumsum(post_effects)
    
    cumulative_effect[~post_treat_mask] = 0  # Set pre-treatment values to 0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Treatment effects
    for placebo in placebos:
        placebo_effect = placebo.get('effect', {})
        if isinstance(placebo_effect, dict):
            placebo_years = sorted(placebo_effect.keys())
            placebo_effects = [placebo_effect.get(year, np.nan) for year in placebo_years]
        else:
            placebo_years = years  # Assume same years
            placebo_effects = placebo_effect
        
        ax1.plot(placebo_years, placebo_effects, 'gray', alpha=0.3)
    
    # Plot the treated unit last and with a bold line
    ax1.plot(years, effect_values, 'b-', linewidth=2.5, label=country)
    
    # Add treatment year line
    if treatment_year:
        ax1.axvline(x=treatment_year, color='black', linestyle=':', linewidth=2,
                   label=f'Treatment ({treatment_year})')
    
    # Add horizontal line at 0
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax1.set_title(f'Treatment Effect: {country} vs. Placebos ({method_name})')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Effect')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative effects
    for placebo in placebos:
        placebo_effect = placebo.get('effect', {})
        
        if isinstance(placebo_effect, dict):
            placebo_years = sorted(placebo_effect.keys())
            placebo_effects = [placebo_effect.get(year, np.nan) for year in placebo_years]
        else:
            placebo_years = years
            placebo_effects = placebo_effect
        
        # Calculate cumulative effect for placebo
        placebo_pre_mask = np.array(placebo_years) < treatment_year
        placebo_post_mask = np.array(placebo_years) >= treatment_year
        
        placebo_cumulative = np.zeros_like(placebo_years, dtype=float)
        if any(placebo_post_mask):
            placebo_post_effects = [e for e, m in zip(placebo_effects, placebo_post_mask) 
                                  if m and not np.isnan(e)]
            p_cum_idx = np.where(placebo_post_mask)[0]
            if len(placebo_post_effects) > 0 and len(p_cum_idx) > 0:
                placebo_cumulative[p_cum_idx] = np.cumsum(placebo_post_effects)
        
        placebo_cumulative[~placebo_post_mask] = 0
        
        ax2.plot(placebo_years, placebo_cumulative, 'gray', alpha=0.3)
    
    # Plot the treated unit last and with a bold line
    ax2.plot(years, cumulative_effect, 'b-', linewidth=2.5, label=country)
    
    # Add treatment year line
    if treatment_year:
        ax2.axvline(x=treatment_year, color='black', linestyle=':', linewidth=2,
                   label=f'Treatment ({treatment_year})')
    
    # Add horizontal line at 0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add p-value to title
    p_value = placebo_result.get('p_value_ratio', np.nan)
    p_value_str = f" (p-value: {p_value:.3f})" if not np.isnan(p_value) else ""
    
    ax2.set_title(f'Cumulative Effect: {country} vs. Placebos{p_value_str}')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Effect')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_interactive_counterfactual_plot(country_result: Dict,
                                         placebo_result: Optional[Dict] = None,
                                         method_name: str = "Method") -> go.Figure:
    """
    Create interactive counterfactual plot using Plotly.
    
    Parameters:
    -----------
    country_result : dict
        Results for a specific country
    placebo_result : dict, optional
        Placebo test results
    method_name : str
        Name of the method
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    country = country_result['country']
    treatment_year = country_result['treatment_year']
    
    # Extract data
    actual_key = next((k for k in ['actual_values', 'treated_values'] if k in country_result), None)
    counterfactual_key = next((k for k in ['counterfactual_values', 'synthetic_values', 'synthetic', 'shifted_control'] if k in country_result), None)
    
    if not actual_key or not counterfactual_key:
        raise ValueError("Could not find actual and counterfactual values")
    
    actual_values = country_result[actual_key]
    counterfactual_values = country_result[counterfactual_key]
    
    # Handle different data formats
    if isinstance(actual_values, dict):
        years = sorted(actual_values.keys())
        actual_vals = [actual_values.get(year, np.nan) for year in years]
        counterfactual_vals = [counterfactual_values.get(year, np.nan) for year in years]
    else:
        years = list(range(len(actual_values)))
        actual_vals = actual_values
        counterfactual_vals = counterfactual_values
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Outcome Comparison: {country}", f"Treatment Effect: {country}"],
        horizontal_spacing=0.1
    )
    
    # Plot actual values
    fig.add_trace(
        go.Scatter(
            x=years,
            y=actual_vals,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            hovertemplate='Year: %{x}<br>Actual: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot counterfactual values
    fig.add_trace(
        go.Scatter(
            x=years,
            y=counterfactual_vals,
            mode='lines+markers',
            name=f'{method_name} Counterfactual',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Year: %{x}<br>Counterfactual: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add treatment year line
    if treatment_year:
        fig.add_vline(
            x=treatment_year,
            line=dict(color='black', width=2, dash='dot'),
            annotation_text=f"Treatment ({treatment_year})",
            row=1, col=1
        )
        fig.add_vline(
            x=treatment_year,
            line=dict(color='black', width=2, dash='dot'),
            row=1, col=2
        )
    
    # Calculate and plot treatment effects
    effect_values = [a - c if not (np.isnan(a) or np.isnan(c)) else np.nan 
                    for a, c in zip(actual_vals, counterfactual_vals)]
    
    fig.add_trace(
        go.Scatter(
            x=years,
            y=effect_values,
            mode='lines+markers',
            name='Treatment Effect',
            line=dict(color='green', width=2),
            hovertemplate='Year: %{x}<br>Effect: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add horizontal line at 0 for effects plot
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title=f"{method_name} Analysis: {country}",
        template='plotly_white',
        height=500,
        width=1200,
        showlegend=True
    )
    
    # Add p-value annotation if available
    if placebo_result and 'p_value_ratio' in placebo_result:
        p_value = placebo_result['p_value_ratio']
        if not np.isnan(p_value):
            fig.add_annotation(
                text=f"p-value: {p_value:.3f}",
                xref="paper", yref="paper",
                x=0.75, y=0.95,
                showarrow=False,
                font=dict(size=12)
            )
    
    return fig


def create_treatment_effects_dashboard(results_dict: Dict[str, Dict],
                                     method_name: str = "Method") -> go.Figure:
    """
    Create a dashboard showing treatment effects for multiple countries.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping country names to their results
    method_name : str
        Name of the method
        
    Returns:
    --------
    go.Figure : Multi-country dashboard
    """
    countries = list(results_dict.keys())
    n_countries = len(countries)
    
    # Calculate grid dimensions
    cols = min(3, n_countries)
    rows = (n_countries + cols - 1) // cols
    
    # Create subplot titles
    subplot_titles = [f"{country}" for country in countries]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, (country, country_result) in enumerate(results_dict.items()):
        row = i // cols + 1
        col = i % cols + 1
        
        # Extract effect data
        effect_key = 'effect'
        if effect_key not in country_result:
            continue
        
        effect_data = country_result[effect_key]
        treatment_year = country_result.get('treatment_year')
        
        # Handle different data formats
        if isinstance(effect_data, dict):
            years = sorted(effect_data.keys())
            effect_values = [effect_data.get(year, np.nan) for year in years]
        else:
            years = list(range(len(effect_data)))
            effect_values = effect_data
        
        # Add treatment effect line
        fig.add_trace(
            go.Scatter(
                x=years,
                y=effect_values,
                mode='lines',
                name=country,
                line=dict(width=2),
                showlegend=False,
                hovertemplate=f'{country}<br>Year: %{{x}}<br>Effect: %{{y:.3f}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add treatment year line
        if treatment_year:
            fig.add_vline(
                x=treatment_year,
                line=dict(color='red', width=1, dash='dot'),
                row=row, col=col
            )
        
        # Add horizontal line at 0
        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'), row=row, col=col)
    
    fig.update_layout(
        title=f"{method_name} Treatment Effects Dashboard",
        template='plotly_white',
        height=300 * rows,
        width=400 * cols
    )
    
    return fig


def export_counterfactual_plots(results: Any,
                               output_dir: str,
                               format: str = 'png',
                               dpi: int = 300):
    """
    Export counterfactual plots for all countries in results.
    
    Parameters:
    -----------
    results : CausalResults or dict
        Results object or dictionary
    output_dir : str
        Output directory for plots
    format : str
        Output format ('png', 'pdf', 'svg')
    dpi : int
        DPI for raster formats
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract method name and results
    if hasattr(results, 'method_name'):
        method_name = results.method_name
        country_results = results.raw_results
    else:
        method_name = "Method"
        country_results = results
    
    for country, country_data in country_results.items():
        try:
            # Extract country result and placebo result
            if isinstance(country_data, dict):
                # Find the main result key
                result_key = None
                for key in ['sc_result', 'did_result', 'sdid_result', 'gsc_result', 'ce2v_result']:
                    if key in country_data:
                        result_key = key
                        break
                
                if result_key:
                    country_result = country_data[result_key]
                    placebo_result = country_data.get('placebo_result')
                else:
                    country_result = country_data
                    placebo_result = None
            else:
                country_result = country_data
                placebo_result = None
            
            # Create main plot
            fig_main = plot_counterfactual_comparison(
                country_result, placebo_result, method_name=method_name
            )
            fig_main.suptitle(f"{method_name}: {country}", fontsize=16)
            
            # Save main plot
            main_filename = f"{country}_main.{format}"
            fig_main.savefig(
                os.path.join(output_dir, main_filename),
                dpi=dpi, bbox_inches='tight'
            )
            plt.close(fig_main)
            
            # Create placebo plot if available
            if placebo_result and placebo_result.get('placebos'):
                fig_placebo = plot_placebo_comparison(
                    country_result, placebo_result, method_name=method_name
                )
                if fig_placebo:
                    fig_placebo.suptitle(f"{method_name} Placebo: {country}", fontsize=16)
                    
                    # Save placebo plot
                    placebo_filename = f"{country}_placebo.{format}"
                    fig_placebo.savefig(
                        os.path.join(output_dir, placebo_filename),
                        dpi=dpi, bbox_inches='tight'
                    )
                    plt.close(fig_placebo)
            
        except Exception as e:
            warnings.warn(f"Failed to create plots for {country}: {e}")
    
    print(f"Plots exported to {output_dir}")
