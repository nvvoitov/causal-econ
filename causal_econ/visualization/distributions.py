"""
Distribution visualizations for causal inference results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional, Any, Union
import warnings

from ..core.base import CausalResults


def prepare_analysis_data(results: CausalResults, 
                         method_name: str,
                         data_dict: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Prepare standardized analysis data from any causal method results.
    
    Parameters:
    -----------
    results : CausalResults
        Results from any causal method
    method_name : str
        Name of the method for data extraction
    data_dict : dict, optional
        Original data dictionary (needed for some methods)
        
    Returns:
    --------
    dict : Dictionary containing 'growth_df', 'rmse_df', and metadata
    """
    
    growth_data = []
    rmse_data = []
    
    # Get p-values and significance from summary table
    p_values = {}
    significant_countries = []
    if results.summary_table is not None and not results.summary_table.empty:
        p_value_col = next((col for col in results.summary_table.columns 
                           if 'p-val' in col.lower() or 'p_value' in col.lower()), None)
        if p_value_col:
            for _, row in results.summary_table.iterrows():
                country = row['country']
                p_val = row[p_value_col]
                p_values[country] = p_val
                if p_val <= 0.1:
                    significant_countries.append(country)
    
    # Process each country's results
    for country, country_data in results.raw_results.items():
        # Extract method-specific result
        result_key = f"{method_name.lower().replace(' ', '_').replace('-', '_')}_result"
        country_result = None
        
        if isinstance(country_data, dict):
            # Try different possible keys
            for key in [result_key, 'sc_result', 'did_result', 'sdid_result', 'gsc_result', 'ce2v_result']:
                if key in country_data:
                    country_result = country_data[key]
                    break
        else:
            country_result = country_data
            
        if country_result is None:
            continue
            
        # Extract basic info
        treatment_year = country_result.get('treatment_year')
        att = country_result.get('att', np.nan)
        is_positive = not np.isnan(att) and att > 0
        is_significant = country in significant_countries
        
        # Get time series data
        actual_key = next((k for k in ['actual_values', 'treated_values'] if k in country_result), None)
        counterfactual_key = next((k for k in ['counterfactual_values', 'synthetic_values', 'synthetic', 'shifted_control'] if k in country_result), None)
        
        if actual_key and counterfactual_key:
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
            
            # Add growth data
            for i, year in enumerate(years):
                if not np.isnan(actual_vals[i]):
                    period = 'Pre-Intervention' if treatment_year and year < treatment_year else 'Post-Intervention Actual'
                    growth_data.append({
                        'country': country,
                        'year': year,
                        'gdp_growth': actual_vals[i],
                        'period': period,
                        'att_positive': is_positive,
                        'significant': is_significant
                    })
                
                if not np.isnan(counterfactual_vals[i]) and treatment_year and year >= treatment_year:
                    growth_data.append({
                        'country': country,
                        'year': year,
                        'gdp_growth': counterfactual_vals[i],
                        'period': 'Post-Intervention Counterfactual',
                        'att_positive': is_positive,
                        'significant': is_significant
                    })
        
        # Add RMSE data
        pre_rmse = country_result.get('pre_rmse')
        post_rmse = country_result.get('post_rmse')
        rmse_ratio = country_result.get('rmse_ratio')
        
        if pre_rmse is not None:
            rmse_data.append({
                'country': country,
                'rmse': min(pre_rmse, 20),  # Cap at 20
                'period': 'Pre-Intervention RMSE',
                'att_positive': is_positive,
                'significant': is_significant
            })
        
        if post_rmse is not None:
            rmse_data.append({
                'country': country,
                'rmse': min(post_rmse, 20),  # Cap at 20
                'period': 'Post-Intervention RMSE',
                'att_positive': is_positive,
                'significant': is_significant
            })
        
        if rmse_ratio is not None and not np.isnan(rmse_ratio):
            rmse_data.append({
                'country': country,
                'rmse': min(rmse_ratio, 20),  # Cap at 20
                'period': 'RMSE Ratio (Post/Pre)',
                'att_positive': is_positive,
                'significant': is_significant
            })
    
    return {
        'growth_df': pd.DataFrame(growth_data),
        'rmse_df': pd.DataFrame(rmse_data),
        'significant_countries': significant_countries
    }


def create_gdp_distribution_plot(growth_df: pd.DataFrame,
                                title: str = "GDP Growth Distribution",
                                x_range: tuple = (-15, 15)) -> go.Figure:
    """
    Create GDP growth distribution plot with KDE overlays.
    
    Parameters:
    -----------
    growth_df : pd.DataFrame
        Growth data with columns: gdp_growth, period, att_positive, significant
    title : str
        Plot title
    x_range : tuple
        X-axis range
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    
    colors = {
        'Pre-Intervention': 'blue',
        'Post-Intervention Actual': 'red',
        'Post-Intervention Counterfactual': 'green'
    }
    
    # Create histogram
    fig = px.histogram(
        growth_df,
        x='gdp_growth',
        color='period',
        barmode='overlay',
        opacity=0.5,
        histnorm='percent',
        template='plotly_white',
        marginal='box',
        nbins=50,
        height=600,
        width=900,
        color_discrete_map=colors
    )
    
    # Add KDE overlays
    for period in ['Pre-Intervention', 'Post-Intervention Actual', 'Post-Intervention Counterfactual']:
        period_data = growth_df[growth_df['period'] == period]
        if len(period_data) > 1:
            try:
                x_range_kde = np.linspace(min(period_data['gdp_growth']), max(period_data['gdp_growth']), 100)
                kde = gaussian_kde(period_data['gdp_growth'].dropna())
                y_kde = kde(x_range_kde) * 100 * (max(period_data['gdp_growth']) - min(period_data['gdp_growth'])) / 20
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range_kde,
                        y=y_kde,
                        mode='lines',
                        line=dict(width=2, color=colors[period]),
                        name=f"{period} KDE",
                        showlegend=False
                    )
                )
            except Exception:
                # Skip KDE if it fails
                pass
    
    # Update layout
    fig.update_layout(
        title=title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(showgrid=True, showline=True, rangemode="tozero", linecolor='black')
    fig.update_xaxes(range=list(x_range), showgrid=True, showline=True, rangemode="tozero", linecolor='black')
    
    return fig


def create_rmse_distribution_plot(rmse_df: pd.DataFrame,
                                 title: str = "RMSE Distribution") -> go.Figure:
    """
    Create RMSE distribution plot with separate subplots for values and ratios.
    
    Parameters:
    -----------
    rmse_df : pd.DataFrame
        RMSE data with columns: rmse, period, att_positive, significant
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["RMSE Distribution: Pre vs Post Intervention",
                       "RMSE Ratio Distribution (Post/Pre)"],
        vertical_spacing=0.15
    )
    
    # Only include pre and post RMSE in the first plot
    rmse_only_df = rmse_df[rmse_df['period'].isin(['Pre-Intervention RMSE', 'Post-Intervention RMSE'])]
    
    # Colors for RMSE plots
    colors = {'Pre-Intervention RMSE': 'blue', 'Post-Intervention RMSE': 'red'}
    
    for period in ['Pre-Intervention RMSE', 'Post-Intervention RMSE']:
        period_data = rmse_only_df[rmse_only_df['period'] == period]
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=period_data['rmse'],
                name=period,
                opacity=0.6,
                marker_color=colors[period],
                nbinsx=20,
                histnorm='percent'
            ),
            row=1, col=1
        )
        
        # Add KDE overlay
        if len(period_data) > 1:
            try:
                x_range = np.linspace(min(period_data['rmse']), max(period_data['rmse']), 100)
                kde = gaussian_kde(period_data['rmse'].dropna())
                y_kde = kde(x_range) * 100 * (max(period_data['rmse']) - min(period_data['rmse'])) / 20
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_kde,
                        mode='lines',
                        line=dict(width=2, color=colors[period]),
                        name=f"{period} KDE",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            except Exception:
                pass
    
    # Add ratio histogram in the second subplot
    ratio_data = rmse_df[rmse_df['period'] == 'RMSE Ratio (Post/Pre)']
    
    fig.add_trace(
        go.Histogram(
            x=ratio_data['rmse'],
            opacity=0.7,
            marker_color='green',
            nbinsx=20,
            name='RMSE Ratio',
            histnorm='percent'
        ),
        row=1, col=2
    )
    
    # Add KDE for ratio
    if len(ratio_data) > 1:
        try:
            ratio_range = np.linspace(min(ratio_data['rmse']), max(ratio_data['rmse']), 100)
            kde_ratio = gaussian_kde(ratio_data['rmse'].dropna())
            y_kde_ratio = kde_ratio(ratio_range) * 100 * (max(ratio_data['rmse']) - min(ratio_data['rmse'])) / 20
            
            fig.add_trace(
                go.Scatter(
                    x=ratio_range,
                    y=y_kde_ratio,
                    mode='lines',
                    line=dict(width=2, color='darkgreen'),
                    name='Ratio KDE',
                    showlegend=False
                ),
                row=1, col=2
            )
        except Exception:
            pass
    
    # Add reference lines
    fig.add_vline(x=1, line_width=1, line_dash="dash", line_color="black", row=1, col=2)
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black", row=1, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=500,
        width=900,
        template='plotly_white',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="RMSE Value", row=1, col=1, rangemode="tozero", showgrid=True, showline=True, linecolor='black')
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=1, rangemode="tozero", showgrid=True, showline=True, linecolor='black')
    
    fig.update_xaxes(title_text="RMSE Ratio (Post/Pre)", row=1, col=2, rangemode="tozero", showgrid=True, showline=True, linecolor='black')
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=2, rangemode="tozero", showgrid=True, showline=True, linecolor='black')
    
    return fig


def create_method_comparison_distributions(results_dict: Dict[str, CausalResults],
                                          variable: str = 'gdp_growth') -> go.Figure:
    """
    Create comparison distributions across different methods.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to CausalResults objects
    variable : str
        Variable to compare ('gdp_growth' or 'rmse')
        
    Returns:
    --------
    go.Figure : Plotly figure comparing methods
    """
    
    all_data = []
    
    for method_name, results in results_dict.items():
        analysis_data = prepare_analysis_data(results, method_name)
        
        if variable == 'gdp_growth':
            df = analysis_data['growth_df']
            df['method'] = method_name
            all_data.append(df)
        elif variable == 'rmse':
            df = analysis_data['rmse_df']
            df['method'] = method_name
            all_data.append(df)
    
    if not all_data:
        return go.Figure()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if variable == 'gdp_growth':
        fig = px.histogram(
            combined_df,
            x='gdp_growth',
            color='method',
            facet_col='period',
            barmode='overlay',
            opacity=0.6,
            histnorm='percent',
            template='plotly_white',
            title="GDP Growth Distribution by Method"
        )
    else:
        fig = px.histogram(
            combined_df,
            x='rmse',
            color='method',
            facet_col='period',
            barmode='overlay',
            opacity=0.6,
            histnorm='percent',
            template='plotly_white',
            title="RMSE Distribution by Method"
        )
    
    return fig


def export_distribution_plots(results: CausalResults,
                            method_name: str,
                            output_dir: str,
                            data_dict: Optional[Dict] = None):
    """
    Export distribution plots for a method.
    
    Parameters:
    -----------
    results : CausalResults
        Results object
    method_name : str
        Name of the method
    output_dir : str
        Output directory
    data_dict : dict, optional
        Original data dictionary
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    analysis_data = prepare_analysis_data(results, method_name, data_dict)
    
    # Create and save GDP plot
    gdp_fig = create_gdp_distribution_plot(
        analysis_data['growth_df'],
        title=f"{method_name} GDP Growth Distribution"
    )
    gdp_fig.write_image(
        os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_gdp_kde.png"),
        scale=2
    )
    
    # Create and save RMSE plot
    rmse_fig = create_rmse_distribution_plot(
        analysis_data['rmse_df'],
        title=f"{method_name} RMSE Distribution"
    )
    rmse_fig.write_image(
        os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_rmse_kde.png"),
        scale=2
    )
    
    print(f"Distribution plots exported to {output_dir}")
