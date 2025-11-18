"""
Unified visualization framework for causal inference results.

This module provides comprehensive plotting classes that eliminate duplicate
visualization code across all methods.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings


class CausalPlotter:
    """
    Unified plotting class for causal inference visualizations.

    Provides standardized plotting methods for:
    - Counterfactual comparisons
    - Treatment effect distributions
    - Placebo test results
    - Weight distributions
    - Pre/post trend comparisons

    Example:
    --------
    >>> plotter = CausalPlotter(figsize=(12, 6), style='seaborn')
    >>> plotter.plot_counterfactual(
    ...     actual_values=actual,
    ...     synthetic_values=synthetic,
    ...     treatment_year=2002,
    ...     country='USA'
    ... )
    >>> plotter.save('usa_counterfactual.png')
    """

    def __init__(self,
                 figsize: Tuple[int, int] = (12, 6),
                 style: str = 'seaborn-v0_8',
                 dpi: int = 100,
                 font_size: int = 10):
        """
        Initialize the plotter.

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) (default: (12, 6))
        style : str, optional
            Matplotlib style (default: 'seaborn-v0_8')
        dpi : int, optional
            DPI for saved figures (default: 100)
        font_size : int, optional
            Base font size (default: 10)
        """
        self.figsize = figsize
        self.dpi = dpi
        self.font_size = font_size

        # Set style
        try:
            plt.style.use(style)
        except:
            warnings.warn(f"Style '{style}' not available, using default")

        # Set default font sizes
        plt.rcParams.update({
            'font.size': font_size,
            'axes.titlesize': font_size + 2,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'legend.fontsize': font_size - 1
        })

        self.current_fig = None
        self.current_ax = None

    def plot_counterfactual(self,
                           actual_values: Union[Dict, pd.Series],
                           synthetic_values: Union[Dict, pd.Series],
                           treatment_year: int,
                           country: str,
                           title: Optional[str] = None,
                           ylabel: str = 'Outcome',
                           show_effect: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot actual vs synthetic/counterfactual comparison.

        Parameters:
        -----------
        actual_values : dict or Series
            Actual outcome values by year
        synthetic_values : dict or Series
            Synthetic/counterfactual values by year
        treatment_year : int
            Year of treatment
        country : str
            Country name for title
        title : str, optional
            Custom title
        ylabel : str, optional
            Y-axis label (default: 'Outcome')
        show_effect : bool, optional
            Show shaded effect region (default: True)

        Returns:
        --------
        tuple : (figure, axes)

        Example:
        --------
        >>> fig, ax = plotter.plot_counterfactual(
        ...     actual_values={2000: 5.0, 2001: 5.2, 2002: 6.5},
        ...     synthetic_values={2000: 5.1, 2001: 5.3, 2002: 5.4},
        ...     treatment_year=2002,
        ...     country='USA'
        ... )
        >>> plt.show()
        """
        # Convert to pandas Series if dict
        if isinstance(actual_values, dict):
            actual_values = pd.Series(actual_values).sort_index()
        if isinstance(synthetic_values, dict):
            synthetic_values = pd.Series(synthetic_values).sort_index()

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Get common years
        common_years = actual_values.index.intersection(synthetic_values.index)
        years = sorted(common_years)

        # Plot lines
        ax.plot(years, actual_values.loc[years], 'o-',
                label='Actual', color='#2E86AB', linewidth=2, markersize=6)
        ax.plot(years, synthetic_values.loc[years], 's--',
                label='Synthetic/Counterfactual', color='#A23B72',
                linewidth=2, markersize=5)

        # Treatment line
        ax.axvline(x=treatment_year, color='red', linestyle=':',
                  linewidth=2, alpha=0.7, label='Treatment')

        # Shaded effect region (post-treatment)
        if show_effect and treatment_year in years:
            post_years = [y for y in years if y >= treatment_year]
            if len(post_years) > 0:
                ax.fill_between(
                    post_years,
                    actual_values.loc[post_years],
                    synthetic_values.loc[post_years],
                    alpha=0.2,
                    color='gray',
                    label='Treatment Effect'
                )

        # Labels and title
        ax.set_xlabel('Year', fontsize=self.font_size)
        ax.set_ylabel(ylabel, fontsize=self.font_size)

        if title is None:
            title = f'{country}: Actual vs Synthetic/Counterfactual'
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')

        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        self.current_fig = fig
        self.current_ax = ax

        return fig, ax

    def plot_treatment_effects(self,
                              effect_values: Union[Dict, pd.Series],
                              treatment_year: int,
                              country: str,
                              title: Optional[str] = None,
                              ylabel: str = 'Treatment Effect',
                              show_zero_line: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot treatment effects over time.

        Parameters:
        -----------
        effect_values : dict or Series
            Effect values (actual - synthetic) by year
        treatment_year : int
            Treatment year
        country : str
            Country name
        title : str, optional
            Custom title
        ylabel : str, optional
            Y-axis label
        show_zero_line : bool, optional
            Show horizontal line at zero (default: True)

        Returns:
        --------
        tuple : (figure, axes)
        """
        if isinstance(effect_values, dict):
            effect_values = pd.Series(effect_values).sort_index()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        years = sorted(effect_values.index)
        effects = effect_values.loc[years]

        # Color by pre/post treatment
        colors = ['#3A86FF' if y < treatment_year else '#FF006E' for y in years]

        # Bar plot
        bars = ax.bar(years, effects, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Treatment line
        ax.axvline(x=treatment_year, color='red', linestyle=':', linewidth=2, alpha=0.7)

        # Zero line
        if show_zero_line:
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # Labels
        ax.set_xlabel('Year', fontsize=self.font_size)
        ax.set_ylabel(ylabel, fontsize=self.font_size)

        if title is None:
            title = f'{country}: Treatment Effects Over Time'
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')

        # Legend
        pre_patch = mpatches.Patch(color='#3A86FF', alpha=0.7, label='Pre-treatment')
        post_patch = mpatches.Patch(color='#FF006E', alpha=0.7, label='Post-treatment')
        ax.legend(handles=[pre_patch, post_patch], loc='best')

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        self.current_fig = fig
        self.current_ax = ax

        return fig, ax

    def plot_placebo_distribution(self,
                                 treated_metric: float,
                                 placebo_metrics: List[float],
                                 metric_name: str = 'RMSE Ratio',
                                 country: str = '',
                                 p_value: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot placebo test distribution.

        Parameters:
        -----------
        treated_metric : float
            Metric value for treated unit
        placebo_metrics : list
            Metric values for placebo units
        metric_name : str, optional
            Name of metric (default: 'RMSE Ratio')
        country : str, optional
            Country name
        p_value : float, optional
            P-value to display

        Returns:
        --------
        tuple : (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Histogram of placebo metrics
        ax.hist(placebo_metrics, bins=20, alpha=0.6, color='#8338EC',
                edgecolor='black', label='Placebo Distribution')

        # Treated unit line
        ax.axvline(x=treated_metric, color='#FF006E', linestyle='--',
                  linewidth=3, label=f'Treated: {treated_metric:.3f}')

        # Labels
        ax.set_xlabel(metric_name, fontsize=self.font_size)
        ax.set_ylabel('Frequency', fontsize=self.font_size)

        title = f'Placebo Test Distribution: {country}' if country else 'Placebo Test Distribution'
        if p_value is not None:
            title += f'\np-value = {p_value:.3f}'
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')

        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        self.current_fig = fig
        self.current_ax = ax

        return fig, ax

    def plot_weights(self,
                    weights: Dict[str, float],
                    country: str,
                    title: Optional[str] = None,
                    threshold: float = 0.01,
                    top_n: Optional[int] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot weight distribution (for SC, SDiD).

        Parameters:
        -----------
        weights : dict
            Donor weights
        country : str
            Treated country name
        title : str, optional
            Custom title
        threshold : float, optional
            Only show weights above threshold (default: 0.01)
        top_n : int, optional
            Show only top N weights

        Returns:
        --------
        tuple : (figure, axes)
        """
        # Filter weights
        filtered_weights = {k: v for k, v in weights.items() if v > threshold}

        if not filtered_weights:
            print("No weights above threshold")
            return None, None

        # Sort by weight
        sorted_weights = sorted(filtered_weights.items(), key=lambda x: x[1], reverse=True)

        # Take top N if specified
        if top_n:
            sorted_weights = sorted_weights[:top_n]

        donors = [w[0] for w in sorted_weights]
        weight_values = [w[1] for w in sorted_weights]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Horizontal bar chart
        y_pos = np.arange(len(donors))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(donors)))

        bars = ax.barh(y_pos, weight_values, color=colors, edgecolor='black', linewidth=0.5)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(donors)
        ax.set_xlabel('Weight', fontsize=self.font_size)
        ax.set_ylabel('Donor Country', fontsize=self.font_size)

        if title is None:
            title = f'Donor Weights for {country}'
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, weight_values)):
            ax.text(value, i, f' {value:.3f}', va='center', fontsize=self.font_size - 2)

        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        self.current_fig = fig
        self.current_ax = ax

        return fig, ax

    def save(self,
            filename: Union[str, Path],
            dpi: Optional[int] = None,
            bbox_inches: str = 'tight',
            **kwargs):
        """
        Save the current figure.

        Parameters:
        -----------
        filename : str or Path
            Output filename
        dpi : int, optional
            DPI for saved figure (uses self.dpi if None)
        bbox_inches : str, optional
            Bbox setting (default: 'tight')
        **kwargs
            Additional arguments for plt.savefig()
        """
        if self.current_fig is None:
            warnings.warn("No figure to save")
            return

        if dpi is None:
            dpi = self.dpi

        self.current_fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"Saved figure to {filename}")

    def close(self):
        """Close the current figure."""
        if self.current_fig is not None:
            plt.close(self.current_fig)
            self.current_fig = None
            self.current_ax = None


class CounterfactualExporter:
    """
    Export counterfactual plots for multiple countries.

    Example:
    --------
    >>> exporter = CounterfactualExporter()
    >>> exporter.export_all(results, output_dir='plots/')
    """

    def __init__(self, plotter: Optional[CausalPlotter] = None):
        """
        Initialize exporter.

        Parameters:
        -----------
        plotter : CausalPlotter, optional
            Plotter instance (creates default if None)
        """
        self.plotter = plotter or CausalPlotter()

    def export_all(self,
                   results: 'CausalResults',
                   output_dir: Union[str, Path],
                   file_format: str = 'png'):
        """
        Export counterfactual plots for all countries in results.

        Parameters:
        -----------
        results : CausalResults
            Results object
        output_dir : str or Path
            Output directory
        file_format : str, optional
            File format (default: 'png')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for country, country_data in results.raw_results.items():
            method_result = country_data.get('method_result', {})

            if 'effect' in method_result:
                # Get actual and synthetic values
                actual = method_result.get('actual_values', {})
                synthetic = (method_result.get('synthetic_values', {})
                            or method_result.get('synthetic', {})
                            or method_result.get('shifted_control', {}))

                if actual and synthetic:
                    treatment_year = method_result.get('treatment_year')

                    # Plot
                    self.plotter.plot_counterfactual(
                        actual_values=actual,
                        synthetic_values=synthetic,
                        treatment_year=treatment_year,
                        country=country
                    )

                    # Save
                    filename = output_dir / f'{country}_counterfactual.{file_format}'
                    self.plotter.save(filename)
                    self.plotter.close()

        print(f"Exported {len(results.raw_results)} counterfactual plots to {output_dir}")


class EffectDistributionPlotter:
    """
    Plot treatment effect distributions.

    Example:
    --------
    >>> plotter = EffectDistributionPlotter()
    >>> plotter.plot_att_distribution(results)
    """

    def __init__(self, plotter: Optional[CausalPlotter] = None):
        """Initialize with optional plotter."""
        self.plotter = plotter or CausalPlotter()

    def plot_att_distribution(self,
                             results: 'CausalResults',
                             bins: int = 15) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot distribution of ATT values across countries.

        Parameters:
        -----------
        results : CausalResults
            Results object
        bins : int, optional
            Number of bins (default: 15)

        Returns:
        --------
        tuple : (figure, axes)
        """
        # Extract ATT values
        att_values = []
        for country, metrics in results.country_results.items():
            att = metrics.get('att')
            if att is not None and not np.isnan(att):
                att_values.append(att)

        if not att_values:
            print("No valid ATT values found")
            return None, None

        # Create histogram
        fig, ax = plt.subplots(figsize=self.plotter.figsize, dpi=self.plotter.dpi)

        ax.hist(att_values, bins=bins, alpha=0.7, color='#06FFA5',
                edgecolor='black', linewidth=1.5)

        # Mean line
        mean_att = np.mean(att_values)
        ax.axvline(x=mean_att, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_att:.3f}')

        # Median line
        median_att = np.median(att_values)
        ax.axvline(x=median_att, color='blue', linestyle=':', linewidth=2,
                  label=f'Median: {median_att:.3f}')

        # Labels
        ax.set_xlabel('Average Treatment Effect on Treated (ATT)', fontsize=self.plotter.font_size)
        ax.set_ylabel('Frequency', fontsize=self.plotter.font_size)
        ax.set_title(f'Distribution of ATT Across Countries\n({len(att_values)} countries)',
                    fontsize=self.plotter.font_size + 2, fontweight='bold')

        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        return fig, ax
