"""
Causal Economics Library for Platform Economy Analysis

A comprehensive library for causal inference in economic analysis,
specifically designed for studying platform economy effects on GDP growth.
"""

__version__ = "0.1.0"
__author__ = "Nikolay Voytov"

# Core imports
from .core.data_loader import load_data, preprocess_panel_data
from .core.base import (
    CausalResults,
    ModelConfig,
    DataValidator,
    calculate_causal_metrics,
    split_pre_post_periods
)
from .core.placebo import PlaceboTestRunner
from .core.pipeline import CausalAnalysisPipeline, AnalysisRegistry
from .core.results import ResultsAggregator
from .core.optimization import WeightOptimizer
from .core.data_processing import (
    PanelDataLoader,
    PanelDataPreprocessor,
    TreatmentIdentifier,
    FeatureExtractor,
    DataDictBuilder
)

# Embeddings
from .embeddings.e2v import Economic2Vec, generate_geminis, run_e2v_analysis

# Donor pools
from .donor_pools.gemini_pools import create_donor_pools, get_donor_pools

# Models - ADD THESE!
from .models.synthetic_control import run_sc_analysis
from .models.did import run_did_analysis_pipeline
from .models.sdid import run_sdid_pipeline
from .models.gsc import run_gsc_analysis
from .models.ce2v import run_causal_e2v_analysis
from .models.helpers import (
    SyntheticControlHelper,
    DifferenceInDifferencesHelper,
    SDiDHelper,
    GSCHelper
)

# Analysis - ADD THESE!
from .analysis.summary import combine_method_summaries, export_summary_tables
from .analysis.impact import create_impact_table, export_impact_tables
from .analysis.utilities import (
    SummaryGenerator,
    ImpactAnalyzer,
    MethodComparator,
    ResultExporter
)

# Visualization - ADD THESE!
from .visualization.counterfactuals import export_counterfactual_plots
from .visualization.distributions import export_distribution_plots
from .visualization.plotter import (
    CausalPlotter,
    CounterfactualExporter,
    EffectDistributionPlotter
)

__all__ = [
    # Core - Data
    'load_data',
    'preprocess_panel_data',

    # Core - Base classes
    'CausalResults',
    'ModelConfig',
    'DataValidator',

    # Core - Utilities
    'calculate_causal_metrics',
    'split_pre_post_periods',

    # Core - Pipeline framework (Phase 2)
    'CausalAnalysisPipeline',
    'AnalysisRegistry',
    'PlaceboTestRunner',
    'ResultsAggregator',
    'WeightOptimizer',

    # Core - Data processing classes
    'PanelDataLoader',
    'PanelDataPreprocessor',
    'TreatmentIdentifier',
    'FeatureExtractor',
    'DataDictBuilder',

    # Embeddings
    'Economic2Vec',
    'generate_geminis',
    'run_e2v_analysis',

    # Donor pools
    'create_donor_pools',
    'get_donor_pools',

    # Models
    'run_sc_analysis',
    'run_did_analysis_pipeline',
    'run_sdid_pipeline',
    'run_gsc_analysis',
    'run_causal_e2v_analysis',

    # Model helpers
    'SyntheticControlHelper',
    'DifferenceInDifferencesHelper',
    'SDiDHelper',
    'GSCHelper',

    # Analysis
    'combine_method_summaries',
    'export_summary_tables',
    'create_impact_table',
    'export_impact_tables',

    # Analysis utilities
    'SummaryGenerator',
    'ImpactAnalyzer',
    'MethodComparator',
    'ResultExporter',

    # Visualization
    'export_counterfactual_plots',
    'export_distribution_plots',

    # Visualization classes
    'CausalPlotter',
    'CounterfactualExporter',
    'EffectDistributionPlotter'
]