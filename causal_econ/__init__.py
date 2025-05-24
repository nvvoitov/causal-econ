"""
Causal Economics Library for Platform Economy Analysis

A comprehensive library for causal inference in economic analysis,
specifically designed for studying platform economy effects on GDP growth.


Include diff-in-diff (including synthetic), synthetic control (including generalized) and causal economic2vec.
"""

__version__ = "0.1.0"
__author__ = "Nikolay Voytov"

# Core imports
from .core.data_loader import load_data, preprocess_panel_data
from .core.base import CausalResults, ModelConfig

# Embeddings
from .embeddings.e2v import Economic2Vec, generate_geminis, run_e2v_analysis

# Donor pools
from .donor_pools.gemini_pools import create_donor_pools, get_donor_pools

__all__ = [
    # Core
    'load_data',
    'preprocess_panel_data', 
    'CausalResults',
    'ModelConfig',
    
    # Embeddings
    'Economic2Vec',
    'generate_geminis',
    'run_e2v_analysis',
    
    # Donor pools
    'create_donor_pools',
    'get_donor_pools'
]
