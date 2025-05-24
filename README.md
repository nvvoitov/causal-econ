# Causal-Econ: Causal Inference Library for Economic Analysis

A comprehensive Python library for causal inference in economic analysis, specifically designed for studying platform economy effects on GDP growth.

## Features

- **Multiple Causal Inference Methods**: Synthetic Control, Difference-in-Differences, Synthetic DiD, Generalized Synthetic Control, Causal Economic2Vec
- **Economic Embeddings**: Economic2Vec for finding similar countries (geminis)
- **Flexible Donor Pools**: Automatic donor selection based on economic similarity
- **Comprehensive Testing**: Built-in placebo tests for all methods
- **Rich Visualizations**: Interactive plots for counterfactuals, distributions, and embeddings
- **Standardized Results**: Unified interface across all methods

## Installation

```bash
pip install git+https://github.com/nvvoitov/causal-econ.git
```

## Important Notes!

- In E2V clusterization part could be conflicts due to numpy version (recommended to install the most up to date version).
- Library still in progress: functions could be used, however, some interfaces produce outputs that need to be heavily modified before passing furter onto the pipeline.
- No documentation published yet: models have various parameters that for now are hardcoded in the scripts. This is subject for future changes.
