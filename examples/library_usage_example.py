"""
Complete usage example of the causal_econ library.
"""

# Core imports
from causal_econ.core.data_loader import load_data, preprocess_panel_data
from causal_econ.donor_pools.gemini_pools import create_donor_pools
from causal_econ.embeddings.e2v import run_e2v_analysis

# Model imports
from causal_econ.models.synthetic_control import run_sc_analysis
from causal_econ.models.sdid import run_sdid_pipeline
from causal_econ.models.gsc import run_gsc_analysis

# Analysis and visualization imports
from causal_econ.analysis.summary import combine_method_summaries, export_summary_tables
from causal_econ.analysis.impact import create_cross_method_impact_comparison, export_impact_tables
from causal_econ.visualization.distributions import create_gdp_distribution_plot, export_distribution_plots
from causal_econ.visualization.counterfactuals import export_counterfactual_plots

def run_complete_analysis():
    """Run complete causal inference analysis pipeline."""
    
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    data_dict = load_data('../../data/processed/models_input/geminis_020525.csv')
    data_dict = preprocess_panel_data(data_dict)
    
    # 2. Generate embeddings and donor pools
    print("Generating E2V embeddings...")
    e2v_results = run_e2v_analysis(data_dict, embedding_dim=64, n_clusters=8)
    
    print("Creating donor pools...")
    donor_pools = create_donor_pools(
        data_dict, 
        gemini_df=e2v_results['clusters_df'],
        method='e2v',
        n_donors=20
    )
    
    # 3. Run all causal methods
    print("Running causal inference methods...")
    methods_results = {}
    
    # Synthetic Control
    methods_results['Synthetic Control'] = run_sc_analysis(
        data_dict, donor_pools, n_placebos=20
    )
    
    # Synthetic Difference-in-Differences  
    methods_results['SDiD'] = run_sdid_pipeline(
        data_dict, donor_pools, n_placebos=20
    )
    
    # Generalized Synthetic Control
    methods_results['GSC'] = run_gsc_analysis(
        data_dict, donor_pools, num_factors=2, n_placebos=20
    )
    
    # 4. Generate comprehensive analysis
    print("Creating summary tables...")
    combined_summary = combine_method_summaries(methods_results)
    print(combined_summary.head())
    
    print("Creating impact comparison...")
    impact_comparison = create_cross_method_impact_comparison(methods_results)
    print(impact_comparison.head())
    
    # 5. Export everything
    print("Exporting results...")
    export_summary_tables(methods_results, '../../results/summary_tables/')
    export_impact_tables(methods_results, '../../results/impact_tables/')
    
    # Export visualizations for each method
    for method_name, results in methods_results.items():
        output_dir = f'../../results/plots/{method_name.lower().replace(" ", "_")}'
        
        # Distribution plots
        export_distribution_plots(results, method_name, output_dir)
        
        # Counterfactual plots
        export_counterfactual_plots(results, output_dir)
    
    print("Analysis complete!")
    return methods_results, combined_summary, impact_comparison

# Run the complete analysis
if __name__ == "__main__":
    results, summary, impact = run_complete_analysis()
