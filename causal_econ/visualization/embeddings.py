"""
Visualization tools for economic embeddings (E2V and CE2V).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional, Tuple, Any
import warnings


def create_tsne_embeddings(embeddings: np.ndarray, 
                          n_components: int = 2, 
                          perplexity: int = 30,
                          random_state: int = 42) -> np.ndarray:
    """
    Create t-SNE embeddings for visualization.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        High-dimensional embeddings
    n_components : int
        Number of dimensions for t-SNE output
    perplexity : int
        Perplexity parameter for t-SNE
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    np.ndarray : t-SNE embeddings
    """
    print(f"Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    return tsne.fit_transform(embeddings)


def create_pca_embeddings(embeddings: np.ndarray, 
                         n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create PCA embeddings for visualization.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        High-dimensional embeddings
    n_components : int
        Number of components for PCA
        
    Returns:
    --------
    tuple : (pca_embeddings, explained_variance_ratio)
    """
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    return pca_embeddings, pca.explained_variance_ratio_


def visualize_country_embeddings_2d(embeddings: np.ndarray,
                                   countries: np.ndarray,
                                   country_metadata: Optional[Dict] = None,
                                   method: str = 'tsne',
                                   title: str = "Country Embeddings",
                                   color_by: str = 'treatment_status') -> go.Figure:
    """
    Create 2D visualization of country embeddings.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Country embeddings
    countries : np.ndarray
        Country names
    country_metadata : dict, optional
        Metadata for countries (treatment status, etc.)
    method : str
        Dimensionality reduction method ('tsne' or 'pca')
    title : str
        Plot title
    color_by : str
        What to color points by ('treatment_status', 'cluster', 'none')
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    # Get unique countries and their average embeddings
    unique_countries = np.unique(countries)
    country_embeddings = np.zeros((len(unique_countries), embeddings.shape[1]))
    
    for i, country in enumerate(unique_countries):
        mask = countries == country
        country_embeddings[i] = np.mean(embeddings[mask], axis=0)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        coords_2d = create_tsne_embeddings(country_embeddings, n_components=2)
        x_label, y_label = "t-SNE 1", "t-SNE 2"
    elif method == 'pca':
        coords_2d, var_ratio = create_pca_embeddings(country_embeddings, n_components=2)
        x_label = f"PC1 ({var_ratio[0]:.1%} var)"
        y_label = f"PC2 ({var_ratio[1]:.1%} var)"
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'country': unique_countries,
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1]
    })
    
    # Add coloring information
    if country_metadata and color_by == 'treatment_status':
        plot_data['treatment_status'] = [
            'Treated' if country_metadata.get(country, {}).get('is_treated', False) else 'Control'
            for country in unique_countries
        ]
        color_col = 'treatment_status'
        color_map = {'Treated': 'red', 'Control': 'blue'}
    else:
        plot_data['color'] = 'Country'
        color_col = 'color'
        color_map = None
    
    # Create figure
    fig = px.scatter(
        plot_data,
        x='x',
        y='y',
        color=color_col,
        color_discrete_map=color_map,
        hover_data=['country'],
        title=title,
        template='plotly_white'
    )
    
    # Add country labels
    fig.add_trace(
        go.Scatter(
            x=plot_data['x'],
            y=plot_data['y'],
            mode='text',
            text=plot_data['country'],
            textposition='top center',
            showlegend=False,
            textfont=dict(size=8, color='black')
        )
    )
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=900,
        height=700
    )
    
    return fig


def visualize_country_embeddings_3d(embeddings: np.ndarray,
                                   countries: np.ndarray,
                                   country_metadata: Optional[Dict] = None,
                                   method: str = 'tsne',
                                   title: str = "3D Country Embeddings",
                                   color_by: str = 'treatment_status') -> go.Figure:
    """
    Create 3D visualization of country embeddings.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Country embeddings
    countries : np.ndarray
        Country names
    country_metadata : dict, optional
        Metadata for countries
    method : str
        Dimensionality reduction method ('tsne' or 'pca')
    title : str
        Plot title
    color_by : str
        What to color points by
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    # Get unique countries and their average embeddings
    unique_countries = np.unique(countries)
    country_embeddings = np.zeros((len(unique_countries), embeddings.shape[1]))
    
    for i, country in enumerate(unique_countries):
        mask = countries == country
        country_embeddings[i] = np.mean(embeddings[mask], axis=0)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        coords_3d = create_tsne_embeddings(country_embeddings, n_components=3)
        x_label, y_label, z_label = "t-SNE 1", "t-SNE 2", "t-SNE 3"
    elif method == 'pca':
        coords_3d, var_ratio = create_pca_embeddings(country_embeddings, n_components=3)
        x_label = f"PC1 ({var_ratio[0]:.1%} var)"
        y_label = f"PC2 ({var_ratio[1]:.1%} var)"
        z_label = f"PC3 ({var_ratio[2]:.1%} var)"
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    # Prepare coloring
    if country_metadata and color_by == 'treatment_status':
        colors = [
            'red' if country_metadata.get(country, {}).get('is_treated', False) else 'blue'
            for country in unique_countries
        ]
        color_labels = [
            'Treated' if country_metadata.get(country, {}).get('is_treated', False) else 'Control'
            for country in unique_countries
        ]
    else:
        colors = 'blue'
        color_labels = ['Country'] * len(unique_countries)
    
    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.8
            ),
            text=unique_countries,
            textposition='top center',
            name='Countries',
            hovertemplate='<b>%{text}</b><br>' +
                         f'{x_label}: %{{x:.2f}}<br>' +
                         f'{y_label}: %{{y:.2f}}<br>' +
                         f'{z_label}: %{{z:.2f}}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            bgcolor='lightblue',
            xaxis=dict(backgroundcolor='rgb(230, 230, 250)'),
            yaxis=dict(backgroundcolor='rgb(230, 230, 250)'),
            zaxis=dict(backgroundcolor='rgb(230, 230, 250)')
        ),
        width=900,
        height=700
    )
    
    return fig


def visualize_embedding_density(embeddings: np.ndarray,
                               countries: np.ndarray,
                               title: str = "Embedding Density Visualization") -> go.Figure:
    """
    Create density visualization of embeddings with KDE.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embeddings to visualize
    countries : np.ndarray
        Country names
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    # Reduce to 2D for density visualization
    coords_2d = create_tsne_embeddings(embeddings, n_components=2)
    
    # Calculate density using KDE
    x, y = coords_2d[:, 0], coords_2d[:, 1]
    positions = np.vstack([x, y])
    kernel = gaussian_kde(positions)
    density = kernel(positions)
    
    # Create figure
    fig = go.Figure()
    
    # Add density scatter plot
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=6,
                color=density,
                colorscale='Plasma',
                opacity=0.7,
                colorbar=dict(title="Density")
            ),
            name='Observations',
            hovertemplate='Density: %{marker.color:.3f}<extra></extra>'
        )
    )
    
    # Identify high density clusters
    high_density = density > np.percentile(density, 95)
    
    if np.any(high_density):
        fig.add_trace(
            go.Scatter(
                x=x[high_density],
                y=y[high_density],
                mode='markers',
                marker=dict(
                    size=8,
                    color='orange',
                    opacity=1.0
                ),
                name='High Density'
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        template='plotly_white',
        width=900,
        height=700
    )
    
    return fig


def visualize_treatment_effects_in_embedding_space(embeddings: np.ndarray,
                                                  countries: np.ndarray,
                                                  treatment_effects: Dict[str, float],
                                                  title: str = "Treatment Effects in Embedding Space") -> go.Figure:
    """
    Visualize treatment effects mapped onto embedding space.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Country embeddings
    countries : np.ndarray
        Country names
    treatment_effects : dict
        Mapping of country names to treatment effects
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure : Plotly figure
    """
    # Get unique countries and their average embeddings
    unique_countries = np.unique(countries)
    country_embeddings = np.zeros((len(unique_countries), embeddings.shape[1]))
    effects = []
    
    for i, country in enumerate(unique_countries):
        mask = countries == country
        country_embeddings[i] = np.mean(embeddings[mask], axis=0)
        effects.append(treatment_effects.get(country, 0))
    
    effects = np.array(effects)
    
    # Reduce to 2D
    coords_2d = create_tsne_embeddings(country_embeddings, n_components=2)
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=10,
                color=effects,
                colorscale='RdBu',
                opacity=0.8,
                colorbar=dict(title="Treatment Effect"),
                cmin=-np.max(np.abs(effects)),
                cmax=np.max(np.abs(effects))
            ),
            text=unique_countries,
            textposition='top center',
            name='Countries',
            hovertemplate='<b>%{text}</b><br>' +
                         'Treatment Effect: %{marker.color:.3f}<br>' +
                         't-SNE 1: %{x:.2f}<br>' +
                         't-SNE 2: %{y:.2f}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        template='plotly_white',
        width=900,
        height=700
    )
    
    return fig


def create_embedding_comparison_dashboard(e2v_results: Dict,
                                        ce2v_results: Optional[Dict] = None,
                                        country_metadata: Optional[Dict] = None) -> go.Figure:
    """
    Create a dashboard comparing E2V and CE2V embeddings.
    
    Parameters:
    -----------
    e2v_results : dict
        Results from E2V analysis
    ce2v_results : dict, optional
        Results from CE2V analysis
    country_metadata : dict, optional
        Country metadata for coloring
        
    Returns:
    --------
    go.Figure : Multi-panel dashboard
    """
    if ce2v_results is None:
        # Single method dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["E2V Embeddings (2D)", "E2V Embeddings (3D)", 
                          "E2V Density", "E2V Clusters"],
            specs=[[{"type": "scatter"}, {"type": "scatter3d"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Add E2V plots
        embeddings = e2v_results['embeddings']
        countries = e2v_results.get('countries', np.arange(len(embeddings)))
        
        # 2D plot
        coords_2d = create_tsne_embeddings(embeddings, n_components=2)
        fig.add_trace(
            go.Scatter(x=coords_2d[:, 0], y=coords_2d[:, 1], mode='markers', name='E2V'),
            row=1, col=1
        )
        
    else:
        # Comparison dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["E2V Embeddings", "CE2V Embeddings", 
                          "E2V Density", "CE2V Density"],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Add both E2V and CE2V plots
        # Implementation would depend on the specific structure of results
        pass
    
    fig.update_layout(
        title="Economic Embeddings Dashboard",
        showlegend=True,
        height=800,
        width=1200
    )
    
    return fig


def analyze_embedding_clusters(embeddings: np.ndarray,
                             countries: np.ndarray,
                             cluster_labels: np.ndarray,
                             country_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Analyze the composition and characteristics of embedding clusters.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Country embeddings
    countries : np.ndarray
        Country names
    cluster_labels : np.ndarray
        Cluster assignments
    country_metadata : dict, optional
        Country metadata
        
    Returns:
    --------
    dict : Cluster analysis results
    """
    unique_countries = np.unique(countries)
    n_clusters = len(np.unique(cluster_labels))
    
    cluster_analysis = {
        'n_clusters': n_clusters,
        'cluster_sizes': {},
        'cluster_composition': {},
        'treatment_distribution': {}
    }
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_countries = unique_countries[cluster_mask]
        
        cluster_analysis['cluster_sizes'][cluster_id] = len(cluster_countries)
        cluster_analysis['cluster_composition'][cluster_id] = cluster_countries.tolist()
        
        if country_metadata:
            treated_count = sum(1 for c in cluster_countries 
                              if country_metadata.get(c, {}).get('is_treated', False))
            cluster_analysis['treatment_distribution'][cluster_id] = {
                'treated': treated_count,
                'control': len(cluster_countries) - treated_count,
                'treatment_rate': treated_count / len(cluster_countries)
            }
    
    return cluster_analysis
