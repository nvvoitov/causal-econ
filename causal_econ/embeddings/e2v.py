"""
Economic2Vec: Economic embedding generation for similarity analysis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')


class EconomicDataset(Dataset):
    """Dataset for economic panel data with sliding windows."""
    
    def __init__(self, data: np.ndarray, window_size: int = 5):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
    
    def __len__(self) -> int:
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.data[idx:idx + self.window_size]
        return window, window  # Same window for input and target


class Economic2Vec(nn.Module):
    """Economic2Vec model for generating country embeddings."""
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super(Economic2Vec, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, input_dim = x.shape
        
        # Reshape to process all windows together
        x_reshaped = x.reshape(-1, input_dim)
        
        # Encode and decode
        encoded = self.encoder(x_reshaped)
        decoded = self.decoder(encoded)
        
        # Reshape back
        decoded = decoded.reshape(batch_size, window_size, input_dim)
        
        return decoded
    
    def get_embeddings(self, x: torch.Tensor) -> np.ndarray:
        """Extract embeddings without decoding."""
        with torch.no_grad():
            return self.encoder(x).detach().numpy()


def train_economic2vec(data: np.ndarray, 
                      embedding_dim: int = 64,
                      epochs: int = 50,
                      batch_size: int = 32,
                      window_size: int = 5,
                      initial_lr: float = 0.001,
                      min_lr: float = 1e-6) -> Economic2Vec:
    """
    Train Economic2Vec model.
    
    Parameters:
    -----------
    data : np.ndarray
        Scaled economic data
    embedding_dim : int
        Dimension of embeddings
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    window_size : int
        Size of temporal windows
    initial_lr : float
        Initial learning rate
    min_lr : float
        Minimum learning rate
        
    Returns:
    --------
    Economic2Vec : Trained model
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create datasets
    full_dataset = EconomicDataset(data, window_size)
    
    # Create train/val split
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = Economic2Vec(data.shape[1], embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    print(f"Training Economic2Vec with {embedding_dim}D embeddings for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        
        for batch_windows, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_windows)
            
            loss = criterion(output, batch_windows)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for batch_windows, _ in val_loader:
                output = model(batch_windows)
                val_loss = criterion(output, batch_windows)
                epoch_val_loss += val_loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
    
    return model


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    sim_matrix = 1 - cdist(embeddings, embeddings, metric='cosine')
    return np.nan_to_num(sim_matrix, nan=0.0)


def perform_kmeans_clustering(embeddings: np.ndarray, 
                            countries: np.ndarray,
                            n_clusters: int = 8) -> Tuple[np.ndarray, KMeans, np.ndarray]:
    """
    Perform K-means clustering on country embeddings.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Country embeddings
    countries : np.ndarray
        Country names
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    tuple : (cluster_labels, kmeans_model, country_embeddings)
    """
    # Calculate country embeddings (average across time)
    unique_countries = np.unique(countries)
    country_embeddings = np.zeros((len(unique_countries), embeddings.shape[1]))
    
    for i, country in enumerate(unique_countries):
        mask = countries == country
        country_embeddings[i] = np.mean(embeddings[mask], axis=0)
    
    # Perform K-means clustering with error handling
    try:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans_model.fit_predict(country_embeddings)
    except AttributeError as e:
        if 'NoneType' in str(e) and 'split' in str(e):
            # BLAS configuration error - use alternative approach
            print("Warning: BLAS configuration issue detected. Using alternative clustering approach.")
            import os
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Try again with single thread
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, algorithm='elkan')
            cluster_labels = kmeans_model.fit_predict(country_embeddings)
        else:
            raise
    
    return cluster_labels, kmeans_model, country_embeddings


def export_gemini_clusters(embeddings: np.ndarray,
                          countries: np.ndarray,
                          cluster_labels: np.ndarray,
                          similarity_matrix: np.ndarray,
                          output_file: str = 'gemini_clusters.csv') -> pd.DataFrame:
    """Export country clusters and their most similar partners (geminis)."""
    
    # Aggregate by country
    unique_countries = np.unique(countries)
    country_similarities = np.zeros((len(unique_countries), len(unique_countries)))
    
    for i, country1 in enumerate(unique_countries):
        for j, country2 in enumerate(unique_countries):
            mask1 = countries == country1
            mask2 = countries == country2
            if np.any(mask1) and np.any(mask2):
                country_similarities[i, j] = np.mean(similarity_matrix[mask1][:, mask2])
    
    # Find most similar partner (gemini) for each country
    mask = np.eye(len(unique_countries), dtype=bool)
    masked_similarities = country_similarities.copy()
    masked_similarities[mask] = -np.inf  # Exclude self-similarities
    gemini_indices = np.argmax(masked_similarities, axis=1)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Country': unique_countries,
        'Cluster': cluster_labels,
        'Gemini': unique_countries[gemini_indices],
        'Similarity_Score': [country_similarities[i, gemini_indices[i]] 
                           for i in range(len(unique_countries))]
    })
    
    # Sort by cluster and similarity score
    results_df = results_df.sort_values(['Cluster', 'Similarity_Score'], 
                                      ascending=[True, False])
    
    # Save to CSV
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Gemini clusters saved to {output_file}")
    
    return results_df


def export_top_geminis(unique_countries: np.ndarray,
                      country_similarities: np.ndarray,
                      top_n: int = 20,
                      output_file: str = 'top_geminis.csv') -> pd.DataFrame:
    """Export the top N most similar countries (chains) for each country."""
    
    # Create a dictionary to store each country's top N similar countries
    country_geminis = {}
    
    # Create a mask for the diagonal to exclude self-similarities
    mask = np.eye(len(unique_countries), dtype=bool)
    masked_similarities = country_similarities.copy()
    masked_similarities[mask] = -np.inf  # Set diagonal to negative infinity
    
    for i, country in enumerate(unique_countries):
        # Get indices of top N similar countries (sorted by similarity)
        top_indices = np.argsort(masked_similarities[i])[::-1][:top_n]
        
        # Get similarity scores for these countries
        top_scores = [masked_similarities[i, idx] for idx in top_indices]
        
        # Get country names
        top_geminis = [unique_countries[idx] for idx in top_indices]
        
        # Store results
        country_geminis[country] = {
            'geminis': top_geminis,
            'scores': top_scores
        }
    
    # Create a DataFrame for export
    rows = []
    for country, data in country_geminis.items():
        for gemini, score in zip(data['geminis'], data['scores']):
            rows.append({
                'Country': country,
                'Gemini': gemini,
                'Similarity_Score': score
            })
    
    gemini_df = pd.DataFrame(rows)
    
    # Save to CSV
    if output_file:
        gemini_df.to_csv(output_file, index=False)
        print(f"Top {top_n} geminis for each country saved to {output_file}")
    
    return gemini_df


def run_e2v_analysis(data_dict: Dict,
                     embedding_dim: int = 64,
                     n_clusters: int = 8,
                     top_n: int = 20,
                     epochs: int = 50,
                     output_dir: Optional[str] = None) -> Dict:
    """
    Run complete E2V analysis pipeline.
    
    Parameters:
    -----------
    data_dict : dict
        Preprocessed data dictionary
    embedding_dim : int
        Embedding dimension
    n_clusters : int
        Number of clusters for gemini clustering
    top_n : int
        Number of top similar countries to extract
    epochs : int
        Training epochs
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    dict : Complete E2V results
    """
    df = data_dict['df']
    feature_cols = data_dict['feature_cols']
    
    print(f"Starting E2V analysis...")
    print(f"Features: {len(feature_cols)}")
    print(f"Countries: {df[data_dict['country_col']].nunique()}")
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Train Economic2Vec model
    model = train_economic2vec(
        scaled_data, 
        embedding_dim=embedding_dim,
        epochs=epochs
    )
    
    # Get embeddings
    embeddings = model.get_embeddings(torch.FloatTensor(scaled_data))
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Perform clustering
    countries = df[data_dict['country_col']].values
    cluster_labels, kmeans_model, country_embeddings = perform_kmeans_clustering(
        embeddings, countries, n_clusters=n_clusters
    )
    
    # Calculate country-level similarity matrix
    unique_countries = np.unique(countries)
    country_similarities = np.zeros((len(unique_countries), len(unique_countries)))
    
    for i, country1 in enumerate(unique_countries):
        for j, country2 in enumerate(unique_countries):
            mask1 = countries == country1
            mask2 = countries == country2
            if np.any(mask1) and np.any(mask2):
                country_similarities[i, j] = np.mean(similarity_matrix[mask1][:, mask2])
    
    # Export results
    output_files = {}
    if output_dir:
        clusters_file = f"{output_dir}/gemini_clusters.csv"
        chains_file = f"{output_dir}/top_geminis.csv"
    else:
        clusters_file = "gemini_clusters.csv"
        chains_file = "top_geminis.csv"
    
    # Export gemini clusters
    clusters_df = export_gemini_clusters(
        embeddings, countries, cluster_labels, similarity_matrix,
        output_file=clusters_file
    )
    
    # Export top geminis (chains)
    chains_df = export_top_geminis(
        unique_countries, country_similarities, top_n=top_n,
        output_file=chains_file
    )
    
    return {
        'model': model,
        'embeddings': embeddings,
        'country_embeddings': country_embeddings,
        'similarity_matrix': similarity_matrix,
        'country_similarities': country_similarities,
        'cluster_labels': cluster_labels,
        'kmeans_model': kmeans_model,
        'clusters_df': clusters_df,
        'chains_df': chains_df,
        'scaler': scaler,
        'unique_countries': unique_countries
    }


def generate_geminis(data_dict: Dict, 
                    method: str = 'chains',
                    **kwargs) -> pd.DataFrame:
    """
    Simplified interface for generating geminis.
    
    Parameters:
    -----------
    data_dict : dict
        Preprocessed data dictionary
    method : str
        'chains' for top_geminis or 'clusters' for gemini_clusters
    **kwargs : dict
        Additional parameters for run_e2v_analysis
        
    Returns:
    --------
    pd.DataFrame : Gemini relationships
    """
    results = run_e2v_analysis(data_dict, **kwargs)
    
    if method == 'chains':
        return results['chains_df']
    elif method == 'clusters':
        return results['clusters_df']
    else:
        raise ValueError("Method must be 'chains' or 'clusters'")
