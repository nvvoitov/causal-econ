"""
Causal Economic2Vec implementation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings

from ..core.base import CausalResults, calculate_treatment_effect, validate_panel_data


class PanelDataset(Dataset):
    """Dataset for panel data respecting temporal structure."""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], treatment_col: str = 'has_platform', 
                 outcome_col: str = 'gdp_growth_annual_share', seq_length: int = 5):
        self.df = df
        self.feature_cols = feature_cols
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.seq_length = seq_length
        
        # Create country and time identifiers
        self.df['country_id'] = self.df['country'].astype('category').cat.codes
        
        # Generate panel sequences
        self.sequences = []
        for country in df['country'].unique():
            country_data = df[df['country'] == country].sort_values('year')
            
            # Skip if not enough data
            if len(country_data) < seq_length:
                continue
                
            # Create sliding windows
            for i in range(len(country_data) - seq_length + 1):
                seq_data = country_data.iloc[i:i+seq_length]
                
                # Store sequence metadata
                self.sequences.append({
                    'country': country,
                    'start_year': seq_data['year'].iloc[0],
                    'end_year': seq_data['year'].iloc[-1],
                    'features': seq_data[feature_cols].values,
                    'treatment': seq_data[treatment_col].values,
                    'outcome': seq_data[outcome_col].values,
                    'country_id': seq_data['country_id'].iloc[0],
                    'years': seq_data['year'].values,
                    'idx': seq_data.index.values
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Create normalized time encoding
        years = seq['years']
        time_norm = (years - years.min()) / (years.max() - years.min() + 1e-6)
        
        return {
            'features': torch.FloatTensor(seq['features']),
            'treatment': torch.FloatTensor(seq['treatment']),
            'outcome': torch.FloatTensor(seq['outcome']),
            'country_id': torch.LongTensor([seq['country_id']]),
            'time': torch.FloatTensor(time_norm),
            'metadata': {
                'country': seq['country'],
                'years': seq['years'],
                'idx': seq['idx']
            }
        }


class EconomicEmbeddingModel(nn.Module):
    """Economic embedding model with panel awareness."""
    
    def __init__(self, input_dim: int, embedding_dim: int = 32, num_countries: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Country embeddings (entity fixed effects)
        self.country_embeddings = nn.Embedding(num_countries, embedding_dim // 4)
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Temporal module
        self.temporal_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True
        )
        
        # Final combiner
        self.combiner = nn.Linear(embedding_dim + embedding_dim // 4, embedding_dim)
        
    def forward(self, features, country_ids, time=None):
        batch_size, seq_len, feat_dim = features.shape
        
        # Feature encoding
        feat_embeddings = self.feature_encoder(features)
        
        # Temporal processing
        temporal_emb, _ = self.temporal_encoder(feat_embeddings)
        
        # Get country embeddings and expand to match temporal dimensions
        country_emb = self.country_embeddings(country_ids)
        country_emb = country_emb.expand(batch_size, seq_len, -1)
        
        # Combine embeddings
        combined = torch.cat([temporal_emb, country_emb], dim=-1)
        final_emb = self.combiner(combined)
        
        return final_emb


class CausalModel(nn.Module):
    """Causal model that predicts outcomes based on embeddings."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        # Treatment effect module
        self.treatment_processor = nn.Linear(1, embedding_dim // 2)
        
        # Outcome predictor
        self.outcome_net = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
    
    def forward(self, embeddings, treatment):
        # Process treatment
        treatment_expanded = treatment.unsqueeze(-1)
        treatment_emb = self.treatment_processor(treatment_expanded)
        
        # Combine embeddings with treatment effect
        combined = torch.cat([embeddings, treatment_emb], dim=-1)
        
        # Predict outcome
        outcome = self.outcome_net(combined).squeeze(-1)
        
        return outcome
    
    def counterfactual(self, embeddings, factual_treatment, counterfactual_treatment):
        """Generate counterfactual predictions."""
        factual = self.forward(embeddings, factual_treatment)
        counterfactual = self.forward(embeddings, counterfactual_treatment)
        return factual, counterfactual


class CausalEconomic2Vec(nn.Module):
    """Combined model for causal economic embeddings."""
    
    def __init__(self, input_dim: int, embedding_dim: int = 32, num_countries: Optional[int] = None):
        super().__init__()
        self.embedding_model = EconomicEmbeddingModel(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_countries=num_countries
        )
        self.causal_model = CausalModel(embedding_dim=embedding_dim)
    
    def forward(self, features, treatment, country_ids, time=None):
        # Get embeddings
        embeddings = self.embedding_model(features, country_ids, time)
        
        # Predict outcomes
        outcomes = self.causal_model(embeddings, treatment)
        
        return {
            'outcomes': outcomes,
            'embeddings': embeddings
        }
    
    def get_counterfactual(self, features, factual_treatment, 
                           counterfactual_treatment, country_ids, time=None):
        """Generate counterfactual predictions."""
        # Get embeddings
        embeddings = self.embedding_model(features, country_ids, time)
        
        # Get factual and counterfactual predictions
        factual, counterfactual = self.causal_model.counterfactual(
            embeddings, factual_treatment, counterfactual_treatment
        )
        
        return factual, counterfactual


def preprocess_panel_data_ce2v(df_path: str, treatment_col: str = 'has_platform', 
                              outcome_col: str = 'gdp_growth_annual_share',
                              country_col: str = 'country', year_col: str = 'year') -> Tuple[pd.DataFrame, Dict, List[str], StandardScaler]:
    """Prepare economic panel data for CE2V analysis."""
    
    # Load data
    df = pd.read_csv(df_path)
    
    # Basic cleaning
    print(f"Initial data shape: {df.shape}")
    
    # Identify feature columns (exclude country, year, treatment)
    id_cols = [country_col, year_col, treatment_col]
    feature_cols = [col for col in df.columns if col not in id_cols]
    
    # Ensure panel is sorted by country and year
    df = df.sort_values([country_col, year_col])
    
    # Fill missing values appropriately for time series
    df[feature_cols] = df.groupby(country_col)[feature_cols].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    
    # Calculate treatment timing for each country
    treatment_timing = df[df[treatment_col] == 1].groupby(country_col)[year_col].min()
    
    # Create treatment timing metadata
    countries = df[country_col].unique()
    country_metadata = {}
    
    for country in countries:
        country_data = df[df[country_col] == country]
        years = country_data[year_col].values
        
        if country in treatment_timing.index:
            treat_year = treatment_timing[country]
            pre_treat_years = sum(years < treat_year)
            is_treated = True
        else:
            treat_year = None
            pre_treat_years = len(years)
            is_treated = False
            
        country_metadata[country] = {
            'treatment_year': treat_year,
            'pre_treatment_periods': pre_treat_years,
            'is_treated': is_treated,
            'years': years
        }
    
    # Check balance and print stats
    treated_countries = [c for c, m in country_metadata.items() if m['is_treated']]
    control_countries = [c for c, m in country_metadata.items() if not m['is_treated']]
    
    print(f"Treated countries: {len(treated_countries)}")
    print(f"Control countries: {len(control_countries)}")
    print(f"Average pre-treatment periods: {np.mean([m['pre_treatment_periods'] for c, m in country_metadata.items() if m['is_treated']]):.1f}")
    
    # Remove countries with insufficient data
    min_periods = 5
    valid_countries = [c for c, m in country_metadata.items() 
                      if len(m['years']) >= min_periods]
    df = df[df[country_col].isin(valid_countries)]
    
    # Handle remaining NaNs in features
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Create scaler but DO NOT fit it yet
    scaler = StandardScaler()
    
    return df, country_metadata, feature_cols, scaler


def train_causal_e2v(model: CausalEconomic2Vec, train_loader: DataLoader, val_loader: DataLoader, 
                    epochs: int = 50, balance_weight: float = 0.5) -> CausalEconomic2Vec:
    """Train the Causal Economic2Vec model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            features = batch['features'].to(device)
            treatment = batch['treatment'].to(device)
            outcome = batch['outcome'].to(device)
            country_ids = batch['country_id'].to(device)
            time = batch['time'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features, treatment, country_ids, time)
            
            # Basic prediction loss
            pred_loss = F.mse_loss(outputs['outcomes'], outcome)
            
            # Balance loss - make embeddings similar across treatment groups
            embeddings = outputs['embeddings']
            treated_mask = treatment.bool().any(dim=1)
            
            if treated_mask.any() and (~treated_mask).any():
                treated_emb = embeddings[treated_mask].mean(0)
                control_emb = embeddings[~treated_mask].mean(0)
                balance_loss = F.mse_loss(treated_emb, control_emb)
            else:
                balance_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = pred_loss + balance_weight * balance_loss
            
            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(total_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                treatment = batch['treatment'].to(device)
                outcome = batch['outcome'].to(device)
                country_ids = batch['country_id'].to(device)
                time = batch['time'].to(device)
                
                outputs = model(features, treatment, country_ids, time)
                val_loss = F.mse_loss(outputs['outcomes'], outcome)
                val_losses.append(val_loss.item())
        
        # Track metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
    
    # Load best model
    if best_model:
        model.load_state_dict(best_model)
        model = model.to(device)
    
    return model


def generate_and_evaluate_counterfactuals(model: CausalEconomic2Vec, df: pd.DataFrame, 
                                        country_metadata: Dict, feature_cols: List[str], 
                                        scaler: StandardScaler, treatment_col: str = 'has_platform',
                                        outcome_col: str = 'gdp_growth_annual_share',
                                        country_col: str = 'country', year_col: str = 'year') -> Dict:
    """Generate counterfactuals and evaluate them for all treated countries."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create consistent country ID mapping
    country_to_id = {country: i for i, country in enumerate(df[country_col].unique())}
    
    # Get all treated countries with sufficient pre-treatment data
    treated_countries = [c for c, m in country_metadata.items() 
                         if m['is_treated'] and m['pre_treatment_periods'] >= 3 and c in df[country_col].unique()]
    
    results = {}
    
    with torch.no_grad():
        for country in treated_countries:
            # Get country data (already scaled)
            country_data = df[df[country_col] == country].sort_values(year_col)
            treatment_year = country_metadata[country]['treatment_year']
            
            if treatment_year is None or country_data.empty:
                continue
                
            # Prepare data tensors
            features = torch.FloatTensor(country_data[feature_cols].values).to(device)
            actual_treatment = torch.FloatTensor(country_data[treatment_col].values).to(device)
            counterfactual_treatment = torch.zeros_like(actual_treatment).to(device)
            
            # Use consistent country ID mapping
            country_id = torch.LongTensor([country_to_id[country]]).to(device)
            
            # Get years and create normalized time
            years = country_data[year_col].values
            time_norm = (years - years.min()) / (years.max() - years.min() + 1e-6)
            time = torch.FloatTensor(time_norm).to(device)
            
            # Add batch dimension
            features = features.unsqueeze(0)
            actual_treatment = actual_treatment.unsqueeze(0)
            counterfactual_treatment = counterfactual_treatment.unsqueeze(0)
            time = time.unsqueeze(0)
            
            # Generate counterfactual
            factual, counterfactual = model.get_counterfactual(
                features, actual_treatment, counterfactual_treatment, country_id, time
            )
            
            # Extract values
            factual_values = factual.squeeze().cpu().numpy()
            counterfactual_values = counterfactual.squeeze().cpu().numpy()
            
            # Get actual outcome values
            actual_values = country_data[outcome_col].values
            
            # Align with years
            results[country] = {
                'years': years,
                'actual': actual_values,
                'factual': factual_values,
                'counterfactual': counterfactual_values,
                'treatment_year': treatment_year
            }
            
            # Calculate treatment effect metrics
            pre_treatment = years < treatment_year
            post_treatment = years >= treatment_year
            
            if sum(pre_treatment) > 0:
                # Pre-treatment fit error
                pre_treat_error = np.mean((actual_values[pre_treatment] - counterfactual_values[pre_treatment])**2)
                results[country]['pre_treatment_error'] = pre_treat_error
            
            if sum(post_treatment) > 0:
                # Estimated treatment effect
                treatment_effect = np.mean(actual_values[post_treatment] - counterfactual_values[post_treatment])
                results[country]['treatment_effect'] = treatment_effect
    
    return results


def run_gemini_placebo_test_ce2v(country: str, ce2v_result: Dict, data_dict: Dict, 
                                model: CausalEconomic2Vec, scaler: StandardScaler,
                                feature_cols: List[str], n_placebos: int = 20) -> Dict:
    """Run placebo tests using each country's actual treatment year and gemini countries."""
    
    if ce2v_result is None:
        return {'p_value_ratio': np.nan, 'placebos': []}
    
    panel = data_dict['panel']
    treated_countries = data_dict['treated_countries']
    treatment_year = ce2v_result['treatment_year']
    gemini_df = data_dict.get('gemini_df')
    
    if gemini_df is None:
        return {'p_value_ratio': np.nan, 'placebos': []}
    
    # Get donor pool from gemini data (reuse logic from donor_pools)
    country_map = {'Korea, Rep.': 'Korea', 'Russian Federation': 'Russia'}
    reverse_map = {v: k for k, v in country_map.items()}
    
    gemini_country = country_map.get(country, country)
    
    # Check which gemini data format we're using
    is_top_geminis = 'Gemini' in gemini_df.columns and 'Cluster' not in gemini_df.columns
    
    if is_top_geminis:
        # We're using top_geminis.csv
        gemini_matches = gemini_df[gemini_df['Country'] == gemini_country]['Gemini'].tolist()
        placebo_donors = [reverse_map.get(g, g) for g in gemini_matches 
                         if reverse_map.get(g, g) in panel.columns]
    else:
        # We're using gemini_clusters.csv
        try:
            row = gemini_df[gemini_df['Country'] == gemini_country]
            if row.empty:
                return {'p_value_ratio': np.nan, 'placebos': []}
            
            cluster = row['Cluster'].values[0]
            cluster_countries = gemini_df[gemini_df['Cluster'] == cluster]['Country'].tolist()
            placebo_donors = [reverse_map.get(c, c) for c in cluster_countries 
                            if c != gemini_country and 
                            reverse_map.get(c, c) in panel.columns]
        except Exception:
            return {'p_value_ratio': np.nan, 'placebos': []}
    
    # Remove treated countries
    placebo_donors = [d for d in placebo_donors if d not in treated_countries]
    placebo_countries = placebo_donors[:n_placebos]
    
    print(f"Using {len(placebo_countries)} placebo countries for {country}")
    
    # Store placebo results
    placebo_results = []
    
    # Device for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create country ID mapping
    df = data_dict['df']
    country_to_id = {country: i for i, country in enumerate(df['country'].unique())}
    
    # Find the index of the outcome column in feature_cols
    outcome_col = 'gdp_growth_annual_share'
    outcome_idx = feature_cols.index(outcome_col) if outcome_col in feature_cols else None
    
    # Run placebo tests
    for placebo_country in placebo_countries:
        # Skip if placebo not in dataset
        if placebo_country not in df['country'].unique() or placebo_country not in country_to_id:
            continue
            
        # Skip if placebo country is already treated
        placebo_is_treated = df[(df['country'] == placebo_country) & (df['has_platform'] == 1)].shape[0] > 0
        if placebo_is_treated:
            continue
        
        # Get country data
        country_data = df[df['country'] == placebo_country].sort_values('year')
        years = country_data['year'].values
        
        # Skip if insufficient data
        if len(years) < 5:
            continue
        
        # Skip if treatment year outside data range
        if treatment_year < min(years) or treatment_year > max(years):
            continue
            
        # Create mock treatment vector
        mock_treatment = np.zeros(len(years))
        mock_treatment[years >= treatment_year] = 1
        
        # Check if we have enough pre-treatment periods
        pre_treatment = years < treatment_year
        if sum(pre_treatment) < 3:
            continue
            
        # Generate placebo counterfactual
        try:
            with torch.no_grad():
                # Prepare data tensors
                scaled_features = scaler.transform(country_data[feature_cols])
                features = torch.FloatTensor(scaled_features).to(device)
                
                actual_treatment = torch.FloatTensor(mock_treatment).to(device)
                counterfactual_treatment = torch.zeros_like(actual_treatment).to(device)
                country_id = torch.LongTensor([country_to_id[placebo_country]]).to(device)
                
                # Get years and create normalized time
                time_norm = (years - years.min()) / (years.max() - years.min() + 1e-6)
                time = torch.FloatTensor(time_norm).to(device)
                
                # Add batch dimension
                features = features.unsqueeze(0)
                actual_treatment = actual_treatment.unsqueeze(0)
                counterfactual_treatment = counterfactual_treatment.unsqueeze(0)
                time = time.unsqueeze(0)
                
                # Generate counterfactual
                factual, counterfactual = model.get_counterfactual(
                    features, actual_treatment, counterfactual_treatment, country_id, time
                )
                
                # Extract values
                factual_values = factual.squeeze().cpu().numpy()
                counterfactual_values = counterfactual.squeeze().cpu().numpy()
                
                # Get actual outcome values
                actual_values = scaled_features[:, outcome_idx] if outcome_idx is not None else country_data[outcome_col].values
                
                # Inverse transform to original scale if outcome_col was in feature_cols
                if outcome_idx is not None:
                    outcome_mean = scaler.mean_[outcome_idx]
                    outcome_std = scaler.scale_[outcome_idx]
                    
                    actual_values = actual_values * outcome_std + outcome_mean
                    factual_values = factual_values * outcome_std + outcome_mean
                    counterfactual_values = counterfactual_values * outcome_std + outcome_mean
                
                # Calculate the effect (difference between actual and counterfactual)
                effect = actual_values - counterfactual_values
                
                # Calculate RMSPE for pre-treatment periods
                pre_rmse = np.sqrt(np.mean((actual_values[pre_treatment] - counterfactual_values[pre_treatment])**2))
                
                # Calculate post-treatment RMSPE if applicable
                post_treatment = years >= treatment_year
                if sum(post_treatment) > 0:
                    post_rmse = np.sqrt(np.mean((actual_values[post_treatment] - counterfactual_values[post_treatment])**2))
                    post_pre_ratio = post_rmse / pre_rmse if pre_rmse > 0 else None
                    avg_effect = np.mean(actual_values[post_treatment] - counterfactual_values[post_treatment])
                    
                    # Store results
                    placebo_results.append({
                        'country': placebo_country,
                        'treatment_year': treatment_year,
                        'pre_rmse': pre_rmse,
                        'post_rmse': post_rmse,
                        'rmse_ratio': post_pre_ratio,
                        'att': avg_effect
                    })
                    
        except Exception:
            continue
    
    # Calculate p-values
    if placebo_results:
        # For RMSE ratio
        ratio = ce2v_result.get('rmse_ratio', np.nan)
        if np.isnan(ratio):
            p_value_ratio = np.nan
        else:
            p_value_ratio = sum(1 for p in placebo_results 
                              if p.get('rmse_ratio') is not None and p['rmse_ratio'] >= ratio) / len(placebo_results)
        
        return {
            'p_value_ratio': p_value_ratio,
            'placebos': placebo_results,
            'placebo_countries': [p['country'] for p in placebo_results]
        }
    else:
        return {'p_value_ratio': np.nan, 'placebos': [], 'placebo_countries': []}


def run_causal_e2v_analysis(data_dict: Dict, donor_pools: Optional[Dict[str, List[str]]] = None,
                           embedding_dim: int = 32, epochs: int = 30, batch_size: int = 16,
                           seq_length: int = 5, balance_weight: float = 0.5,
                           train_ratio: float = 0.7, val_ratio: float = 0.15,
                           n_placebos: int = 20) -> CausalResults:
    """
    Complete causal economic embedding analysis pipeline.
    
    Parameters:
    -----------
    data_dict : dict
        Preprocessed data dictionary
    donor_pools : dict, optional
        Donor pools for placebo testing (if None, uses gemini data)
    embedding_dim : int
        Embedding dimension
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    seq_length : int
        Sequence length for temporal modeling
    balance_weight : float
        Weight for balance loss
    train_ratio : float
        Training data ratio
    val_ratio : float
        Validation data ratio
    n_placebos : int
        Number of placebo tests
        
    Returns:
    --------
    CausalResults : Standardized results object
    """
    results = CausalResults('Causal Economic2Vec')
    
    # Extract data
    df = data_dict['df']
    feature_cols = data_dict['feature_cols']
    
    # 1. Split by country to prevent data leakage
    countries = df['country'].unique()
    np.random.shuffle(countries)
    
    n_countries = len(countries)
    n_train = int(n_countries * train_ratio)
    n_val = int(n_countries * val_ratio)
    
    train_countries = countries[:n_train]
    val_countries = countries[n_train:n_train+n_val]
    test_countries = countries[n_train+n_val:]
    
    print(f"Split: {len(train_countries)} train countries, {len(val_countries)} val countries, {len(test_countries)} test countries")
    
    # Create separate dataframes for train/val/test
    train_df = df[df['country'].isin(train_countries)].copy()
    val_df = df[df['country'].isin(val_countries)].copy()
    test_df = df[df['country'].isin(test_countries)].copy()
    
    # 2. Fit scaler ONLY on training data features
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    # 3. Apply scaling to each dataset
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # 4. Create datasets from scaled dataframes
    train_dataset = PanelDataset(train_df, feature_cols, seq_length=seq_length)
    val_dataset = PanelDataset(val_df, feature_cols, seq_length=seq_length)
    
    # 5. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 6. Create model
    input_dim = len(feature_cols)
    num_countries = df['country'].nunique()
    
    model = CausalEconomic2Vec(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_countries=num_countries
    )
    
    # 7. Train model
    print(f"Training CE2V model for {epochs} epochs...")
    model = train_causal_e2v(model, train_loader, val_loader, epochs=epochs, balance_weight=balance_weight)
    
    # 8. For counterfactual generation, scale the full dataset using the same scaler
    df_scaled_full = df.copy()
    df_scaled_full[feature_cols] = scaler.transform(df_scaled_full[feature_cols])
    
    # 9. Generate counterfactuals using the scaled full dataset
    country_metadata = data_dict['country_metadata']
    counterfactual_results = generate_and_evaluate_counterfactuals(
        model, df_scaled_full, country_metadata, feature_cols, scaler
    )
    
    # 10. Run placebo tests for each treated country
    all_country_results = {}
    
    for country, ce2v_result in counterfactual_results.items():
        print(f"Running placebo tests for {country}...")
        
        # Calculate standard metrics
        years = ce2v_result['years']
        actual = ce2v_result['actual']
        counterfactual = ce2v_result['counterfactual']
        treatment_year = ce2v_result['treatment_year']
        
        # Calculate pre and post treatment periods
        pre_years = years < treatment_year
        post_years = years >= treatment_year
        
        # Calculate effects and RMSE metrics
        effect = actual - counterfactual
        pre_effects = effect[pre_years]
        post_effects = effect[post_years]
        
        pre_rmse = np.sqrt(np.mean(pre_effects**2)) if len(pre_effects) > 0 else np.nan
        post_rmse = np.sqrt(np.mean(post_effects**2)) if len(post_effects) > 0 else np.nan
        rmse_ratio = post_rmse / pre_rmse if pre_rmse > 0 else np.nan
        att = np.mean(post_effects) if len(post_effects) > 0 else np.nan
        
        # Add metrics to result
        ce2v_result.update({
            'pre_rmse': pre_rmse,
            'post_rmse': post_rmse,
            'rmse_ratio': rmse_ratio,
            'att': att
        })
        
        # Run placebo test
        placebo_result = run_gemini_placebo_test_ce2v(
            country, ce2v_result, data_dict, model, scaler, feature_cols, n_placebos
        )
        
        # Store results
        all_country_results[country] = {
            'ce2v_result': ce2v_result,
            'placebo_result': placebo_result
        }
        
        # Add to results object
        country_metrics = {
            'att': ce2v_result['att'],
            'pre_rmse': ce2v_result['pre_rmse'],
            'post_rmse': ce2v_result['post_rmse'],
            'rmse_ratio': ce2v_result['rmse_ratio'],
            'p_value': placebo_result.get('p_value_ratio', np.nan),
            'significant': placebo_result.get('p_value_ratio', 1) <= 0.1 if placebo_result.get('p_value_ratio') is not None else False
        }
        results.add_country_result(country, country_metrics)
    
    # Create summary table
    summary_data = []
    for country, country_results in all_country_results.items():
        ce2v_result = country_results['ce2v_result']
        placebo_result = country_results['placebo_result']
        
        p_value = placebo_result.get('p_value_ratio', np.nan)
        
        summary_data.append({
            'country': country,
            'ATE': ce2v_result['att'],
            'RMSE pre': ce2v_result['pre_rmse'],
            'RMSE post': ce2v_result['post_rmse'],
            'relation': ce2v_result['rmse_ratio'],
            'p-val': p_value,
            'significant': 'Yes' if p_value is not None and p_value <= 0.1 else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('p-val')
    
    results.summary_table = summary_df
    results.raw_results = all_country_results
    
    # Store additional CE2V-specific results
    results.model = model
    results.scaler = scaler
    results.feature_cols = feature_cols
    
    return results
