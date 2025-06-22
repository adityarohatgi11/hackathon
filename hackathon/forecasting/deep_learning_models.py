"""
Advanced Deep Learning Models for Energy Trading and Forecasting
===============================================================

This module provides state-of-the-art deep learning architectures for:
- Transformer-based time series forecasting
- Graph Neural Networks for grid topology modeling
- Variational Autoencoders for anomaly detection
- Reinforcement Learning with advanced architectures
- Multi-modal fusion networks
- Attention mechanisms and self-supervised learning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    import torch.distributions as dist
    TORCH_AVAILABLE = True
    logger.info("PyTorch available for deep learning models")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - deep learning models disabled")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available")


class EnergyTimeSeriesDataset(Dataset):
    """Custom dataset for energy time series data."""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 24, 
                 target_column: str = 'price', feature_columns: List[str] = None):
        """
        Initialize dataset.
        
        Args:
            data: Time series data
            sequence_length: Length of input sequences
            target_column: Column to predict
            feature_columns: Feature columns to use
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.target_column = target_column
        
        if feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
        
        # Normalize data
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        
        self.features = self.scaler_features.fit_transform(data[self.feature_columns])
        self.targets = self.scaler_target.fit_transform(data[[target_column]])
        
        logger.info(f"Dataset initialized: {len(self.features)} samples, "
                   f"{len(self.feature_columns)} features, seq_len={sequence_length}")
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.features[idx:idx + self.sequence_length])
        y = torch.FloatTensor(self.targets[idx + self.sequence_length])
        return X, y


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for time series."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.W_o(context)
        
        # Residual connection and layer norm
        return self.layer_norm(output + x)


class TransformerForecaster(nn.Module):
    """Transformer-based time series forecasting model."""
    
    def __init__(self, input_dim: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1,
                 max_seq_length: int = 1000):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_seq_length, d_model)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        logger.info(f"Transformer forecaster initialized: {d_model}D model, {n_layers} layers")
    
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        """Generate positional encoding for transformer."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, return_uncertainty=False):
        batch_size, seq_len, _ = x.size()
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.max_seq_length:
            pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_encoding
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Use last time step for prediction
        last_hidden = encoded[:, -1, :]
        
        # Prediction
        prediction = self.output_projection(last_hidden)
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(last_hidden)
            return prediction, uncertainty
        
        return prediction


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for anomaly detection in energy data."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"VAE initialized: {input_dim}D -> {latent_dim}D latent space")
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def anomaly_score(self, x):
        """Calculate anomaly score based on reconstruction error."""
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            recon_error = F.mse_loss(recon, x, reduction='none').sum(dim=1)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return recon_error + kl_div


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for modeling grid topology and power flow."""
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 64,
                 num_layers: int = 3, output_dim: int = 1):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Edge embedding
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        logger.info(f"GNN initialized: {node_features} node features, {num_layers} layers")
    
    def forward(self, node_features, edge_index, edge_features):
        # Embed nodes and edges
        x = F.relu(self.node_embedding(node_features))
        edge_attr = F.relu(self.edge_embedding(edge_features))
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Global pooling (mean)
        graph_embedding = torch.mean(x, dim=0, keepdim=True)
        
        # Output prediction
        output = self.output_layer(graph_embedding)
        return output


class GraphConvLayer(nn.Module):
    """Graph convolution layer."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.edge_linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        
        # Message passing
        messages = torch.cat([x[row], x[col]], dim=1)
        messages = self.linear(messages)
        
        # Edge features
        edge_messages = self.edge_linear(edge_attr)
        messages = messages + edge_messages
        
        # Aggregate messages
        out = torch.zeros_like(x)
        for i in range(x.size(0)):
            mask = col == i
            if mask.any():
                out[i] = torch.mean(messages[mask], dim=0)
        
        return out


class MultiModalFusionNetwork(nn.Module):
    """Multi-modal fusion network for combining different data types."""
    
    def __init__(self, time_series_dim: int, tabular_dim: int, text_dim: int = 0,
                 fusion_dim: int = 256, output_dim: int = 1):
        super().__init__()
        
        # Time series encoder (LSTM)
        self.ts_encoder = nn.LSTM(
            input_size=time_series_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Tabular data encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Text encoder (if available)
        if text_dim > 0:
            self.text_encoder = nn.Sequential(
                nn.Linear(text_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            fusion_input_dim = 128 + 64 + 64
        else:
            self.text_encoder = None
            fusion_input_dim = 128 + 64
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
        # Attention mechanism for fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_input_dim,
            num_heads=8,
            batch_first=True
        )
        
        logger.info(f"Multi-modal fusion network initialized")
    
    def forward(self, time_series, tabular, text=None):
        # Encode time series
        ts_out, (h_n, c_n) = self.ts_encoder(time_series)
        ts_features = h_n[-1]  # Use last hidden state
        
        # Encode tabular data
        tab_features = self.tabular_encoder(tabular)
        
        # Combine features
        if text is not None and self.text_encoder is not None:
            text_features = self.text_encoder(text)
            combined_features = torch.cat([ts_features, tab_features, text_features], dim=1)
        else:
            combined_features = torch.cat([ts_features, tab_features], dim=1)
        
        # Apply attention (self-attention on combined features)
        combined_features = combined_features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.attention(combined_features, combined_features, combined_features)
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Final prediction
        output = self.fusion_network(attended_features)
        return output


class AdvancedRLAgent(nn.Module):
    """Advanced Reinforcement Learning agent with modern architectures."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512,
                 use_attention: bool = True, use_distributional: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_attention = use_attention
        self.use_distributional = use_distributional
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = MultiHeadAttention(hidden_dim, n_heads=8)
        
        # Value and advantage streams (Dueling DQN)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        if use_distributional:
            # Distributional RL (C51)
            self.num_atoms = 51
            self.v_min = -100
            self.v_max = 100
            self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim * self.num_atoms)
            )
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        
        logger.info(f"Advanced RL agent initialized: {state_dim}D state, {action_dim} actions")
    
    def forward(self, state):
        batch_size = state.size(0)
        
        # Encode state
        encoded = self.state_encoder(state)
        
        # Apply attention if enabled
        if self.use_attention:
            encoded = encoded.unsqueeze(1)  # Add sequence dimension
            encoded = self.attention(encoded)
            encoded = encoded.squeeze(1)  # Remove sequence dimension
        
        # Value and advantage streams
        value = self.value_stream(encoded)
        
        if self.use_distributional:
            advantage = self.advantage_stream(encoded)
            advantage = advantage.view(batch_size, self.action_dim, self.num_atoms)
            
            # Combine value and advantage
            q_dist = value.unsqueeze(1) + advantage - advantage.mean(dim=1, keepdim=True)
            q_dist = F.softmax(q_dist, dim=2)
            
            return q_dist
        else:
            advantage = self.advantage_stream(encoded)
            
            # Combine value and advantage
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            
            return q_values
    
    def get_action(self, state, epsilon=0.0):
        """Get action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            if self.use_distributional:
                q_dist = self.forward(state.unsqueeze(0))
                q_values = (q_dist * self.support.to(q_dist.device)).sum(dim=2)
            else:
                q_values = self.forward(state.unsqueeze(0))
            
            return q_values.argmax().item()


class DeepLearningEnsemble:
    """Ensemble of deep learning models for robust predictions."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        
        logger.info(f"Deep learning ensemble initialized with {len(models)} models")
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make ensemble predictions.
        
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Weighted average
        weights = torch.tensor(self.weights, device=predictions.device)
        mean_pred = torch.sum(predictions * weights.view(-1, 1, 1), dim=0)
        
        # Uncertainty as standard deviation
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty


def create_deep_learning_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive deep learning system for energy trading.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing all initialized models
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available - returning empty system")
        return {}
    
    system = {}
    
    # Transformer forecaster
    if config.get('use_transformer', True):
        system['transformer'] = TransformerForecaster(
            input_dim=config.get('input_dim', 10),
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4)
        )
    
    # VAE for anomaly detection
    if config.get('use_vae', True):
        system['vae'] = VariationalAutoencoder(
            input_dim=config.get('input_dim', 10),
            latent_dim=config.get('latent_dim', 16)
        )
    
    # Advanced RL agent
    if config.get('use_rl', True):
        system['rl_agent'] = AdvancedRLAgent(
            state_dim=config.get('state_dim', 23),
            action_dim=config.get('action_dim', 5),
            use_attention=config.get('use_attention', True),
            use_distributional=config.get('use_distributional', True)
        )
    
    # Multi-modal fusion
    if config.get('use_multimodal', True):
        system['multimodal'] = MultiModalFusionNetwork(
            time_series_dim=config.get('ts_dim', 5),
            tabular_dim=config.get('tab_dim', 10)
        )
    
    logger.info(f"Deep learning system created with {len(system)} models")
    return system


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 100, lr: float = 0.001, device: str = 'cpu') -> Dict[str, List[float]]:
    """
    Train a deep learning model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Training history
    """
    if not TORCH_AVAILABLE:
        return {'train_loss': [], 'val_loss': []}
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    return history 