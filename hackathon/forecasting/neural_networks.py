"""
Advanced Neural Networks for Energy Trading
==========================================

This module provides deep learning architectures including:
- LSTM/GRU networks for time series
- Transformer architectures
- Convolutional networks for pattern recognition
- Ensemble methods
- Advanced optimization techniques
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    logger.info("PyTorch available for neural networks")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - neural networks disabled")


class LSTMForecaster(nn.Module):
    """LSTM-based forecasting network."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 output_size: int = 1, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            batch_first=True
        )
        
        logger.info(f"LSTM Forecaster initialized: {input_size}D input, {hidden_size}D hidden")
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step
        last_output = attended_out[:, -1, :]
        
        # Final prediction
        prediction = self.output_layer(last_output)
        
        return prediction


class ConvolutionalForecaster(nn.Module):
    """1D CNN for time series pattern recognition."""
    
    def __init__(self, input_channels: int, sequence_length: int, output_size: int = 1):
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )
        
        logger.info(f"CNN Forecaster initialized: {input_channels} channels, {sequence_length} length")
    
    def forward(self, x):
        # Reshape for conv1d: (batch, channels, sequence)
        if x.dim() == 3:
            x = x.transpose(1, 2)
        
        # Convolutional layers
        conv_out = self.conv_layers(x)
        
        # Flatten
        flattened = conv_out.view(conv_out.size(0), -1)
        
        # Fully connected layers
        output = self.fc_layers(flattened)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block for time series."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class EnergyTransformer(nn.Module):
    """Transformer architecture for energy forecasting."""
    
    def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 4, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        logger.info(f"Energy Transformer initialized: {d_model}D model, {n_layers} layers")
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class EnsembleNetwork(nn.Module):
    """Ensemble of multiple neural networks."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if weights is None:
            self.weights = torch.ones(self.num_models) / self.num_models
        else:
            self.weights = torch.tensor(weights)
        
        # Meta-learner for dynamic weighting
        self.meta_learner = nn.Sequential(
            nn.Linear(self.num_models, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_models),
            nn.Softmax(dim=1)
        )
        
        logger.info(f"Ensemble network initialized with {self.num_models} models")
    
    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=2)  # (batch, output, num_models)
        
        # Simple weighted average
        weighted_pred = torch.sum(predictions * self.weights.to(x.device), dim=2)
        
        # Dynamic weighting (optional)
        model_scores = torch.mean(predictions, dim=1)  # Average across output dimensions
        dynamic_weights = self.meta_learner(model_scores)
        dynamic_pred = torch.sum(predictions * dynamic_weights.unsqueeze(1), dim=2)
        
        return weighted_pred, dynamic_pred


class VariationalAutoencoder(nn.Module):
    """VAE for anomaly detection and data generation."""
    
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
        """Calculate anomaly score."""
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            recon_error = F.mse_loss(recon, x, reduction='none').sum(dim=1)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return recon_error + kl_div


class NeuralNetworkTrainer:
    """Trainer class for neural networks."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001, 
                 device: str = 'cpu', patience: int = 10):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=patience//2, factor=0.5
        )
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Neural network trainer initialized on {device}")
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, criterion: nn.Module = None) -> Dict[str, List[float]]:
        """Train the model."""
        if criterion is None:
            criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return history


def create_neural_network_ensemble(input_dim: int, config: Dict[str, Any]) -> EnsembleNetwork:
    """Create an ensemble of neural networks."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available - cannot create ensemble")
        return None
    
    models = []
    
    # LSTM model
    if config.get('use_lstm', True):
        lstm_model = LSTMForecaster(
            input_size=input_dim,
            hidden_size=config.get('lstm_hidden', 128),
            num_layers=config.get('lstm_layers', 2)
        )
        models.append(lstm_model)
    
    # CNN model
    if config.get('use_cnn', True):
        cnn_model = ConvolutionalForecaster(
            input_channels=input_dim,
            sequence_length=config.get('sequence_length', 24)
        )
        models.append(cnn_model)
    
    # Transformer model
    if config.get('use_transformer', True):
        transformer_model = EnergyTransformer(
            input_dim=input_dim,
            d_model=config.get('d_model', 256),
            n_layers=config.get('transformer_layers', 4)
        )
        models.append(transformer_model)
    
    if models:
        ensemble = EnsembleNetwork(models)
        logger.info(f"Neural network ensemble created with {len(models)} models")
        return ensemble
    else:
        logger.warning("No models added to ensemble")
        return None


def create_anomaly_detector(input_dim: int, latent_dim: int = 32) -> VariationalAutoencoder:
    """Create VAE-based anomaly detector."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available - cannot create anomaly detector")
        return None
    
    vae = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    logger.info(f"VAE anomaly detector created: {input_dim}D -> {latent_dim}D")
    return vae 