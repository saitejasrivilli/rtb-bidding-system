"""
CTR Prediction Model for Real-Time Bidding
Implements a deep learning model with embeddings for user, ad, and context features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


class CTRPredictor(nn.Module):
    """
    Deep neural network for Click-Through Rate prediction
    Uses embeddings for categorical features and deep layers for interaction learning
    """
    
    def __init__(
        self,
        n_users: int,
        n_ads: int,
        n_contexts: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3
    ):
        super(CTRPredictor, self).__init__()
        
        # Embedding layers for categorical features
        self.embedding_layers = nn.ModuleDict({
            'user': nn.Embedding(n_users, embedding_dim),
            'ad': nn.Embedding(n_ads, embedding_dim),
            'context': nn.Embedding(n_contexts, embedding_dim // 2)
        })
        
        # Calculate total embedding dimension
        total_embedding_dim = embedding_dim * 2 + embedding_dim // 2
        
        # Deep neural network
        layers = []
        input_dim = total_embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.dnn = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def get_embeddings(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get embeddings for all categorical features
        
        Args:
            features: Dictionary with 'user_id', 'ad_id', 'context_id'
        
        Returns:
            Concatenated embeddings tensor
        """
        embeddings = []
        
        if 'user_id' in features:
            embeddings.append(self.embedding_layers['user'](features['user_id']))
        
        if 'ad_id' in features:
            embeddings.append(self.embedding_layers['ad'](features['ad_id']))
        
        if 'context_id' in features:
            embeddings.append(self.embedding_layers['context'](features['context_id']))
        
        return torch.cat(embeddings, dim=1)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: Dictionary of feature tensors
        
        Returns:
            Predicted CTR (probability between 0 and 1)
        """
        embeddings = self.get_embeddings(features)
        ctr = self.dnn(embeddings)
        return ctr.squeeze(-1)


class FMCTRPredictor(nn.Module):
    """
    Factorization Machine based CTR predictor
    Captures feature interactions more efficiently
    """
    
    def __init__(
        self,
        n_users: int,
        n_ads: int,
        n_contexts: int,
        embedding_dim: int = 64,
        use_deep: bool = True
    ):
        super(FMCTRPredictor, self).__init__()
        
        self.n_features = 3  # user, ad, context
        self.use_deep = use_deep
        
        # First-order embeddings (linear)
        self.linear_embeddings = nn.ModuleDict({
            'user': nn.Embedding(n_users, 1),
            'ad': nn.Embedding(n_ads, 1),
            'context': nn.Embedding(n_contexts, 1)
        })
        
        # Second-order embeddings (interactions)
        self.interaction_embeddings = nn.ModuleDict({
            'user': nn.Embedding(n_users, embedding_dim),
            'ad': nn.Embedding(n_ads, embedding_dim),
            'context': nn.Embedding(n_contexts, embedding_dim)
        })
        
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Optional deep component
        if use_deep:
            self.deep_layers = nn.Sequential(
                nn.Linear(embedding_dim * 3, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass implementing FM with optional deep component
        """
        # First-order (linear) part
        linear_terms = []
        interaction_embs = []
        
        for key, emb_key in [('user_id', 'user'), ('ad_id', 'ad'), ('context_id', 'context')]:
            if key in features:
                linear_terms.append(self.linear_embeddings[emb_key](features[key]))
                interaction_embs.append(self.interaction_embeddings[emb_key](features[key]))
        
        linear_sum = sum(linear_terms).squeeze(-1)
        
        # Second-order (interaction) part
        # FM: sum of (sum of embeddings)^2 - sum of (embeddings^2)
        stacked_embs = torch.stack(interaction_embs, dim=1)  # [batch, n_features, dim]
        sum_of_embs = torch.sum(stacked_embs, dim=1)  # [batch, dim]
        sum_of_squares = torch.sum(sum_of_embs ** 2, dim=1)  # [batch]
        
        square_of_sums = torch.sum(stacked_embs ** 2, dim=[1, 2])  # [batch]
        
        interaction = 0.5 * (sum_of_squares - square_of_sums)
        
        # Combine
        output = self.bias + linear_sum + interaction
        
        # Add deep component
        if self.use_deep:
            concat_embs = torch.cat(interaction_embs, dim=1)
            deep_output = self.deep_layers(concat_embs).squeeze(-1)
            output = output + deep_output
        
        return torch.sigmoid(output)


class CTRTrainer:
    """Trainer for CTR prediction models"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            features = {k: v.to(self.device) for k, v in batch['features'].items()}
            labels = batch['labels'].to(self.device).float()
            
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = {k: v.to(self.device) for k, v in batch['features'].items()}
                labels = batch['labels'].to(self.device).float()
                
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        from sklearn.metrics import roc_auc_score, log_loss
        
        auc = roc_auc_score(all_labels, all_predictions)
        logloss = log_loss(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'logloss': logloss
        }
    
    def fit(self, train_loader, val_loader, epochs: int = 10, early_stopping_patience: int = 3):
        """Train the model with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), '/home/claude/rtb_system/models/best_ctr_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('/home/claude/rtb_system/models/best_ctr_model.pt'))
        return self.model
