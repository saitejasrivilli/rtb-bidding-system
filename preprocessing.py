"""
Data Preprocessing for RTB Datasets
Handles iPinYou, Avazu, and other RTB datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path


class RTBDataset(Dataset):
    """
    PyTorch Dataset for RTB data
    """
    
    def __init__(
        self,
        features: Dict[str, np.ndarray],
        labels: np.ndarray
    ):
        """
        Args:
            features: Dictionary of feature arrays
            labels: Click labels (0 or 1)
        """
        self.features = features
        self.labels = labels
        
        # Validate shapes
        n_samples = len(labels)
        for key, arr in features.items():
            assert len(arr) == n_samples, f"Feature {key} has wrong length"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features_dict = {
            key: torch.tensor(arr[idx], dtype=torch.long)
            for key, arr in self.features.items()
        }
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {
            'features': features_dict,
            'labels': label
        }


class iPinYouDataProcessor:
    """
    Data processor for iPinYou RTB dataset
    """
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.ad_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()
        self.city_encoder = LabelEncoder()
        
        self.encoders_fitted = False
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load iPinYou dataset
        
        Format: click, winning_price, hour, user_id, user_tags, ad_slot_id, etc.
        """
        # iPinYou format (tab-separated)
        columns = [
            'click', 'winning_price', 'hour', 'log_type', 'ip',
            'region', 'city', 'ad_exchange', 'domain', 'url',
            'anonymous_url', 'ad_slot_id', 'ad_slot_width', 'ad_slot_height',
            'ad_slot_visibility', 'ad_slot_format', 'paying_price',
            'creative_id', 'user_tags'
        ]
        
        try:
            df = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                names=columns,
                na_values=['null', 'NULL', ''],
                low_memory=False
            )
            print(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            # Generate synthetic data for demo
            return self._generate_synthetic_data(100000)
    
    def _generate_synthetic_data(self, n_samples: int = 100000) -> pd.DataFrame:
        """Generate synthetic data for demonstration"""
        print(f"Generating {n_samples} synthetic samples...")
        
        np.random.seed(42)
        
        # Generate features
        data = {
            'click': np.random.binomial(1, 0.01, n_samples),  # 1% CTR
            'winning_price': np.random.lognormal(2, 1, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'region': np.random.randint(0, 50, n_samples),
            'city': np.random.randint(0, 200, n_samples),
            'ad_slot_id': np.random.randint(0, 100, n_samples),
            'ad_slot_width': np.random.choice([300, 728, 160], n_samples),
            'ad_slot_height': np.random.choice([250, 90, 600], n_samples),
            'creative_id': np.random.randint(0, 500, n_samples),
            'paying_price': np.random.lognormal(2, 1, n_samples),
            'user_tags': np.random.randint(0, 1000, n_samples)
        }
        
        df = pd.DataFrame(data)
        return df
    
    def preprocess(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Preprocess the data
        
        Args:
            df: Raw dataframe
            fit: Whether to fit encoders (True for training, False for test)
        
        Returns:
            features dict, labels array
        """
        print("Preprocessing data...")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Encode categorical features
        if fit:
            df['user_id_encoded'] = self.user_encoder.fit_transform(
                df['user_tags'].astype(str)
            )
            df['ad_id_encoded'] = self.ad_encoder.fit_transform(
                df['creative_id'].astype(str)
            )
            df['region_encoded'] = self.region_encoder.fit_transform(
                df['region'].astype(str)
            )
            df['city_encoded'] = self.city_encoder.fit_transform(
                df['city'].astype(str)
            )
            self.encoders_fitted = True
        else:
            if not self.encoders_fitted:
                raise ValueError("Encoders not fitted. Call with fit=True first.")
            
            # Handle unseen labels
            df['user_id_encoded'] = df['user_tags'].astype(str).apply(
                lambda x: self._safe_transform(self.user_encoder, x)
            )
            df['ad_id_encoded'] = df['creative_id'].astype(str).apply(
                lambda x: self._safe_transform(self.ad_encoder, x)
            )
            df['region_encoded'] = df['region'].astype(str).apply(
                lambda x: self._safe_transform(self.region_encoder, x)
            )
            df['city_encoded'] = df['city'].astype(str).apply(
                lambda x: self._safe_transform(self.city_encoder, x)
            )
        
        # Create context features (combination of region + city + hour)
        df['context_id'] = (
            df['region_encoded'] * 1000 +
            df['city_encoded'] * 100 +
            df['hour']
        )
        
        # Extract features
        features = {
            'user_id': df['user_id_encoded'].values,
            'ad_id': df['ad_id_encoded'].values,
            'context_id': df['context_id'].values
        }
        
        # Extract labels
        labels = df['click'].values
        
        print(f"Processed {len(labels)} samples")
        print(f"CTR: {np.mean(labels):.4f}")
        print(f"Unique users: {len(np.unique(features['user_id']))}")
        print(f"Unique ads: {len(np.unique(features['ad_id']))}")
        print(f"Unique contexts: {len(np.unique(features['context_id']))}")
        
        return features, labels
    
    def _safe_transform(self, encoder: LabelEncoder, value: str) -> int:
        """Safely transform value, handling unseen labels"""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # Return 0 for unseen labels
            return 0
    
    def save_encoders(self, path: str):
        """Save fitted encoders"""
        encoders = {
            'user': self.user_encoder,
            'ad': self.ad_encoder,
            'region': self.region_encoder,
            'city': self.city_encoder
        }
        
        with open(path, 'wb') as f:
            pickle.dump(encoders, f)
        
        print(f"Encoders saved to {path}")
    
    def load_encoders(self, path: str):
        """Load fitted encoders"""
        with open(path, 'rb') as f:
            encoders = pickle.load(f)
        
        self.user_encoder = encoders['user']
        self.ad_encoder = encoders['ad']
        self.region_encoder = encoders['region']
        self.city_encoder = encoders['city']
        self.encoders_fitted = True
        
        print(f"Encoders loaded from {path}")


class AvazuDataProcessor:
    """
    Data processor for Avazu CTR dataset
    """
    
    def __init__(self):
        self.feature_encoders = {}
        self.encoders_fitted = False
    
    def load_data(self, file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load Avazu dataset"""
        try:
            df = pd.read_csv(file_path, nrows=nrows)
            print(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._generate_synthetic_data(100000)
    
    def _generate_synthetic_data(self, n_samples: int = 100000) -> pd.DataFrame:
        """Generate synthetic Avazu-like data"""
        print(f"Generating {n_samples} synthetic Avazu samples...")
        
        np.random.seed(42)
        
        data = {
            'click': np.random.binomial(1, 0.17, n_samples),  # 17% CTR
            'hour': np.random.randint(14100000, 14103000, n_samples),
            'C1': np.random.randint(1000, 1010, n_samples),
            'banner_pos': np.random.randint(0, 3, n_samples),
            'site_id': np.random.randint(0, 1000, n_samples),
            'site_domain': np.random.randint(0, 500, n_samples),
            'site_category': np.random.randint(0, 50, n_samples),
            'app_id': np.random.randint(0, 100, n_samples),
            'app_domain': np.random.randint(0, 50, n_samples),
            'app_category': np.random.randint(0, 30, n_samples),
            'device_id': np.random.randint(0, 10000, n_samples),
            'device_model': np.random.randint(0, 1000, n_samples),
            'device_type': np.random.randint(0, 5, n_samples),
            'device_conn_type': np.random.randint(0, 4, n_samples),
            'C14': np.random.randint(0, 1000, n_samples),
            'C15': np.random.randint(300, 500, n_samples),
            'C16': np.random.randint(30, 50, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def preprocess(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Preprocess Avazu data"""
        print("Preprocessing Avazu data...")
        
        # Features to encode
        categorical_features = [
            'site_id', 'site_domain', 'site_category',
            'app_id', 'app_domain', 'app_category',
            'device_model', 'device_type'
        ]
        
        if fit:
            for feature in categorical_features:
                if feature in df.columns:
                    encoder = LabelEncoder()
                    df[f'{feature}_encoded'] = encoder.fit_transform(
                        df[feature].astype(str)
                    )
                    self.feature_encoders[feature] = encoder
            self.encoders_fitted = True
        else:
            for feature in categorical_features:
                if feature in df.columns and feature in self.feature_encoders:
                    df[f'{feature}_encoded'] = df[feature].astype(str).apply(
                        lambda x: self._safe_transform(self.feature_encoders[feature], x)
                    )
        
        # Create simplified features for CTR model
        features = {
            'user_id': df['device_model_encoded'].values if 'device_model_encoded' in df else np.zeros(len(df)),
            'ad_id': df['site_id_encoded'].values if 'site_id_encoded' in df else np.zeros(len(df)),
            'context_id': df['banner_pos'].values if 'banner_pos' in df else np.zeros(len(df))
        }
        
        labels = df['click'].values
        
        print(f"Processed {len(labels)} samples")
        print(f"CTR: {np.mean(labels):.4f}")
        
        return features, labels
    
    def _safe_transform(self, encoder: LabelEncoder, value: str) -> int:
        """Safely transform value"""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return 0


def create_dataloaders(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    batch_size: int = 256,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        features: Feature dictionary
        labels: Labels array
        batch_size: Batch size
        test_size: Fraction for test set
        val_size: Fraction of remaining for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split data
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    # Train/test split
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    # Train/val split
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=42,
        stratify=labels[train_val_idx]
    )
    
    # Create datasets
    def create_dataset(idx):
        subset_features = {k: v[idx] for k, v in features.items()}
        subset_labels = labels[idx]
        return RTBDataset(subset_features, subset_labels)
    
    train_dataset = create_dataset(train_idx)
    val_dataset = create_dataset(val_idx)
    test_dataset = create_dataset(test_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def calculate_feature_stats(features: Dict[str, np.ndarray]) -> Dict:
    """Calculate statistics about features"""
    stats = {}
    
    for key, arr in features.items():
        stats[key] = {
            'unique_values': len(np.unique(arr)),
            'min': np.min(arr),
            'max': np.max(arr),
            'mean': np.mean(arr),
            'std': np.std(arr)
        }
    
    return stats


def balance_dataset(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    strategy: str = 'undersample'
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Balance dataset for imbalanced classes
    
    Args:
        features: Feature dictionary
        labels: Labels
        strategy: 'undersample' or 'oversample'
    
    Returns:
        Balanced features and labels
    """
    positive_idx = np.where(labels == 1)[0]
    negative_idx = np.where(labels == 0)[0]
    
    n_positive = len(positive_idx)
    n_negative = len(negative_idx)
    
    print(f"Original - Positive: {n_positive}, Negative: {n_negative}")
    
    if strategy == 'undersample':
        # Undersample majority class
        if n_negative > n_positive:
            negative_idx = np.random.choice(negative_idx, n_positive, replace=False)
        else:
            positive_idx = np.random.choice(positive_idx, n_negative, replace=False)
    
    elif strategy == 'oversample':
        # Oversample minority class
        if n_positive < n_negative:
            additional = np.random.choice(positive_idx, n_negative - n_positive, replace=True)
            positive_idx = np.concatenate([positive_idx, additional])
        else:
            additional = np.random.choice(negative_idx, n_positive - n_negative, replace=True)
            negative_idx = np.concatenate([negative_idx, additional])
    
    # Combine
    balanced_idx = np.concatenate([positive_idx, negative_idx])
    np.random.shuffle(balanced_idx)
    
    balanced_features = {k: v[balanced_idx] for k, v in features.items()}
    balanced_labels = labels[balanced_idx]
    
    print(f"Balanced - Positive: {np.sum(balanced_labels == 1)}, Negative: {np.sum(balanced_labels == 0)}")
    
    return balanced_features, balanced_labels
