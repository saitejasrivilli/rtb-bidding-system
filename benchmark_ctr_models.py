#!/usr/bin/env python3
"""
Benchmark CTR prediction models: DNN vs GBDT variants
Generates synthetic RTB data and compares performance metrics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ctr_model import CTRPredictor, CTRTrainer
from gbdt_ctr_model import (
    GBDTCTRPredictor, LGBMCTRPredictor, CatBoostCTRPredictor,
    CTRModelComparator
)


def generate_synthetic_rtb_data(n_samples=10000):
    """
    Generate synthetic RTB dataset with realistic features
    Mimics structure of real RTB data (users, ads, context)
    """
    np.random.seed(42)

    n_users = 500
    n_ads = 200
    n_contexts = 100

    # Categorical features
    user_ids = np.random.randint(0, n_users, n_samples)
    ad_ids = np.random.randint(0, n_ads, n_samples)
    context_ids = np.random.randint(0, n_contexts, n_samples)

    # Continuous features
    hour_of_day = np.random.randint(0, 24, n_samples)
    user_history = np.random.exponential(5, n_samples)
    ad_quality = np.random.uniform(0, 1, n_samples)

    # Generate target with realistic CTR (~3%)
    # Base CTR
    base_ctr = 0.03

    # Feature interaction: user affinity + ad quality
    user_affinity = np.random.uniform(0, 1, n_samples)
    time_factor = np.sin(hour_of_day * np.pi / 12) * 0.3 + 0.7
    noise = np.random.normal(0, 0.1, n_samples)

    click_prob = base_ctr * (1 + user_affinity * 2) * (ad_quality * 1.5) * time_factor + noise
    click_prob = np.clip(click_prob, 0.001, 0.5)

    clicks = (np.random.uniform(0, 1, n_samples) < click_prob).astype(int)

    # Create feature matrix for GBDT models
    X_gbdt = np.column_stack([
        user_ids, ad_ids, context_ids,
        hour_of_day, user_history, ad_quality,
        user_affinity, time_factor
    ])

    # Feature dict for DNN
    X_dnn = {
        'user_id': torch.LongTensor(user_ids),
        'ad_id': torch.LongTensor(ad_ids),
        'context_id': torch.LongTensor(context_ids),
    }

    y = clicks

    return (X_gbdt, X_dnn, y, n_users, n_ads, n_contexts)


class DNNDataset(torch.utils.data.Dataset):
    """Custom dataset for DNN that returns proper format"""
    def __init__(self, features_dict, labels):
        self.features = features_dict
        self.labels = torch.FloatTensor(labels)
        self.n_samples = len(labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'features': {k: v[idx] for k, v in self.features.items()},
            'labels': self.labels[idx]
        }


def train_dnn_model(X_dnn, y, n_users, n_ads, n_contexts, X_test_dnn, y_test):
    """Train DNN CTR predictor"""
    print("\n" + "=" * 60)
    print("Training DNN CTR Model")
    print("=" * 60)

    # Split data
    n_train = int(0.7 * len(y))
    n_val = int(0.15 * len(y))

    # Create train/val/test splits
    train_indices = np.arange(n_train)
    val_indices = np.arange(n_train, n_train + n_val)

    X_train_dnn = {k: v[train_indices] for k, v in X_dnn.items()}
    y_train = y[train_indices]

    X_val_dnn = {k: v[val_indices] for k, v in X_dnn.items()}
    y_val = y[val_indices]

    # Create DataLoader
    train_dataset = DNNDataset(X_train_dnn, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = DNNDataset(X_val_dnn, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train model
    start_time = time.time()

    model = CTRPredictor(n_users, n_ads, n_contexts)
    trainer = CTRTrainer(model, learning_rate=0.001)

    for epoch in range(10):
        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        print(f"Epoch {epoch+1}/10 | Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc']:.4f}")

    training_time = time.time() - start_time

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_batch = {k: v.unsqueeze(0) for k, v in X_test_dnn.items()}
        y_pred_dnn = []
        for i in range(len(y_test)):
            test_batch_single = {k: v[i:i+1] for k, v in X_test_dnn.items()}
            y_pred_dnn.append(model(test_batch_single).item())

    y_pred_dnn = np.array(y_pred_dnn)

    test_metrics = CTRModelComparator.evaluate(y_test, y_pred_dnn)
    test_metrics['model'] = 'DNN'
    test_metrics['training_time_s'] = training_time

    return pd.Series(test_metrics), y_pred_dnn


def train_gbdt_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate GBDT models"""
    print("\n" + "=" * 60)
    print("Training GBDT Models")
    print("=" * 60)

    models = {}

    # XGBoost
    print("\n[1/3] XGBoost...")
    xgb_model = GBDTCTRPredictor()
    xgb_model.fit(X_train, y_train, X_val, y_val, epochs=100)
    models['XGBoost'] = xgb_model

    # LightGBM
    print("[2/3] LightGBM...")
    lgb_model = LGBMCTRPredictor()
    lgb_model.fit(X_train, y_train, X_val, y_val, epochs=100)
    models['LightGBM'] = lgb_model

    # CatBoost
    print("[3/3] CatBoost...")
    cb_model = CatBoostCTRPredictor()
    cb_model.fit(X_train, y_train, X_val, y_val, epochs=100)
    models['CatBoost'] = cb_model

    # Evaluate all
    comparison_df = CTRModelComparator.compare_models(models, X_test, y_test)
    return comparison_df


def main():
    print("\n" + "=" * 60)
    print("CTR Prediction Model Benchmark: DNN vs GBDT")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic RTB data...")
    X_gbdt, X_dnn, y, n_users, n_ads, n_contexts = generate_synthetic_rtb_data(n_samples=10000)

    # Split for GBDT models
    X_train_gbdt, X_test_gbdt, y_train_gbdt, y_test_gbdt = train_test_split(
        X_gbdt, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_gbdt, X_val_gbdt, y_train_gbdt, y_val_gbdt = train_test_split(
        X_train_gbdt, y_train_gbdt, test_size=0.15/0.8, random_state=42, stratify=y_train_gbdt
    )

    # For DNN, create test set from same indices
    test_indices = np.arange(len(y_test_gbdt)) + (len(y) - len(y_test_gbdt))
    X_test_dnn = {k: v[test_indices] for k, v in X_dnn.items()}
    y_test_dnn = y[test_indices]

    # Train models
    dnn_results, _ = train_dnn_model(
        X_dnn, y_train_gbdt, n_users, n_ads, n_contexts,
        X_test_dnn, y_test_dnn
    )

    gbdt_results = train_gbdt_models(
        X_train_gbdt, y_train_gbdt,
        X_val_gbdt, y_val_gbdt,
        X_test_gbdt, y_test_gbdt
    )

    # Combine results
    all_results = pd.concat([
        gbdt_results,
        pd.DataFrame([dnn_results])
    ], ignore_index=True)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(all_results.to_string(index=False))

    # Save results
    all_results.to_csv('/Users/saitejasrivillibhutturu/Downloads/real-time-bidding-system/results/ctr_model_comparison.csv', index=False)
    print("\n✓ Results saved to: results/ctr_model_comparison.csv")

    # Print summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    best_auc_model = all_results.loc[all_results['auc'].idxmax()]
    fastest_model = all_results.loc[all_results['inference_time_ms'].idxmin()]

    print(f"\nBest AUC: {best_auc_model['model']} ({best_auc_model['auc']:.4f})")
    print(f"Fastest Inference: {fastest_model['model']} ({fastest_model['inference_time_ms']:.3f}ms per sample)")

    print("\nLatency Comparison:")
    for _, row in all_results.iterrows():
        print(f"  {row['model']:12} {row['inference_time_ms']:7.3f} ms/sample")

    print("\nAUC Comparison:")
    for _, row in all_results.iterrows():
        print(f"  {row['model']:12} {row['auc']:.4f}")

    return all_results


if __name__ == '__main__':
    results = main()
