#!/usr/bin/env python3
"""
Benchmark GBDT CTR Models: XGBoost vs LightGBM vs CatBoost
Lightweight comparison without DNN (avoids PyTorch segfaults)
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def generate_synthetic_rtb_data(n_samples=10000):
    """Generate synthetic RTB dataset"""
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
    base_ctr = 0.03
    user_affinity = np.random.uniform(0, 1, n_samples)
    time_factor = np.sin(hour_of_day * np.pi / 12) * 0.3 + 0.7
    noise = np.random.normal(0, 0.1, n_samples)

    click_prob = base_ctr * (1 + user_affinity * 2) * (ad_quality * 1.5) * time_factor + noise
    click_prob = np.clip(click_prob, 0.001, 0.5)

    clicks = (np.random.uniform(0, 1, n_samples) < click_prob).astype(int)

    # Create feature matrix
    X = np.column_stack([
        user_ids, ad_ids, context_ids,
        hour_of_day, user_history, ad_quality,
        user_affinity, time_factor
    ])

    return X, clicks


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate XGBoost"""
    print("Training XGBoost...")
    start = time.time()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'verbosity': 0
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    train_time = time.time() - start

    # Inference
    start_inf = time.time()
    y_pred = model.predict(dtest)
    inf_time = (time.time() - start_inf) / len(X_test) * 1000

    auc = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    gini = 2 * auc - 1

    return {
        'model': 'XGBoost',
        'auc': auc,
        'logloss': logloss,
        'gini': gini,
        'training_time_s': train_time,
        'inference_time_ms': inf_time,
        'n_trees': model.best_iteration
    }


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate LightGBM"""
    print("Training LightGBM...")
    start = time.time()

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbose': -1
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(10)]
    )

    train_time = time.time() - start

    # Inference
    start_inf = time.time()
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    inf_time = (time.time() - start_inf) / len(X_test) * 1000

    auc = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    gini = 2 * auc - 1

    return {
        'model': 'LightGBM',
        'auc': auc,
        'logloss': logloss,
        'gini': gini,
        'training_time_s': train_time,
        'inference_time_ms': inf_time,
        'n_trees': model.best_iteration
    }


def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate CatBoost"""
    print("Training CatBoost...")
    start = time.time()

    model = CatBoostClassifier(
        iterations=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        verbose=False,
        early_stopping_rounds=10,
        thread_count=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )

    train_time = time.time() - start

    # Inference
    start_inf = time.time()
    y_pred = model.predict_proba(X_test)[:, 1]
    inf_time = (time.time() - start_inf) / len(X_test) * 1000

    auc = roc_auc_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred)
    gini = 2 * auc - 1

    return {
        'model': 'CatBoost',
        'auc': auc,
        'logloss': logloss,
        'gini': gini,
        'training_time_s': train_time,
        'inference_time_ms': inf_time,
        'n_trees': model.tree_count_
    }


def main():
    print("\n" + "=" * 70)
    print("GBDT CTR Prediction Model Benchmark")
    print("=" * 70)

    # Generate data
    print("\nGenerating synthetic RTB data (10,000 samples)...")
    X, y = generate_synthetic_rtb_data(n_samples=10000)

    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {X.shape[1]} (user_id, ad_id, context_id, hour, history, quality, affinity, time_factor)")
    print(f"Target CTR: {y.mean()*100:.2f}%")

    # Train models
    print("\n" + "=" * 70)
    print("Training Models")
    print("=" * 70 + "\n")

    results = []
    results.append(train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test))
    results.append(train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test))
    results.append(train_catboost(X_train, y_train, X_val, y_val, X_test, y_test))

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Save to CSV
    results_df.to_csv('results/gbdt_ctr_benchmark.csv', index=False)
    print("\n✓ Results saved to: results/gbdt_ctr_benchmark.csv")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    best_auc = results_df.loc[results_df['auc'].idxmax()]
    best_logloss = results_df.loc[results_df['logloss'].idxmin()]
    fastest_train = results_df.loc[results_df['training_time_s'].idxmin()]
    fastest_inf = results_df.loc[results_df['inference_time_ms'].idxmin()]

    print(f"\n✓ Best AUC: {best_auc['model']} ({best_auc['auc']:.4f})")
    print(f"✓ Best Log Loss: {best_logloss['model']} ({best_logloss['logloss']:.4f})")
    print(f"✓ Fastest Training: {fastest_train['model']} ({fastest_train['training_time_s']:.2f}s)")
    print(f"✓ Fastest Inference: {fastest_inf['model']} ({fastest_inf['inference_time_ms']:.3f}ms/sample)")

    print("\n" + "-" * 70)
    print("Performance Ranking (AUC):")
    print("-" * 70)
    for idx, row in results_df.sort_values('auc', ascending=False).iterrows():
        print(f"  {idx+1}. {row['model']:12} AUC={row['auc']:.4f}  LogLoss={row['logloss']:.4f}  Gini={row['gini']:.4f}")

    print("\n" + "-" * 70)
    print("Latency Comparison:")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"  {row['model']:12} Inference: {row['inference_time_ms']:.4f} ms/sample")

    print("\n" + "-" * 70)
    print("Training Speed (n_trees trained):")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"  {row['model']:12} {row['training_time_s']:.2f}s to train {int(row['n_trees'])} trees")

    return results_df


if __name__ == '__main__':
    results = main()
