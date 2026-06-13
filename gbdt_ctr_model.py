"""
Gradient Boosting Decision Tree Models for CTR Prediction
Implements XGBoost, LightGBM, and CatBoost for comparison with DNN
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import time


class GBDTCTRPredictor:
    """XGBoost CTR predictor - gradient boosting baseline"""

    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'eval_metric': 'logloss'
            }
        self.params = params
        self.model = None
        self.feature_names = None
        self.training_time = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 100):
        """Train XGBoost model"""
        start_time = time.time()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals = [(dtrain, 'train'), (dval, 'val')]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=epochs,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CTR probabilities"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def get_feature_importance(self, top_k: int = 10) -> pd.DataFrame:
        """Get feature importance scores"""
        importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        return importance_df.head(top_k)


class LGBMCTRPredictor:
    """LightGBM CTR predictor - faster training variant"""

    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'verbose': -1
            }
        self.params = params
        self.model = None
        self.training_time = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 100):
        """Train LightGBM model"""
        start_time = time.time()

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=epochs,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(10)]
        )

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CTR probabilities"""
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self, top_k: int = 10) -> pd.DataFrame:
        """Get feature importance scores"""
        importance = self.model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': [f'f{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df.head(top_k)


class CatBoostCTRPredictor:
    """CatBoost CTR predictor - handles categorical features natively"""

    def __init__(self, params: Dict[str, Any] = None):
        if params is None:
            params = {
                'loss_function': 'Logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'iterations': 100,
                'subsample': 0.8,
                'verbose': False,
                'thread_count': -1
            }
        self.params = params
        self.model = None
        self.training_time = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 100):
        """Train CatBoost model"""
        start_time = time.time()

        self.model = CatBoostClassifier(
            **self.params,
            iterations=epochs,
            early_stopping_rounds=10,
            use_best_model=True
        )

        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CTR probabilities"""
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, top_k: int = 10) -> pd.DataFrame:
        """Get feature importance scores"""
        importance = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': [f'f{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df.head(top_k)


class CTRModelComparator:
    """Compare GBDT models with DNN baseline"""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate CTR prediction metrics"""
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)

        # Calculate Gini coefficient: 2*AUC - 1
        gini = 2 * auc - 1

        # Calibration: average predicted vs actual
        calibration_error = np.mean(np.abs(y_pred - y_true))

        return {
            'auc': auc,
            'logloss': logloss,
            'gini': gini,
            'calibration_error': calibration_error
        }

    @staticmethod
    def compare_models(models: Dict[str, Any],
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> pd.DataFrame:
        """Compare multiple models"""
        results = []

        for model_name, model_obj in models.items():
            # Measure inference time
            start_time = time.time()
            y_pred = model_obj.predict(X_test)
            inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

            metrics = CTRModelComparator.evaluate(y_test, y_pred)
            metrics['model'] = model_name
            metrics['inference_time_ms'] = inference_time

            if hasattr(model_obj, 'training_time'):
                metrics['training_time_s'] = model_obj.training_time

            results.append(metrics)

        return pd.DataFrame(results)
