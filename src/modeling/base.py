# src/modeling/base.py
"""
Base classes and interfaces for machine learning models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix

from ..utils import logger
from ..config import settings


class AMLModel(ABC, BaseEstimator, ClassifierMixin):
    """Abstract base class for AML models"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logger.getChild(f"modeling.{name}")
        self.is_fitted = False
        self.feature_names = []
        self.model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'AMLModel':
        """
        Fit the model

        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional fit parameters

        Returns:
            Fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Feature matrix

        Returns:
            Class probabilities
        """
        pass

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Score the model (default: accuracy for classification)

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y.values)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores

        Returns:
            Dictionary of feature importance or None
        """
        return None

    def validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data

        Args:
            X: Feature matrix
            y: Target vector (optional)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        if X.empty:
            raise ValueError("X cannot be empty")

        if y is not None:
            if not isinstance(y, pd.Series):
                raise ValueError("y must be a pandas Series")

            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

            # Check for class imbalance
            class_counts = y.value_counts()
            minority_class_ratio = class_counts.min() / class_counts.max()

            if minority_class_ratio < 0.01:  # Less than 1%
                self.logger.warning(".4f",
                                  minority_ratio=minority_class_ratio)

    def log_model_info(self) -> None:
        """Log model information"""
        self.logger.info(f"Model: {self.name}",
                        fitted=self.is_fitted,
                        features=len(self.feature_names),
                        config=self.config)

    def save_model(self, path: str) -> None:
        """
        Save model to disk

        Args:
            path: Path to save model
        """
        import joblib

        model_data = {
            'model': self.model,
            'name': self.name,
            'config': self.config,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str) -> 'AMLModel':
        """
        Load model from disk

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        import joblib

        model_data = joblib.load(path)

        # Create instance
        instance = cls(model_data['name'], model_data['config'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']

        instance.logger.info(f"Model loaded from {path}")
        return instance


class EnsembleModel(AMLModel):
    """Base class for ensemble models"""

    def __init__(self, name: str, models: List[AMLModel], config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.models = models
        self.weights = config.get('weights', None)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnsembleModel':
        """Fit ensemble models"""
        self.logger.info(f"Fitting ensemble: {self.name}")

        for i, model in enumerate(self.models):
            self.logger.info(f"Fitting model {i+1}/{len(self.models)}: {model.name}")
            model.fit(X, y, **kwargs)

        self.is_fitted = True
        self.feature_names = self.models[0].feature_names if self.models else []

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Combine predictions
        if self.weights is not None:
            # Weighted voting
            pred_matrix = np.column_stack(predictions)
            weighted_pred = np.average(pred_matrix, axis=1, weights=self.weights)
            return (weighted_pred >= 0.5).astype(int)
        else:
            # Majority voting
            pred_matrix = np.column_stack(predictions)
            return (np.mean(pred_matrix, axis=1) >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble probability prediction"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        probas = []

        for model in self.models:
            proba = model.predict_proba(X)
            probas.append(proba)

        # Combine probabilities
        if self.weights is not None:
            # Weighted average
            proba_matrix = np.stack(probas, axis=0)  # Shape: (n_models, n_samples, n_classes)
            weighted_proba = np.average(proba_matrix, axis=0, weights=self.weights)
            return weighted_proba
        else:
            # Simple average
            return np.mean(probas, axis=0)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get ensemble feature importance"""
        if not self.models:
            return None

        # Average importance across models
        importance_dicts = []
        for model in self.models:
            imp = model.get_feature_importance()
            if imp is not None:
                importance_dicts.append(imp)

        if not importance_dicts:
            return None

        # Combine importances
        all_features = set()
        for imp_dict in importance_dicts:
            all_features.update(imp_dict.keys())

        combined_importance = {}
        for feature in all_features:
            values = [imp_dict.get(feature, 0) for imp_dict in importance_dicts]
            combined_importance[feature] = np.mean(values)

        return combined_importance


class ModelEvaluator:
    """Model evaluation utilities"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger.getChild("model_evaluator")

    def evaluate_model(self, model: AMLModel, X: pd.DataFrame, y: pd.Series,
                      X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation

        Args:
            model: Fitted model
            X: Test features
            y: Test target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Evaluation results
        """
        self.logger.info("Evaluating model performance")

        results = {
            'model_name': model.name,
            'test_metrics': {},
            'validation_metrics': {},
            'feature_importance': model.get_feature_importance(),
            'predictions': {}
        }

        # Test set evaluation
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        results['test_metrics'] = self._calculate_metrics(y, y_pred, y_proba)
        results['predictions']['test'] = {
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist()
        }

        # Validation set evaluation
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)

            results['validation_metrics'] = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
            results['predictions']['validation'] = {
                'y_true': y_val.tolist(),
                'y_pred': y_val_pred.tolist(),
                'y_proba': y_val_proba.tolist()
            }

        self.logger.info("Model evaluation completed",
                        test_accuracy=results['test_metrics'].get('accuracy', 'N/A'))

        return results

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                   roc_auc_score, average_precision_score, classification_report)

        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        # AUC metrics (for binary classification)
        if len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC metrics: {e}")

        # Detailed classification report
        metrics['classification_report'] = classification_report(y_true, y_pred,
                                                               output_dict=True, zero_division=0)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        return metrics

    def compare_models(self, models: List[AMLModel], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            models: List of fitted models
            X: Test features
            y: Test target

        Returns:
            Comparison DataFrame
        """
        self.logger.info(f"Comparing {len(models)} models")

        results = []

        for model in models:
            try:
                metrics = self.evaluate_model(model, X, y)['test_metrics']
                metrics['model_name'] = model.name
                results.append(metrics)
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model.name}: {e}")
                results.append({'model_name': model.name, 'error': str(e)})

        return pd.DataFrame(results)


__all__ = [
    "AMLModel",
    "EnsembleModel",
    "ModelEvaluator"
]