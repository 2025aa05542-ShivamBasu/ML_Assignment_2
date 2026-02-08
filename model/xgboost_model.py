"""
XGBoost model for Obesity dataset

Gradient Boosting Ensemble - provides training, evaluation, save/load and predict API.
"""

import os
from typing import Dict, Any

import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

from model.obesity_dataset_loader import ObesityDatasetLoader


class XGBoostModel:
    """XGBoost Classifier (Gradient Boosting Ensemble) for Obesity dataset"""
    
    def __init__(self):
        self.model: XGBClassifier = None

    def train(self, X, y, **kwargs) -> XGBClassifier:
        """
        Train an XGBoost model.
        
        Args:
            X: Features
            y: Target
            **kwargs: Additional parameters for XGBClassifier
                     (n_estimators, learning_rate, max_depth, subsample, etc.)
        """
        self.model = XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 6),
            subsample=kwargs.get("subsample", 1.0),
            colsample_bytree=kwargs.get("colsample_bytree", 1.0),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,  # Use all available processors
            verbosity=0,
        )
        self.model.fit(X, y)
        return self.model

    def evaluate(self, X, y) -> Dict[str, Any]:
        """
        Evaluate the trained model and return metrics.

        Returns a dictionary with:
          - accuracy
          - auc (macro, supports multiclass using one-vs-rest)
          - precision (macro)
          - recall (macro)
          - f1_score (macro)
          - matthews_corrcoef
          - report (classification_report output_dict)
          - confusion_matrix
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        preds = self.model.predict(X)
        acc = accuracy_score(y, preds)
        report = classification_report(y, preds, output_dict=True)
        cm = confusion_matrix(y, preds)

        # Precision / Recall / F1 (macro to handle multiclass fairly)
        precision = precision_score(y, preds, average="macro", zero_division=0)
        recall = recall_score(y, preds, average="macro", zero_division=0)
        f1 = f1_score(y, preds, average="macro", zero_division=0)

        # Matthews correlation coefficient (supports multiclass)
        try:
            mcc = matthews_corrcoef(y, preds)
        except Exception:
            mcc = None

        # AUC: for multiclass use one-vs-rest with label binarization
        auc = None
        try:
            if len(np.unique(y)) == 2:
                # binary
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)[:, 1]
                    auc = roc_auc_score(y, probs)
            else:
                # multiclass
                if hasattr(self.model, "predict_proba"):
                    classes = np.unique(y)
                    y_bin = label_binarize(y, classes=classes)
                    probs = self.model.predict_proba(X)
                    auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
        except Exception:
            auc = None

        return {
            "accuracy": acc,
            "auc": auc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "mcc": mcc,
            "report": report,
            "confusion_matrix": cm,
        }

    def predict(self, X) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X)

    def save(self, filepath: str, extras: Dict[str, Any] = None) -> None:
        """Save model and optional extras (encoders, columns) using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {"model": self.model}
        if extras:
            payload.update(extras)
        joblib.dump(payload, filepath)

    def load(self, filepath: str) -> Dict[str, Any]:
        """Load model payload saved by `save()` and attach the model."""
        payload = joblib.load(filepath)
        if "model" in payload:
            self.model = payload["model"]
        return payload

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the trained XGBoost model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.feature_importances_


def main(save_path: str = "model/xgboost_model.joblib"):
    """Train and evaluate XGBoost on Obesity dataset"""
    loader = ObesityDatasetLoader()
    df = loader.load_data()
    if df is None:
        print("Failed to load dataset. Exiting.")
        return

    X, y = loader.preprocess_data()
    X_train, X_test, y_train, y_test = loader.train_test_split_data(test_size=0.2)

    xgb = XGBoostModel()
    xgb.train(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6)

    metrics = xgb.evaluate(X_test, y_test)
    print(f"✓ XGBoost Training Complete (100 rounds)")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC (macro): {metrics['auc']:.4f}" if metrics['auc'] is not None else "AUC: N/A")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    
    print("\nClassification report:")
    for cls, vals in metrics["report"].items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        print(f"  {cls}: precision={vals['precision']:.3f}, recall={vals['recall']:.3f}, f1={vals['f1-score']:.3f}")

    # Save model along with label encoders and feature columns from loader
    extras = {
        "label_encoders": loader.label_encoders,
        "feature_columns": loader.get_feature_columns(),
        "feature_importance": xgb.get_feature_importance(),
    }
    xgb.save(save_path, extras=extras)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
