"""
Logistic Regression model for Obesity dataset

Provides a simple training, evaluation, save/load and predict API.
"""

import os
from typing import Tuple, Dict, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
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


class LogisticRegressionModel:
    def __init__(self):
        self.model: LogisticRegression = None

    def train(self, X, y, **kwargs) -> LogisticRegression:
        """Train a logistic regression model."""
        self.model = LogisticRegression(
            multi_class=kwargs.get("multi_class", "multinomial"),
            solver=kwargs.get("solver", "lbfgs"),
            max_iter=kwargs.get("max_iter", 1000),
            C=kwargs.get("C", 1.0),
        )
        self.model.fit(X, y)
        return self.model

    def evaluate(self, X, y) -> Dict[str, Any]:
        """Evaluate the trained model and return metrics.

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


def main(save_path: str = "model/logistic_regression_model.joblib"):
    loader = ObesityDatasetLoader()
    df = loader.load_data()
    if df is None:
        print("Failed to load dataset. Exiting.")
        return

    X, y = loader.preprocess_data()
    X_train, X_test, y_train, y_test = loader.train_test_split_data(test_size=0.2)

    lr = LogisticRegressionModel()
    lr.train(X_train, y_train)

    metrics = lr.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Classification report:")
    for cls, vals in metrics["report"].items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        print(f"  {cls}: precision={vals['precision']:.3f}, recall={vals['recall']:.3f}, f1={vals['f1-score']:.3f}")

    # Save model along with label encoders and feature columns from loader
    extras = {
        "label_encoders": loader.label_encoders,
        "feature_columns": loader.get_feature_columns(),
    }
    lr.save(save_path, extras=extras)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
