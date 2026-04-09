"""
AutoML Training Module
======================
Uses FLAML (Fast Lightweight AutoML) for automated model selection and
hyperparameter optimization. This module demonstrates how AutoML tools
streamline the model selection process in an MLOps pipeline.

FLAML is chosen for its efficiency and ability to find good models quickly
with limited computational budget.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from flaml import AutoML
import mlflow
import mlflow.sklearn
import joblib
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMLTrainer:
    """
    AutoML-based model training using FLAML with MLflow experiment tracking.

    This class wraps FLAML's AutoML functionality and adds MLOps best practices:
    - Experiment tracking with MLflow
    - Model versioning and artifact storage
    - Comprehensive evaluation metrics
    - Model comparison and selection
    """

    def __init__(self, experiment_name="bearing_fault_detection", models_dir="models"):
        """
        Parameters
        ----------
        experiment_name : str
            MLflow experiment name
        models_dir : str
            Directory to save trained models
        """
        self.experiment_name = experiment_name
        self.models_dir = models_dir
        self.scaler = StandardScaler()
        self.automl = None
        self.best_model = None
        self.metrics = {}

        os.makedirs(models_dir, exist_ok=True)

        # Set up MLflow tracking
        mlflow.set_tracking_uri(f"file://{os.path.abspath(models_dir)}/mlruns")
        mlflow.set_experiment(experiment_name)

    def prepare_data(self, feature_dfs):
        """
        Prepare feature DataFrames for training.

        Applies StandardScaler normalization to ensure features are on
        the same scale, which is important for many ML algorithms.

        Parameters
        ----------
        feature_dfs : dict
            Dictionary with 'train', 'validation', 'test' DataFrames

        Returns
        -------
        data : dict
            Prepared data splits with X and y arrays
        """
        train_df = feature_dfs["train"]
        val_df = feature_dfs["validation"]
        test_df = feature_dfs["test"]

        feature_cols = [c for c in train_df.columns if c != "label"]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        data = {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "feature_names": feature_cols
        }

        logger.info(f"Prepared data shapes:")
        logger.info(f"  Train: {X_train_scaled.shape}")
        logger.info(f"  Validation: {X_val_scaled.shape}")
        logger.info(f"  Test: {X_test_scaled.shape}")

        return data

    def train_automl(self, data, time_budget=120, metric="accuracy"):
        """
        Train models using FLAML AutoML.

        FLAML automatically searches over multiple ML algorithms including:
        - LightGBM, XGBoost, Random Forest, Extra Trees
        - Logistic Regression, k-NN
        - CatBoost (if installed)

        It uses a cost-effective hyperparameter optimization strategy
        that adaptively allocates budget to promising configurations.

        Parameters
        ----------
        data : dict
            Prepared data from prepare_data()
        time_budget : int
            Total time budget in seconds for AutoML search
        metric : str
            Optimization metric

        Returns
        -------
        automl : AutoML
            Trained FLAML AutoML instance
        """
        logger.info("=" * 60)
        logger.info("STEP 3: AutoML MODEL TRAINING (FLAML)")
        logger.info("=" * 60)
        logger.info(f"Time budget: {time_budget}s | Metric: {metric}")

        self.automl = AutoML()

        # Start MLflow run for experiment tracking
        with mlflow.start_run(run_name=f"automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("time_budget", time_budget)
            mlflow.log_param("metric", metric)
            mlflow.log_param("n_train_samples", len(data["y_train"]))
            mlflow.log_param("n_features", data["X_train"].shape[1])
            mlflow.log_param("n_classes", len(np.unique(data["y_train"])))

            # Run AutoML
            self.automl.fit(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                task="classification",
                time_budget=time_budget,
                metric=metric,
                estimator_list=[
                    "lgbm", "xgboost", "rf", "extra_tree", "lrl1"
                ],
                log_training_metric=True,
                verbose=1,
                seed=42
            )

            self.best_model = self.automl.model

            # Evaluate on all splits
            splits = {
                "train": (data["X_train"], data["y_train"]),
                "val": (data["X_val"], data["y_val"]),
                "test": (data["X_test"], data["y_test"])
            }

            for split_name, (X, y) in splits.items():
                y_pred = self.automl.predict(X)
                acc = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred, average="weighted")
                rec = recall_score(y, y_pred, average="weighted")
                f1 = f1_score(y, y_pred, average="weighted")
                cm = confusion_matrix(y, y_pred)

                self.metrics[split_name] = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "confusion_matrix": cm.tolist()
                }

                # Log to MLflow
                mlflow.log_metric(f"{split_name}_accuracy", acc)
                mlflow.log_metric(f"{split_name}_precision", prec)
                mlflow.log_metric(f"{split_name}_recall", rec)
                mlflow.log_metric(f"{split_name}_f1", f1)

                logger.info(f"\n{split_name.upper()} Results:")
                logger.info(f"  Accuracy:  {acc:.4f}")
                logger.info(f"  Precision: {prec:.4f}")
                logger.info(f"  Recall:    {rec:.4f}")
                logger.info(f"  F1 Score:  {f1:.4f}")

            # Log best model info
            logger.info(f"\nBest model: {self.automl.best_estimator}")
            logger.info(f"Best config: {self.automl.best_config}")
            mlflow.log_param("best_estimator", self.automl.best_estimator)
            mlflow.log_param("best_config", json.dumps(self.automl.best_config))

            # Log the model
            mlflow.sklearn.log_model(self.best_model, "best_model")

            # Print classification report for test set
            y_test_pred = self.automl.predict(data["X_test"])
            logger.info(f"\nTest Set Classification Report:")
            logger.info(classification_report(data["y_test"], y_test_pred))

        return self.automl

    def save_model(self, filename="best_model.pkl"):
        """Save the best model and scaler for deployment."""
        model_path = os.path.join(self.models_dir, filename)
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")

        return model_path, scaler_path

    def get_feature_importance(self, feature_names):
        """
        Get feature importance from the best model (if available).

        Parameters
        ----------
        feature_names : list
            List of feature names

        Returns
        -------
        importance_df : pd.DataFrame
            Feature importance sorted by importance
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_).mean(axis=0)
        else:
            logger.warning("Model does not support feature importance")
            return None

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df


def run_automl_training(feature_dfs, time_budget=120, models_dir="models"):
    """
    Run the complete AutoML training pipeline.

    Parameters
    ----------
    feature_dfs : dict
        Feature DataFrames from feature engineering
    time_budget : int
        AutoML time budget in seconds
    models_dir : str
        Directory for saving models

    Returns
    -------
    trainer : AutoMLTrainer
        Trained AutoML trainer with results
    data : dict
        Prepared data splits
    """
    trainer = AutoMLTrainer(models_dir=models_dir)
    data = trainer.prepare_data(feature_dfs)
    trainer.train_automl(data, time_budget=time_budget)
    trainer.save_model()

    return trainer, data


if __name__ == "__main__":
    from data_ingestion import load_data_pipeline
    from feature_engineering import run_feature_engineering

    dataset_dict, label_names, metadata = load_data_pipeline(n_samples=200)
    feature_dfs, extractor = run_feature_engineering(dataset_dict, metadata["sampling_rate"])
    trainer, data = run_automl_training(feature_dfs, time_budget=60)

    print(f"\nBest model: {trainer.automl.best_estimator}")
    print(f"\nTest metrics: {trainer.metrics['test']}")
