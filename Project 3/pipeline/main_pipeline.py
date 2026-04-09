"""
Main Pipeline Orchestrator
===========================
AIG130 - Project 3: Cloud Computing for Machine Learning

End-to-end ML pipeline for Bearing Fault Detection from vibration signals.
Integrates: HuggingFace Datasets, DSP Feature Engineering, FLAML AutoML,
HuggingFace Neural Network, MLflow Tracking, and AWS SageMaker Deployment.
"""

import sys
import os
import json
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import load_data_pipeline, generate_synthetic_vibration_data
from feature_engineering import run_feature_engineering
from automl_training import run_automl_training
from huggingface_model import run_huggingface_training
from sagemaker_deploy import generate_deployment_files
from visualization import generate_all_plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_full_pipeline(
    n_samples=500,
    signal_length=2048,
    sampling_rate=12000,
    automl_time_budget=120,
    nn_epochs=50,
    output_dir="output"
):
    """
    Execute the complete bearing fault detection pipeline.

    Pipeline Steps:
    1. Data Ingestion (HuggingFace Datasets)
    2. DSP Feature Extraction (FFT, Hilbert, Statistics)
    3. AutoML Model Training (FLAML)
    4. HuggingFace Neural Network Training
    5. Model Comparison and Selection
    6. Deployment Configuration (AWS SageMaker)
    7. Visualization and Reporting

    Parameters
    ----------
    n_samples : int
        Number of samples per fault class
    signal_length : int
        Signal length in data points
    sampling_rate : int
        Sampling rate in Hz
    automl_time_budget : int
        FLAML AutoML search budget in seconds
    nn_epochs : int
        Neural network training epochs
    output_dir : str
        Output directory for models, plots, etc.
    """
    start_time = datetime.now()

    models_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots")
    deploy_dir = os.path.join(output_dir, "deployment")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BEARING FAULT DETECTION - COMPLETE ML PIPELINE")
    logger.info(f"Started at: {start_time}")
    logger.info("=" * 70)

    # ============================================================
    # STEP 1: Data Ingestion (HuggingFace)
    # ============================================================
    dataset_dict, label_names, metadata = load_data_pipeline(
        n_samples=n_samples,
        signal_length=signal_length,
        sampling_rate=sampling_rate
    )

    # Also keep raw signals for visualization
    signals_raw, labels_raw, _ = generate_synthetic_vibration_data(
        n_samples=n_samples,
        signal_length=signal_length,
        sampling_rate=sampling_rate
    )

    # ============================================================
    # STEP 2: DSP Feature Extraction
    # ============================================================
    feature_dfs, extractor = run_feature_engineering(
        dataset_dict, sampling_rate=sampling_rate
    )

    # ============================================================
    # STEP 3: AutoML Training (FLAML)
    # ============================================================
    trainer, data = run_automl_training(
        feature_dfs,
        time_budget=automl_time_budget,
        models_dir=models_dir
    )

    # Get feature importance
    importance_df = trainer.get_feature_importance(data["feature_names"])

    # Get AutoML predictions for visualization
    automl_test_pred = trainer.automl.predict(data["X_test"])

    # ============================================================
    # STEP 4: HuggingFace Neural Network
    # ============================================================
    hf_trainer = run_huggingface_training(
        data, label_names, models_dir=models_dir, n_epochs=nn_epochs
    )

    # ============================================================
    # STEP 5: Model Comparison
    # ============================================================
    logger.info("=" * 60)
    logger.info("STEP 5: MODEL COMPARISON")
    logger.info("=" * 60)

    automl_metrics = trainer.metrics["test"]
    nn_metrics = hf_trainer.test_metrics

    logger.info(f"AutoML (FLAML {trainer.automl.best_estimator}):")
    logger.info(f"  Test Accuracy: {automl_metrics['accuracy']:.4f}")
    logger.info(f"  Test F1: {automl_metrics['f1_score']:.4f}")
    logger.info(f"\nHuggingFace Neural Network:")
    logger.info(f"  Test Accuracy: {nn_metrics['accuracy']:.4f}")
    logger.info(f"  Test F1: {nn_metrics['f1_score']:.4f}")

    best_model_type = "AutoML" if automl_metrics["accuracy"] >= nn_metrics["accuracy"] else "HuggingFace NN"
    logger.info(f"\nBest model: {best_model_type}")

    # ============================================================
    # STEP 6: Deployment Configuration
    # ============================================================
    generate_deployment_files(output_dir=deploy_dir)

    # ============================================================
    # STEP 7: Visualization
    # ============================================================
    logger.info("=" * 60)
    logger.info("STEP 7: GENERATING VISUALIZATIONS")
    logger.info("=" * 60)

    viz_results = {
        'signals': signals_raw,
        'labels': labels_raw,
        'label_names': label_names,
        'sampling_rate': sampling_rate,
        'feature_df': feature_dfs['train'],
        'automl_cm': {'true': data['y_test'], 'pred': automl_test_pred},
        'nn_cm': {'true': hf_trainer.test_metrics['true_labels'],
                  'pred': hf_trainer.test_metrics['predictions']},
        'importance_df': importance_df,
        'training_history': hf_trainer.training_history,
        'automl_metrics': automl_metrics,
        'nn_metrics': nn_metrics
    }

    plot_paths = generate_all_plots(viz_results, save_dir=plots_dir)

    # ============================================================
    # Save Pipeline Summary
    # ============================================================
    elapsed = (datetime.now() - start_time).total_seconds()

    summary = {
        "pipeline_run": {
            "timestamp": start_time.isoformat(),
            "elapsed_seconds": elapsed,
            "n_samples_per_class": n_samples,
            "total_samples": metadata["total_samples"],
            "signal_length": signal_length,
            "sampling_rate": sampling_rate,
            "n_features": len(extractor.feature_names)
        },
        "automl_results": {
            "best_estimator": trainer.automl.best_estimator,
            "best_config": trainer.automl.best_config,
            "test_accuracy": automl_metrics["accuracy"],
            "test_f1": automl_metrics["f1_score"],
            "test_precision": automl_metrics["precision"],
            "test_recall": automl_metrics["recall"]
        },
        "nn_results": {
            "test_accuracy": nn_metrics["accuracy"],
            "test_f1": nn_metrics["f1_score"]
        },
        "best_model": best_model_type,
        "feature_names": extractor.feature_names,
        "plot_paths": plot_paths
    }

    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"PIPELINE COMPLETE - Elapsed: {elapsed:.1f}s")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info(f"{'=' * 70}")

    return summary, plot_paths


if __name__ == "__main__":
    summary, plot_paths = run_full_pipeline(
        n_samples=500,
        automl_time_budget=120,
        nn_epochs=50,
        output_dir="/sessions/gallant-brave-maxwell/project3/output"
    )
    print(json.dumps(summary, indent=2, default=str))
