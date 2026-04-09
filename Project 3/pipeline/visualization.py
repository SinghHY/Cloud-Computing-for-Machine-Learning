"""
Visualization Module
====================
Generates all plots and charts for the project report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.fft import fft, fftfreq
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Style configuration
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})


def plot_signal_samples(signals, labels, label_names, sampling_rate, save_dir="plots"):
    """Plot sample vibration signals for each fault class."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for class_id in range(4):
        idx = np.where(labels == class_id)[0][0]
        sig = signals[idx]
        t = np.arange(len(sig)) / sampling_rate * 1000  # ms

        axes[class_id].plot(t[:500], sig[:500], linewidth=0.5, color='#2196F3')
        axes[class_id].set_title(f"{label_names[class_id]}", fontweight='bold')
        axes[class_id].set_xlabel("Time (ms)")
        axes[class_id].set_ylabel("Amplitude")
        axes[class_id].grid(True, alpha=0.3)

    plt.suptitle("Vibration Signal Samples by Bearing Condition", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, "signal_samples.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_frequency_spectrum(signals, labels, label_names, sampling_rate, save_dir="plots"):
    """Plot FFT frequency spectra for each fault class."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for class_id in range(4):
        idx = np.where(labels == class_id)[0][0]
        sig = signals[idx]
        n = len(sig)
        yf = fft(sig)
        xf = fftfreq(n, 1.0 / sampling_rate)

        positive = xf > 0
        xf_pos = xf[positive]
        magnitude = 2.0 / n * np.abs(yf[positive])

        axes[class_id].plot(xf_pos[:len(xf_pos)//2], magnitude[:len(magnitude)//2],
                           linewidth=0.5, color='#E91E63')
        axes[class_id].set_title(f"{label_names[class_id]} - FFT Spectrum", fontweight='bold')
        axes[class_id].set_xlabel("Frequency (Hz)")
        axes[class_id].set_ylabel("Magnitude")
        axes[class_id].grid(True, alpha=0.3)

    plt.suptitle("Frequency Domain Analysis (FFT)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, "frequency_spectra.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_feature_distributions(feature_df, top_n=6, save_dir="plots"):
    """Plot distributions of top features across classes."""
    os.makedirs(save_dir, exist_ok=True)
    feature_cols = [c for c in feature_df.columns if c != "label"]

    # Use variance as a simple importance proxy
    variances = feature_df[feature_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    label_map = {0: "Normal", 1: "Inner Race", 2: "Outer Race", 3: "Ball"}

    for i, feat in enumerate(top_features):
        for label_id in sorted(feature_df["label"].unique()):
            data = feature_df[feature_df["label"] == label_id][feat]
            axes[i].hist(data, bins=30, alpha=0.5, label=label_map.get(label_id, str(label_id)))
        axes[i].set_title(feat, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Feature Distributions by Fault Class", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_distributions.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_confusion_matrix(y_true, y_pred, label_names, title="Confusion Matrix", save_dir="plots", filename="confusion_matrix.png"):
    """Plot confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred)
    display_labels = [label_names[i] for i in sorted(label_names.keys())]
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')

    path = os.path.join(save_dir, filename)
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_feature_importance(importance_df, top_n=15, save_dir="plots"):
    """Plot feature importance bar chart."""
    os.makedirs(save_dir, exist_ok=True)

    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top)), top['importance'].values, color='#4CAF50', edgecolor='#388E3C')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Top Feature Importances (AutoML Best Model)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_training_history(history, save_dir="plots"):
    """Plot neural network training curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['val_accuracy'], 'g-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle("HuggingFace Neural Network Training Progress", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_model_comparison(automl_metrics, nn_metrics, save_dir="plots"):
    """Plot comparison between AutoML and HuggingFace NN models."""
    os.makedirs(save_dir, exist_ok=True)

    metrics_names = ['Accuracy', 'F1 Score']
    automl_vals = [automl_metrics.get('accuracy', 0), automl_metrics.get('f1_score', 0)]
    nn_vals = [nn_metrics.get('accuracy', 0), nn_metrics.get('f1_score', 0)]

    x = np.arange(len(metrics_names))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, automl_vals, width, label='AutoML (FLAML)', color='#2196F3', edgecolor='#1565C0')
    bars2 = ax.bar(x + width/2, nn_vals, width, label='HuggingFace NN', color='#FF9800', edgecolor='#E65100')

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: AutoML vs HuggingFace NN', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def plot_pipeline_architecture(save_dir="plots"):
    """Generate a pipeline architecture diagram."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Pipeline stages
    stages = [
        (1.5, 3.5, "Data\nIngestion\n(HuggingFace\nDatasets)", '#E3F2FD', '#1565C0'),
        (4.5, 3.5, "DSP Feature\nExtraction\n(FFT, Hilbert,\nStatistics)", '#E8F5E9', '#2E7D32'),
        (7.5, 3.5, "AutoML\nTraining\n(FLAML)", '#FFF3E0', '#E65100'),
        (10.5, 3.5, "HuggingFace\nNN Model\n(PyTorch)", '#FCE4EC', '#AD1457'),
        (13.5, 3.5, "SageMaker\nDeployment\n(AWS)", '#F3E5F5', '#6A1B9A'),
    ]

    for x, y, text, bg_color, border_color in stages:
        rect = plt.Rectangle((x - 1.2, y - 1.2), 2.4, 2.4,
                             facecolor=bg_color, edgecolor=border_color,
                             linewidth=2, zorder=2, joinstyle='round')
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
               fontweight='bold', color=border_color, zorder=3)

    # Arrows
    for i in range(len(stages) - 1):
        ax.annotate('', xy=(stages[i+1][0] - 1.3, stages[i+1][1]),
                    xytext=(stages[i][0] + 1.3, stages[i][1]),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))

    # MLOps layer
    mlops_rect = plt.Rectangle((0.3, 0.3), 15.4, 1.2,
                               facecolor='#ECEFF1', edgecolor='#37474F',
                               linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(mlops_rect)
    ax.text(8, 0.9, "MLOps Layer:  MLflow Tracking  |  CI/CD (GitHub Actions)  |  Model Monitoring  |  Docker Containerization",
           ha='center', va='center', fontsize=9, color='#37474F', fontweight='bold')

    ax.set_title("ML Pipeline Architecture: Bearing Fault Detection System",
                fontsize=15, fontweight='bold', pad=20)

    path = os.path.join(save_dir, "pipeline_architecture.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def generate_all_plots(results, save_dir="plots"):
    """
    Generate all plots for the report.

    Parameters
    ----------
    results : dict
        Pipeline results containing signals, features, metrics, etc.
    save_dir : str
        Directory to save plots

    Returns
    -------
    plot_paths : dict
        Dictionary mapping plot names to file paths
    """
    logger.info("Generating all visualization plots...")
    plot_paths = {}

    plot_paths['architecture'] = plot_pipeline_architecture(save_dir)

    if 'signals' in results and 'labels' in results:
        plot_paths['signals'] = plot_signal_samples(
            results['signals'], results['labels'],
            results['label_names'], results['sampling_rate'], save_dir
        )
        plot_paths['spectra'] = plot_frequency_spectrum(
            results['signals'], results['labels'],
            results['label_names'], results['sampling_rate'], save_dir
        )

    if 'feature_df' in results:
        plot_paths['features'] = plot_feature_distributions(
            results['feature_df'], save_dir=save_dir
        )

    if 'automl_cm' in results:
        plot_paths['automl_cm'] = plot_confusion_matrix(
            results['automl_cm']['true'], results['automl_cm']['pred'],
            results['label_names'], "AutoML Confusion Matrix",
            save_dir, "automl_confusion_matrix.png"
        )

    if 'nn_cm' in results:
        plot_paths['nn_cm'] = plot_confusion_matrix(
            results['nn_cm']['true'], results['nn_cm']['pred'],
            results['label_names'], "HuggingFace NN Confusion Matrix",
            save_dir, "nn_confusion_matrix.png"
        )

    if 'importance_df' in results and results['importance_df'] is not None:
        plot_paths['importance'] = plot_feature_importance(
            results['importance_df'], save_dir=save_dir
        )

    if 'training_history' in results:
        plot_paths['history'] = plot_training_history(
            results['training_history'], save_dir=save_dir
        )

    if 'automl_metrics' in results and 'nn_metrics' in results:
        plot_paths['comparison'] = plot_model_comparison(
            results['automl_metrics'], results['nn_metrics'], save_dir
        )

    logger.info(f"Generated {len(plot_paths)} plots")
    return plot_paths
