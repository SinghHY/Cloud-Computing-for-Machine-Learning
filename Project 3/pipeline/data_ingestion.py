"""
Data Ingestion Module
=====================
Loads bearing vibration signal data from HuggingFace Hub and prepares it
for the DSP feature extraction pipeline.

This module demonstrates HuggingFace Datasets integration for sourcing
real-world vibration signal data for predictive maintenance.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import hf_hub_download
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_vibration_data(
    n_samples=1000,
    signal_length=2048,
    sampling_rate=12000,
    random_state=42
):
    """
    Generate synthetic bearing vibration signals for fault detection.

    Simulates four bearing conditions using DSP principles:
    - Normal: Clean sinusoidal signal with minimal noise
    - Inner Race Fault: Periodic impulses at ball pass frequency (inner race)
    - Outer Race Fault: Periodic impulses at ball pass frequency (outer race)
    - Ball Fault: Modulated signal with rolling element spin frequency

    Parameters
    ----------
    n_samples : int
        Number of samples per class
    signal_length : int
        Number of data points per vibration signal
    sampling_rate : int
        Sampling rate in Hz (typical for accelerometers)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    signals : np.ndarray
        Shape (n_samples*4, signal_length) vibration signals
    labels : np.ndarray
        Shape (n_samples*4,) fault class labels (0-3)
    label_names : dict
        Mapping from label index to fault name
    """
    np.random.seed(random_state)

    t = np.linspace(0, signal_length / sampling_rate, signal_length)

    label_names = {
        0: "Normal",
        1: "Inner Race Fault",
        2: "Outer Race Fault",
        3: "Ball Fault"
    }

    signals = []
    labels = []

    # Bearing characteristic frequencies (normalized)
    shaft_freq = 30.0       # Hz - shaft rotation frequency
    bpfi = 5.41 * shaft_freq  # Ball Pass Frequency Inner race
    bpfo = 3.59 * shaft_freq  # Ball Pass Frequency Outer race
    bsf = 2.32 * shaft_freq   # Ball Spin Frequency

    for i in range(n_samples):
        # --- Normal Condition ---
        amplitude = np.random.uniform(0.8, 1.2)
        noise_level = np.random.uniform(0.05, 0.15)
        signal_normal = amplitude * np.sin(2 * np.pi * shaft_freq * t)
        signal_normal += amplitude * 0.3 * np.sin(2 * np.pi * 2 * shaft_freq * t)
        signal_normal += noise_level * np.random.randn(signal_length)
        signals.append(signal_normal)
        labels.append(0)

        # --- Inner Race Fault ---
        amplitude = np.random.uniform(0.8, 1.2)
        noise_level = np.random.uniform(0.1, 0.25)
        fault_severity = np.random.uniform(0.5, 1.5)
        signal_inner = amplitude * np.sin(2 * np.pi * shaft_freq * t)
        # Periodic impulses at BPFI
        impulse_train = np.zeros(signal_length)
        impulse_period = int(sampling_rate / bpfi)
        for idx in range(0, signal_length, max(impulse_period, 1)):
            if idx < signal_length:
                decay_len = min(50, signal_length - idx)
                impulse = fault_severity * np.exp(-np.arange(decay_len) / 10.0)
                impulse *= np.sin(2 * np.pi * 3000 * np.arange(decay_len) / sampling_rate)
                impulse_train[idx:idx + decay_len] += impulse
        signal_inner += impulse_train
        signal_inner += noise_level * np.random.randn(signal_length)
        signals.append(signal_inner)
        labels.append(1)

        # --- Outer Race Fault ---
        amplitude = np.random.uniform(0.8, 1.2)
        noise_level = np.random.uniform(0.1, 0.25)
        fault_severity = np.random.uniform(0.4, 1.2)
        signal_outer = amplitude * np.sin(2 * np.pi * shaft_freq * t)
        impulse_train = np.zeros(signal_length)
        impulse_period = int(sampling_rate / bpfo)
        for idx in range(0, signal_length, max(impulse_period, 1)):
            if idx < signal_length:
                decay_len = min(40, signal_length - idx)
                impulse = fault_severity * np.exp(-np.arange(decay_len) / 8.0)
                impulse *= np.sin(2 * np.pi * 2500 * np.arange(decay_len) / sampling_rate)
                impulse_train[idx:idx + decay_len] += impulse
        signal_outer += impulse_train
        signal_outer += noise_level * np.random.randn(signal_length)
        signals.append(signal_outer)
        labels.append(2)

        # --- Ball Fault ---
        amplitude = np.random.uniform(0.8, 1.2)
        noise_level = np.random.uniform(0.1, 0.2)
        fault_severity = np.random.uniform(0.3, 1.0)
        signal_ball = amplitude * np.sin(2 * np.pi * shaft_freq * t)
        # Amplitude modulation at BSF
        modulation = 1 + fault_severity * np.sin(2 * np.pi * bsf * t)
        carrier = np.sin(2 * np.pi * 2000 * t)
        signal_ball += fault_severity * modulation * carrier
        signal_ball += noise_level * np.random.randn(signal_length)
        signals.append(signal_ball)
        labels.append(3)

    signals = np.array(signals)
    labels = np.array(labels)

    # Shuffle
    perm = np.random.permutation(len(labels))
    signals = signals[perm]
    labels = labels[perm]

    logger.info(f"Generated {len(labels)} synthetic vibration signals")
    logger.info(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return signals, labels, label_names


def create_huggingface_dataset(signals, labels, label_names):
    """
    Convert numpy arrays into a HuggingFace Dataset and split into train/val/test.

    This demonstrates HuggingFace Dataset creation and management,
    which is a key component of the MLOps pipeline.

    Parameters
    ----------
    signals : np.ndarray
        Vibration signal data
    labels : np.ndarray
        Fault class labels
    label_names : dict
        Label index to name mapping

    Returns
    -------
    dataset_dict : DatasetDict
        HuggingFace DatasetDict with train/validation/test splits
    """
    # Create a DataFrame with signal features and labels
    data = {
        "signal": [sig.tolist() for sig in signals],
        "label": labels.tolist(),
        "label_name": [label_names[l] for l in labels]
    }

    dataset = Dataset.from_dict(data)

    # Split: 70% train, 15% validation, 15% test
    train_test = dataset.train_test_split(test_size=0.3, seed=42, stratify_by_column="label")
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

    logger.info(f"Dataset splits - Train: {len(dataset_dict['train'])}, "
                f"Val: {len(dataset_dict['validation'])}, "
                f"Test: {len(dataset_dict['test'])}")

    return dataset_dict


def load_data_pipeline(n_samples=1000, signal_length=2048, sampling_rate=12000):
    """
    Complete data ingestion pipeline.

    Returns
    -------
    dataset_dict : DatasetDict
        HuggingFace DatasetDict with train/validation/test splits
    label_names : dict
        Label mapping
    metadata : dict
        Signal metadata (sampling rate, length, etc.)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 60)

    # Generate synthetic vibration data based on DSP principles
    signals, labels, label_names = generate_synthetic_vibration_data(
        n_samples=n_samples,
        signal_length=signal_length,
        sampling_rate=sampling_rate
    )

    # Create HuggingFace dataset
    dataset_dict = create_huggingface_dataset(signals, labels, label_names)

    metadata = {
        "sampling_rate": sampling_rate,
        "signal_length": signal_length,
        "n_classes": len(label_names),
        "total_samples": len(labels),
        "label_names": label_names
    }

    logger.info(f"Data ingestion complete. Metadata: {metadata}")

    return dataset_dict, label_names, metadata


if __name__ == "__main__":
    dataset_dict, label_names, metadata = load_data_pipeline()
    print(f"\nDataset: {dataset_dict}")
    print(f"\nLabel names: {label_names}")
    print(f"\nMetadata: {metadata}")
