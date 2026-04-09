import os
import json
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Optional libraries
USE_PYCARET = True
USE_HUGGINGFACE = True

try:
    from pycaret.classification import setup, compare_models, pull, finalize_model, predict_model, save_model
except Exception:
    USE_PYCARET = False

try:
    from datasets import Dataset
    from huggingface_hub import HfApi
except Exception:
    USE_HUGGINGFACE = False


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "input_csv": "signals.csv",          # CSV file with signal columns + label column
    "label_column": "label",             # target class column
    "sampling_rate": 12000,               # Hz
    "lowcut": None,                       # set e.g. 10 for bandpass
    "highcut": None,                      # set e.g. 3000 for bandpass
    "test_size": 0.2,
    "random_state": 42,
    "model_output_dir": "artifacts",
    "hf_dataset_repo": "SinghHY/predictive-maintenance-dataset",
    "hf_model_repo": "SinghHY/predictive-maintenance-model"
}


# ============================================================
# SIGNAL PROCESSING HELPERS
# ============================================================
def butter_filter(signal, fs, lowcut=None, highcut=None, order=4):
    nyq = 0.5 * fs

    if lowcut is None and highcut is None:
        return signal

    if lowcut is not None and highcut is not None:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
    elif lowcut is not None:
        low = lowcut / nyq
        b, a = butter(order, low, btype="high")
    else:
        high = highcut / nyq
        b, a = butter(order, high, btype="low")

    return filtfilt(b, a, signal)


def spectral_entropy(power_spectrum):
    ps = np.asarray(power_spectrum, dtype=float)
    ps = ps / (np.sum(ps) + 1e-12)
    return -np.sum(ps * np.log2(ps + 1e-12))


def extract_features(signal, fs):
    signal = np.asarray(signal, dtype=float)

    # Remove DC offset
    signal = signal - np.mean(signal)

    # Normalize
    std = np.std(signal) + 1e-12
    signal = signal / std

    # FFT
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    mag = np.abs(fft_vals)
    power = mag ** 2

    dominant_idx = np.argmax(mag[1:]) + 1 if len(mag) > 1 else 0
    dominant_freq = fft_freqs[dominant_idx]
    dominant_amp = mag[dominant_idx]

    spectral_centroid = np.sum(fft_freqs * mag) / (np.sum(mag) + 1e-12)
    spec_energy = np.sum(power)
    spec_entropy = spectral_entropy(power)

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / (rms + 1e-12)

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "variance": np.var(signal),
        "rms": rms,
        "peak_to_peak": np.ptp(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "crest_factor": crest_factor,
        "dominant_freq": dominant_freq,
        "dominant_amp": dominant_amp,
        "spectral_centroid": spectral_centroid,
        "spectral_energy": spec_energy,
        "spectral_entropy": spec_entropy,
        "bandpower_total": np.mean(power),
    }
    return features


# ============================================================
# DATA LOADING
# ============================================================
def load_signal_table(csv_path, label_column):
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {csv_path}.")

    feature_cols = [c for c in df.columns if c != label_column]
    X_signals = df[feature_cols].values
    y = df[label_column].values
    return X_signals, y, feature_cols


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_feature_dataframe(X_signals, y, fs, lowcut=None, highcut=None):
    rows = []
    for i, row in enumerate(X_signals):
        signal = np.asarray(row, dtype=float)
        signal = butter_filter(signal, fs=fs, lowcut=lowcut, highcut=highcut)
        feats = extract_features(signal, fs)
        feats["label"] = y[i]
        rows.append(feats)
    return pd.DataFrame(rows)


# ============================================================
# AUTOML / MODEL TRAINING
# ============================================================
def train_with_pycaret(feature_df, label_col="label"):
    exp = setup(
        data=feature_df,
        target=label_col,
        session_id=CONFIG["random_state"],
        train_size=1 - CONFIG["test_size"],
        normalize=True,
        verbose=False,
        html=False,
    )
    best = compare_models()
    leaderboard = pull()
    final_model = finalize_model(best)
    return final_model, leaderboard


def train_with_sklearn(feature_df, label_col="label"):
    X = feature_df.drop(columns=[label_col])
    y = feature_df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=CONFIG["random_state"]))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return model, metrics


# ============================================================
# HUGGING FACE HELPERS
# ============================================================
def push_features_to_huggingface(feature_df, repo_id):
    if not USE_HUGGINGFACE:
        print("Hugging Face libraries not installed. Skipping dataset push.")
        return

    ds = Dataset.from_pandas(feature_df)
    ds.push_to_hub(repo_id)
    print(f"Feature dataset pushed to Hugging Face Hub: {repo_id}")


def create_model_card(repo_id, metrics_path):
    if not USE_HUGGINGFACE:
        print("Hugging Face libraries not installed. Skipping model card.")
        return

    api = HfApi()
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    model_card = f'''# Predictive Maintenance Model

This repository contains a machine learning model for classifying machine conditions from vibration-signal-derived features.

## Task
Multiclass classification of machine health condition.

## Pipeline
- Signal preprocessing
- Time-domain and frequency-domain feature extraction
- AutoML / classifier training
- Model persistence and reproducibility

## Metrics
```json
{json.dumps(metrics, indent=2)}
```
'''
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id=repo_id, repo_type="model")
    print(f"Model card uploaded to Hugging Face Hub: {repo_id}")


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(CONFIG["model_output_dir"], exist_ok=True)

    print("Loading signals...")
    X_signals, y, _ = load_signal_table(CONFIG["input_csv"], CONFIG["label_column"])

    print("Extracting features...")
    feature_df = build_feature_dataframe(
        X_signals,
        y,
        fs=CONFIG["sampling_rate"],
        lowcut=CONFIG["lowcut"],
        highcut=CONFIG["highcut"]
    )
    feature_path = os.path.join(CONFIG["model_output_dir"], "engineered_features.csv")
    feature_df.to_csv(feature_path, index=False)
    print(f"Saved engineered features to {feature_path}")

    if USE_PYCARET:
        print("Running AutoML with PyCaret...")
        final_model, leaderboard = train_with_pycaret(feature_df, label_col="label")
        leaderboard_path = os.path.join(CONFIG["model_output_dir"], "automl_leaderboard.csv")
        leaderboard.to_csv(leaderboard_path, index=False)
        print(leaderboard)
        try:
            save_model(final_model, os.path.join(CONFIG["model_output_dir"], "best_pycaret_model"))
        except Exception as e:
            print(f"Could not save PyCaret model: {e}")
    else:
        print("PyCaret not available. Using sklearn fallback model...")
        model, metrics = train_with_sklearn(feature_df, label_col="label")
        model_path = os.path.join(CONFIG["model_output_dir"], "rf_model.joblib")
        metrics_path = os.path.join(CONFIG["model_output_dir"], "metrics.json")
        dump(model, model_path)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))

    # Hugging Face integration
    if USE_HUGGINGFACE:
        try:
            push_features_to_huggingface(feature_df, CONFIG["hf_dataset_repo"])
        except Exception as e:
            print(f"Dataset push skipped: {e}")

        try:
            metrics_path = os.path.join(CONFIG["model_output_dir"], "metrics.json")
            create_model_card(CONFIG["hf_model_repo"], metrics_path)
        except Exception as e:
            print(f"Model card upload skipped: {e}")

    print("Pipeline completed.")


if __name__ == "__main__":
    main()
