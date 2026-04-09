# Predictive Maintenance Model

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
{
  "accuracy": 1.0,
  "f1_weighted": 1.0,
  "classification_report": {
    "healthy": {
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0,
      "support": 24.0
    },
    "imbalance": {
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0,
      "support": 24.0
    },
    "inner_race_fault": {
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0,
      "support": 24.0
    },
    "outer_race_fault": {
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0,
      "support": 24.0
    },
    "accuracy": 1.0,
    "macro avg": {
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0,
      "support": 96.0
    },
    "weighted avg": {
      "precision": 1.0,
      "recall": 1.0,
      "f1-score": 1.0,
      "support": 96.0
    }
  },
  "confusion_matrix": [
    [
      24,
      0,
      0,
      0
    ],
    [
      0,
      24,
      0,
      0
    ],
    [
      0,
      0,
      24,
      0
    ],
    [
      0,
      0,
      0,
      24
    ]
  ]
}
```
