"""
AWS SageMaker Deployment Module
================================
Provides deployment infrastructure for the bearing fault detection model
on AWS SageMaker. Includes endpoint creation, model monitoring, and
CI/CD pipeline configuration.

NOTE: This module provides the deployment code structure and configuration.
Actual deployment requires AWS credentials and SageMaker access.
"""

import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# SageMaker Inference Script (entry_point for deployment)
# ============================================================

INFERENCE_SCRIPT = '''
import os
import json
import joblib
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import fft, fftfreq


def model_fn(model_dir):
    """Load the trained model and scaler."""
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return {"model": model, "scaler": scaler}


def input_fn(request_body, request_content_type):
    """Parse input data from the request."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["features"]).reshape(1, -1)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """Run prediction using the loaded model."""
    scaler = model_dict["scaler"]
    model = model_dict["model"]

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)

    return {
        "prediction": int(prediction[0]),
        "probabilities": probabilities[0].tolist()
    }


def output_fn(prediction, response_content_type):
    """Format the prediction output."""
    if response_content_type == "application/json":
        label_names = {
            0: "Normal",
            1: "Inner Race Fault",
            2: "Outer Race Fault",
            3: "Ball Fault"
        }
        prediction["label"] = label_names.get(prediction["prediction"], "Unknown")
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")
'''


# ============================================================
# SageMaker Deployment Configuration
# ============================================================

SAGEMAKER_DEPLOY_CODE = '''
"""
SageMaker Deployment Script
Deploys the trained bearing fault detection model to a SageMaker endpoint.
Requires: AWS credentials configured (aws configure)
"""

import sagemaker
from sagemaker.sklearn import SKLearnModel
from sagemaker import Session
import boto3
import tarfile
import os


def package_model(model_dir, output_path="model.tar.gz"):
    """Package model artifacts for SageMaker."""
    with tarfile.open(output_path, "w:gz") as tar:
        for fname in ["best_model.pkl", "scaler.pkl"]:
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                tar.add(fpath, arcname=fname)
    print(f"Model packaged to {output_path}")
    return output_path


def deploy_to_sagemaker(
    model_path="model.tar.gz",
    role_arn="arn:aws:iam::role/SageMakerRole",
    instance_type="ml.m5.large",
    endpoint_name="bearing-fault-detector"
):
    """
    Deploy model to SageMaker real-time endpoint.

    Parameters
    ----------
    model_path : str
        Path to the model.tar.gz artifact
    role_arn : str
        IAM role ARN with SageMaker permissions
    instance_type : str
        EC2 instance type for the endpoint
    endpoint_name : str
        Name for the SageMaker endpoint
    """
    session = sagemaker.Session()
    bucket = session.default_bucket()

    # Upload model to S3
    model_s3_uri = session.upload_data(
        path=model_path,
        bucket=bucket,
        key_prefix="bearing-fault-detection/model"
    )
    print(f"Model uploaded to: {model_s3_uri}")

    # Create SageMaker model
    sklearn_model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        framework_version="1.2-1",
        py_version="py3",
    )

    # Deploy endpoint
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    print(f"Endpoint deployed: {endpoint_name}")
    return predictor


def invoke_endpoint(endpoint_name, features):
    """
    Invoke the SageMaker endpoint for inference.

    Parameters
    ----------
    endpoint_name : str
        Name of the deployed endpoint
    features : list
        Feature values for prediction
    """
    import json
    runtime = boto3.client("sagemaker-runtime")

    payload = json.dumps({"features": features})

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode())
    print(f"Prediction: {result}")
    return result


if __name__ == "__main__":
    # Example usage (requires AWS credentials)
    # package_model("models/")
    # deploy_to_sagemaker()
    pass
'''


# ============================================================
# CI/CD Pipeline Configuration (GitHub Actions)
# ============================================================

GITHUB_ACTIONS_WORKFLOW = '''
name: Bearing Fault Detection MLOps Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly retraining on Mondays

env:
  AWS_REGION: us-east-1
  SAGEMAKER_ENDPOINT: bearing-fault-detector
  PYTHON_VERSION: '3.10'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: python -m pytest tests/ -v
      - name: Run data validation
        run: python -c "from pipeline.data_ingestion import load_data_pipeline; load_data_pipeline(n_samples=50)"

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train model
        run: python pipeline/main_pipeline.py
      - name: Upload model artifacts
        uses: actions/upload-artifact@v4
        with:
          name: model-artifacts
          path: models/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: model-artifacts
          path: models/
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      - name: Deploy to SageMaker
        run: python pipeline/sagemaker_deploy.py

  monitor:
    needs: deploy
    runs-on: ubuntu-latest
    steps:
      - name: Check endpoint health
        run: |
          aws sagemaker describe-endpoint \\
            --endpoint-name ${{ env.SAGEMAKER_ENDPOINT }} \\
            --region ${{ env.AWS_REGION }}
      - name: Run model monitoring
        run: python pipeline/monitoring.py
'''


# ============================================================
# Model Monitoring Configuration
# ============================================================

MONITORING_CONFIG = {
    "data_quality": {
        "baseline_constraints": {
            "feature_drift_threshold": 0.1,
            "missing_value_threshold": 0.05,
            "outlier_detection": True
        },
        "monitoring_schedule": "hourly",
        "alert_channels": ["email", "cloudwatch"]
    },
    "model_quality": {
        "accuracy_threshold": 0.85,
        "f1_threshold": 0.80,
        "drift_detection_method": "kolmogorov_smirnov",
        "retraining_trigger": {
            "accuracy_drop": 0.10,
            "consecutive_failures": 3
        }
    },
    "operational": {
        "latency_threshold_ms": 100,
        "error_rate_threshold": 0.01,
        "throughput_monitoring": True,
        "auto_scaling": {
            "min_instances": 1,
            "max_instances": 4,
            "target_invocations_per_instance": 1000
        }
    }
}


def generate_deployment_files(output_dir="deployment"):
    """
    Generate all deployment configuration files.

    Parameters
    ----------
    output_dir : str
        Directory to save deployment files
    """
    logger.info("=" * 60)
    logger.info("STEP 5: DEPLOYMENT CONFIGURATION")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, ".github", "workflows"), exist_ok=True)

    # Save inference script
    with open(os.path.join(output_dir, "inference.py"), "w") as f:
        f.write(INFERENCE_SCRIPT)
    logger.info("Generated: inference.py")

    # Save deployment script
    with open(os.path.join(output_dir, "deploy_sagemaker.py"), "w") as f:
        f.write(SAGEMAKER_DEPLOY_CODE)
    logger.info("Generated: deploy_sagemaker.py")

    # Save CI/CD workflow
    workflow_path = os.path.join(output_dir, ".github", "workflows", "mlops_pipeline.yml")
    with open(workflow_path, "w") as f:
        f.write(GITHUB_ACTIONS_WORKFLOW)
    logger.info("Generated: .github/workflows/mlops_pipeline.yml")

    # Save monitoring config
    with open(os.path.join(output_dir, "monitoring_config.json"), "w") as f:
        json.dump(MONITORING_CONFIG, f, indent=2)
    logger.info("Generated: monitoring_config.json")

    # Save Dockerfile
    dockerfile = """FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/
COPY models/ ./models/
COPY inference.py .

EXPOSE 8080

CMD ["python", "-m", "sagemaker_containers.serve"]
"""
    with open(os.path.join(output_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)
    logger.info("Generated: Dockerfile")

    logger.info(f"\nAll deployment files saved to {output_dir}/")

    return output_dir


if __name__ == "__main__":
    generate_deployment_files()
