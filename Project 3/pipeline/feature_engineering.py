"""
DSP-Based Feature Engineering Module
=====================================
Extracts time-domain, frequency-domain, and time-frequency domain features
from raw vibration signals using Digital Signal Processing techniques.

This is the core DSP component of the pipeline, applying classical signal
processing methods to transform raw accelerometer data into discriminative
features for bearing fault classification.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import fft, fftfreq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DSPFeatureExtractor:
    """
    Digital Signal Processing feature extractor for vibration signals.

    Extracts three categories of features:
    1. Time-domain statistical features
    2. Frequency-domain spectral features (via FFT)
    3. Time-frequency features (via STFT/envelope analysis)
    """

    def __init__(self, sampling_rate=12000):
        """
        Parameters
        ----------
        sampling_rate : int
            Sampling rate of the vibration signal in Hz
        """
        self.sampling_rate = sampling_rate
        self.feature_names = []

    def extract_time_domain_features(self, signal_data):
        """
        Extract statistical features from the time-domain signal.

        These features capture the amplitude distribution and temporal
        characteristics of the vibration signal.

        Parameters
        ----------
        signal_data : np.ndarray
            1D vibration signal

        Returns
        -------
        features : dict
            Time-domain feature dictionary
        """
        features = {}

        # Basic statistical features
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['variance'] = np.var(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data ** 2))
        features['peak'] = np.max(np.abs(signal_data))
        features['peak_to_peak'] = np.max(signal_data) - np.min(signal_data)

        # Shape features
        features['skewness'] = float(skew(signal_data))
        features['kurtosis'] = float(kurtosis(signal_data))

        # Crest factor: peak / RMS (indicator of impulsiveness)
        features['crest_factor'] = features['peak'] / (features['rms'] + 1e-10)

        # Shape factor: RMS / mean absolute value
        features['shape_factor'] = features['rms'] / (np.mean(np.abs(signal_data)) + 1e-10)

        # Impulse factor: peak / mean absolute value
        features['impulse_factor'] = features['peak'] / (np.mean(np.abs(signal_data)) + 1e-10)

        # Clearance factor
        features['clearance_factor'] = features['peak'] / (np.mean(np.sqrt(np.abs(signal_data))) ** 2 + 1e-10)

        # Energy
        features['energy'] = np.sum(signal_data ** 2)

        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)

        # Entropy of amplitude distribution
        hist, _ = np.histogram(signal_data, bins=50, density=True)
        hist = hist[hist > 0]
        features['signal_entropy'] = float(entropy(hist))

        return features

    def extract_frequency_domain_features(self, signal_data):
        """
        Extract features from the frequency domain using FFT.

        FFT transforms the time-domain signal into its frequency components,
        revealing characteristic fault frequencies of bearings.

        Parameters
        ----------
        signal_data : np.ndarray
            1D vibration signal

        Returns
        -------
        features : dict
            Frequency-domain feature dictionary
        """
        features = {}
        n = len(signal_data)

        # Compute FFT
        yf = fft(signal_data)
        xf = fftfreq(n, 1.0 / self.sampling_rate)

        # Take only positive frequencies
        positive_mask = xf > 0
        xf_pos = xf[positive_mask]
        magnitude = 2.0 / n * np.abs(yf[positive_mask])
        power_spectrum = magnitude ** 2

        # Spectral centroid (center of mass of the spectrum)
        features['spectral_centroid'] = np.sum(xf_pos * magnitude) / (np.sum(magnitude) + 1e-10)

        # Spectral spread (bandwidth)
        features['spectral_spread'] = np.sqrt(
            np.sum(((xf_pos - features['spectral_centroid']) ** 2) * magnitude)
            / (np.sum(magnitude) + 1e-10)
        )

        # Spectral skewness
        features['spectral_skewness'] = (
            np.sum(((xf_pos - features['spectral_centroid']) ** 3) * magnitude)
            / ((features['spectral_spread'] ** 3 + 1e-10) * (np.sum(magnitude) + 1e-10))
        )

        # Spectral kurtosis
        features['spectral_kurtosis'] = (
            np.sum(((xf_pos - features['spectral_centroid']) ** 4) * magnitude)
            / ((features['spectral_spread'] ** 4 + 1e-10) * (np.sum(magnitude) + 1e-10))
        )

        # Spectral entropy
        ps_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        ps_norm = ps_norm[ps_norm > 0]
        features['spectral_entropy'] = float(entropy(ps_norm))

        # Dominant frequency and its magnitude
        dom_idx = np.argmax(magnitude)
        features['dominant_frequency'] = xf_pos[dom_idx]
        features['dominant_magnitude'] = magnitude[dom_idx]

        # Total spectral energy
        features['spectral_energy'] = np.sum(power_spectrum)

        # Mean frequency
        features['mean_frequency'] = np.sum(xf_pos * power_spectrum) / (np.sum(power_spectrum) + 1e-10)

        # Band energy ratios (low/mid/high frequency bands)
        nyquist = self.sampling_rate / 2
        low_mask = xf_pos < nyquist * 0.2
        mid_mask = (xf_pos >= nyquist * 0.2) & (xf_pos < nyquist * 0.6)
        high_mask = xf_pos >= nyquist * 0.6

        total_energy = np.sum(power_spectrum) + 1e-10
        features['low_band_energy_ratio'] = np.sum(power_spectrum[low_mask]) / total_energy
        features['mid_band_energy_ratio'] = np.sum(power_spectrum[mid_mask]) / total_energy
        features['high_band_energy_ratio'] = np.sum(power_spectrum[high_mask]) / total_energy

        return features

    def extract_envelope_features(self, signal_data):
        """
        Extract envelope analysis features using the Hilbert transform.

        Envelope analysis is a key DSP technique for bearing fault detection.
        The envelope (amplitude modulation) of the signal often contains
        the characteristic fault frequencies more clearly than the raw signal.

        Parameters
        ----------
        signal_data : np.ndarray
            1D vibration signal

        Returns
        -------
        features : dict
            Envelope feature dictionary
        """
        features = {}

        # Compute analytic signal using Hilbert transform
        analytic_signal = scipy_signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = (np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sampling_rate)

        # Envelope statistics
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_kurtosis'] = float(kurtosis(envelope))
        features['envelope_skewness'] = float(skew(envelope))

        # Instantaneous frequency statistics
        features['inst_freq_mean'] = np.mean(instantaneous_freq)
        features['inst_freq_std'] = np.std(instantaneous_freq)

        # Envelope spectrum features (FFT of envelope)
        n = len(envelope)
        env_fft = fft(envelope)
        env_freq = fftfreq(n, 1.0 / self.sampling_rate)
        positive_mask = env_freq > 0
        env_magnitude = 2.0 / n * np.abs(env_fft[positive_mask])
        env_freq_pos = env_freq[positive_mask]

        # Dominant envelope frequency
        if len(env_magnitude) > 0:
            dom_idx = np.argmax(env_magnitude)
            features['envelope_dominant_freq'] = env_freq_pos[dom_idx]
            features['envelope_dominant_magnitude'] = env_magnitude[dom_idx]
        else:
            features['envelope_dominant_freq'] = 0.0
            features['envelope_dominant_magnitude'] = 0.0

        return features

    def extract_all_features(self, signal_data):
        """
        Extract all DSP features from a single vibration signal.

        Parameters
        ----------
        signal_data : np.ndarray
            1D vibration signal

        Returns
        -------
        features : dict
            Complete feature dictionary
        """
        features = {}
        features.update(self.extract_time_domain_features(signal_data))
        features.update(self.extract_frequency_domain_features(signal_data))
        features.update(self.extract_envelope_features(signal_data))
        return features

    def transform_dataset(self, dataset_dict):
        """
        Transform an entire HuggingFace DatasetDict by extracting DSP features.

        Parameters
        ----------
        dataset_dict : DatasetDict
            HuggingFace dataset with 'signal' and 'label' columns

        Returns
        -------
        result : dict
            Dictionary with 'train', 'validation', 'test' keys,
            each containing a DataFrame of features and labels
        """
        logger.info("=" * 60)
        logger.info("STEP 2: DSP FEATURE EXTRACTION")
        logger.info("=" * 60)

        result = {}
        for split_name in dataset_dict:
            split = dataset_dict[split_name]
            logger.info(f"Processing {split_name} split ({len(split)} samples)...")

            all_features = []
            for i, example in enumerate(split):
                signal_data = np.array(example["signal"])
                features = self.extract_all_features(signal_data)
                features["label"] = example["label"]
                all_features.append(features)

                if (i + 1) % 500 == 0:
                    logger.info(f"  Processed {i + 1}/{len(split)} samples")

            df = pd.DataFrame(all_features)
            result[split_name] = df

            if not self.feature_names:
                self.feature_names = [c for c in df.columns if c != "label"]

            logger.info(f"  {split_name}: {df.shape[0]} samples, {len(self.feature_names)} features")

        logger.info(f"\nTotal features extracted: {len(self.feature_names)}")
        logger.info(f"Feature categories:")
        logger.info(f"  Time-domain: 15 features")
        logger.info(f"  Frequency-domain: 13 features")
        logger.info(f"  Envelope/Hilbert: 8 features")

        return result


def run_feature_engineering(dataset_dict, sampling_rate=12000):
    """
    Run the complete feature engineering pipeline.

    Parameters
    ----------
    dataset_dict : DatasetDict
        HuggingFace dataset
    sampling_rate : int
        Signal sampling rate

    Returns
    -------
    feature_dfs : dict
        Feature DataFrames for each split
    extractor : DSPFeatureExtractor
        Fitted feature extractor (for reuse during inference)
    """
    extractor = DSPFeatureExtractor(sampling_rate=sampling_rate)
    feature_dfs = extractor.transform_dataset(dataset_dict)
    return feature_dfs, extractor


if __name__ == "__main__":
    from data_ingestion import load_data_pipeline

    dataset_dict, label_names, metadata = load_data_pipeline(n_samples=100)
    feature_dfs, extractor = run_feature_engineering(
        dataset_dict,
        sampling_rate=metadata["sampling_rate"]
    )

    print(f"\nTrain features shape: {feature_dfs['train'].shape}")
    print(f"\nFeature names: {extractor.feature_names}")
    print(f"\nSample features:\n{feature_dfs['train'].head()}")
