"""
Pytest Configuration and Shared Fixtures

This module provides shared test fixtures for the Universal Fiber Sensor Model
test suite. All fixtures use fixed random seeds for deterministic test execution.

Based on:
- pytest best practices (Brian Okken, "Python Testing with pytest", 2022)
- IEEE Standard 829-2008: Software Test Documentation
- Google ML Testing Framework (Breck et al., 2020)
"""

import pytest
import numpy as np
import torch
import os
import sys
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_preprocessing import UniversalSignalPreprocessor, AdaptiveFeatureAggregator
from src.feature_extraction import (
    MultiDomainFeatureExtractor,
    ProprietaryFeatures,
    UniversalFeatureVectorBuilder
)
from src.model_architecture import (
    FusionLayer,
    MultiHeadClassifier,
    UniversalFiberSensorModel
)

# Import signal generators
from tests.fixtures.signals import (
    generate_synthetic_signal,
    generate_das_like_signal,
    generate_multichannel_signal,
    generate_edge_case_signals,
    generate_multi_frequency_signal,
    generate_white_noise,
    generate_known_frequency_signal,
)


# =============================================================================
# Global Configuration
# =============================================================================

# Fixed random seed for deterministic tests
RANDOM_SEED = 42

# Standard test parameters
STANDARD_FS = 10000  # Hz
STANDARD_LENGTH = 10000  # samples
STANDARD_DURATION = 1.0  # seconds


@pytest.fixture(autouse=True)
def set_random_seeds():
    """
    Automatically set random seeds before each test for determinism.
    
    This fixture runs automatically for every test to ensure reproducibility.
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


@pytest.fixture
def suppress_warnings():
    """Suppress warnings during tests that intentionally trigger them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# =============================================================================
# Signal Fixtures
# =============================================================================

@pytest.fixture
def synthetic_sine_1khz():
    """
    Pure 1kHz sine wave at 10kHz sampling rate.
    
    Used for frequency preservation tests in resampling.
    """
    return generate_synthetic_signal(
        freq=1000.0,
        duration=1.0,
        fs=10000,
        noise_level=0.0,
        seed=RANDOM_SEED
    )


@pytest.fixture
def synthetic_sine_with_noise():
    """
    1kHz sine wave with 10% noise at 10kHz sampling rate.
    
    Used for testing robustness to noise.
    """
    return generate_synthetic_signal(
        freq=1000.0,
        duration=1.0,
        fs=10000,
        noise_level=0.1,
        seed=RANDOM_SEED
    )


@pytest.fixture
def multi_frequency_signal():
    """
    Signal with 500Hz, 1kHz, and 2kHz components.
    
    Used for anti-aliasing filter tests.
    """
    return generate_multi_frequency_signal(
        frequencies=(500.0, 1000.0, 2000.0),
        amplitudes=(1.0, 1.0, 1.0),
        duration=1.0,
        fs=10000,
        seed=RANDOM_SEED
    )


@pytest.fixture
def das_signal():
    """
    DAS-like single channel signal.
    
    Standard: 10kHz, 10000 samples
    """
    return generate_das_like_signal(
        length=STANDARD_LENGTH,
        fs=STANDARD_FS,
        seed=RANDOM_SEED
    )


@pytest.fixture
def phi_otdr_signal():
    """
    Phi-OTDR-like multi-channel signal (12 channels).
    
    Standard: 10kHz, 10000 samples, 12 channels
    """
    return generate_multichannel_signal(
        length=STANDARD_LENGTH,
        channels=12,
        fs=STANDARD_FS,
        correlation=0.7,
        seed=RANDOM_SEED
    )


@pytest.fixture
def short_signal():
    """
    Very short signal for edge case testing.
    
    100 samples at 1kHz (0.1 seconds)
    """
    np.random.seed(RANDOM_SEED)
    return np.random.randn(100).astype(np.float64)


@pytest.fixture
def long_signal():
    """
    Long signal for multi-window averaging tests.
    
    50000 samples at 10kHz (5 seconds)
    """
    np.random.seed(RANDOM_SEED)
    return np.random.randn(50000).astype(np.float64)


@pytest.fixture
def exact_length_signal():
    """
    Signal with exact target length (10000 samples).
    
    Used to test no-modification path.
    """
    np.random.seed(RANDOM_SEED)
    return np.random.randn(STANDARD_LENGTH).astype(np.float64)


@pytest.fixture
def white_noise_signal():
    """
    White Gaussian noise with known properties.
    
    Mean ~0, Std ~1, 10000 samples
    """
    return generate_white_noise(
        length=STANDARD_LENGTH,
        std=1.0,
        seed=RANDOM_SEED
    )


@pytest.fixture
def zero_signal():
    """All-zeros signal for edge case testing."""
    return np.zeros(STANDARD_LENGTH, dtype=np.float64)


@pytest.fixture
def constant_signal():
    """Constant value signal for edge case testing."""
    return np.full(STANDARD_LENGTH, 3.14159, dtype=np.float64)


@pytest.fixture
def edge_cases():
    """
    Dictionary of all edge case signals.
    
    Includes: empty, nan, inf, very_short, very_long, etc.
    """
    return generate_edge_case_signals(seed=RANDOM_SEED)


# =============================================================================
# Module Fixtures
# =============================================================================

@pytest.fixture
def preprocessor():
    """
    Initialized UniversalSignalPreprocessor with default settings.
    
    Target: 10kHz, 1.0 second duration
    """
    return UniversalSignalPreprocessor(
        target_sampling_rate=STANDARD_FS,
        target_duration=STANDARD_DURATION,
        use_multi_window=True
    )


@pytest.fixture
def preprocessor_no_multiwindow():
    """
    UniversalSignalPreprocessor with multi-window disabled.
    
    Uses center truncation for long signals.
    """
    return UniversalSignalPreprocessor(
        target_sampling_rate=STANDARD_FS,
        target_duration=STANDARD_DURATION,
        use_multi_window=False
    )


@pytest.fixture
def feature_extractor():
    """
    Initialized MultiDomainFeatureExtractor.
    
    Standard: 10kHz sampling rate
    """
    return MultiDomainFeatureExtractor(fs=STANDARD_FS)


@pytest.fixture
def proprietary_features():
    """Initialized ProprietaryFeatures extractor."""
    return ProprietaryFeatures()


@pytest.fixture
def ufv_builder():
    """
    Initialized UniversalFeatureVectorBuilder.
    
    Builds complete 204-dimensional UFV.
    """
    return UniversalFeatureVectorBuilder()


@pytest.fixture
def adaptive_aggregator():
    """
    Initialized AdaptiveFeatureAggregator.
    
    Default: 10000 window size, 50% overlap, mean aggregation
    """
    return AdaptiveFeatureAggregator(
        window_size=STANDARD_LENGTH,
        overlap=0.5,
        aggregation_method='mean'
    )


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def fusion_layer():
    """
    Initialized FusionLayer for testing.
    
    Input: 204-dim, Output: 128-dim embedding
    """
    model = FusionLayer(input_dim=204, hidden_dim=256, output_dim=128)
    model.eval()
    return model


@pytest.fixture
def multi_head_classifier():
    """
    Initialized MultiHeadClassifier for testing.
    
    15 event classes, 4 damage classes, 3 sensor types
    """
    model = MultiHeadClassifier(
        embedding_dim=128,
        num_event_classes=15,
        num_damage_classes=4,
        num_sensor_types=3
    )
    model.eval()
    return model


@pytest.fixture
def model():
    """
    Initialized UniversalFiberSensorModel (untrained).
    
    Complete model without loaded weights.
    """
    model = UniversalFiberSensorModel(
        ufv_dim=204,
        embedding_dim=128,
        num_event_classes=15,
        num_damage_classes=4,
        num_sensor_types=3
    )
    model.eval()
    return model


@pytest.fixture
def model_training_mode():
    """
    UniversalFiberSensorModel in training mode.
    
    Used for gradient flow tests.
    """
    model = UniversalFiberSensorModel(
        ufv_dim=204,
        embedding_dim=128,
        num_event_classes=15,
        num_damage_classes=4,
        num_sensor_types=3
    )
    model.train()
    return model


# =============================================================================
# Inference Fixtures
# =============================================================================

@pytest.fixture
def model_path():
    """Path to the trained model checkpoint."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models',
        'trained_model.pth'
    )


@pytest.fixture
def inference_model(model_path):
    """
    Fully initialized FiberSensorInference with trained weights.
    
    Note: This fixture requires the trained model file to exist.
    Skip tests using this fixture if model is unavailable.
    """
    if not os.path.exists(model_path):
        pytest.skip(f"Trained model not found at {model_path}")
    
    from src.inference import FiberSensorInference
    return FiberSensorInference(model_path=model_path, device='cpu')


# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def sample_ufv():
    """
    Sample 204-dimensional UFV tensor for model testing.
    
    Normalized to have mean ~0, std ~1.
    """
    np.random.seed(RANDOM_SEED)
    ufv = np.random.randn(204).astype(np.float32)
    ufv = (ufv - np.mean(ufv)) / (np.std(ufv) + 1e-8)
    return torch.FloatTensor(ufv)


@pytest.fixture
def batch_ufv():
    """
    Batch of 8 UFV tensors for batch processing tests.
    
    Shape: (8, 204)
    """
    np.random.seed(RANDOM_SEED)
    batch = np.random.randn(8, 204).astype(np.float32)
    # Normalize each sample
    batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True) + 1e-8)
    return torch.FloatTensor(batch)


@pytest.fixture
def sample_embedding():
    """
    Sample 128-dimensional embedding tensor.
    
    Used for classifier head testing.
    """
    np.random.seed(RANDOM_SEED)
    embedding = np.random.randn(128).astype(np.float32)
    return torch.FloatTensor(embedding).unsqueeze(0)


# =============================================================================
# Expected Values Fixtures
# =============================================================================

@pytest.fixture
def expected_dimensions():
    """
    Dictionary of expected feature dimensions.
    
    Used for dimension validation tests.
    """
    return {
        'mfcc': 120,
        'wavelet': 64,
        'spectral': 6,
        'temporal': 6,
        'spatial': 4,
        'standard_total': 200,
        'proprietary': 4,
        'ufv_total': 204,
        'embedding': 128,
        'event_classes': 15,
        'damage_classes': 4,
        'sensor_types': 3,
    }


@pytest.fixture
def event_class_names():
    """Expected event class names."""
    return [
        'car', 'walk', 'running', 'longboard', 'fence',
        'manipulation', 'construction', 'openclose', 'regular',
        'background', 'dig', 'knock', 'water', 'shake', 'walk_phi'
    ]


@pytest.fixture
def damage_class_names():
    """Expected damage class names."""
    return ['clean', 'reflective', 'non-reflective', 'saturated']


@pytest.fixture
def sensor_type_names():
    """Expected sensor type names."""
    return ['DAS', 'Phi-OTDR', 'OTDR']


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def assert_no_nan_inf():
    """
    Helper fixture to check arrays for NaN/Inf values.
    
    Returns a callable that raises AssertionError if NaN/Inf found.
    """
    def _checker(arr, name="array"):
        assert not np.any(np.isnan(arr)), f"{name} contains NaN values"
        assert not np.any(np.isinf(arr)), f"{name} contains Inf values"
    return _checker


@pytest.fixture
def assert_dimension():
    """
    Helper fixture to check array dimensions.
    
    Returns a callable that raises AssertionError if dimension mismatch.
    """
    def _checker(arr, expected_dim, name="array"):
        assert len(arr) == expected_dim, (
            f"{name} has wrong dimension: expected {expected_dim}, got {len(arr)}"
        )
    return _checker

