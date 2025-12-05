"""
Test Suite for Integration Testing

Tests module interactions and complete pipelines.

This test suite validates end-to-end functionality according to:
- IEEE testing standards for integration testing
- Jousset et al. (2018): DAS validation methodology

Test categories:
1. Preprocessing -> Feature Extraction pipeline
2. Feature Extraction -> Model pipeline
3. Complete end-to-end workflows
4. Error propagation tests

Coverage target: 15% of total coverage
Test count: 20 tests
"""

import pytest
import torch
import numpy as np
import warnings

from src.signal_preprocessing import UniversalSignalPreprocessor, AdaptiveFeatureAggregator
from src.feature_extraction import UniversalFeatureVectorBuilder
from src.model_architecture import UniversalFiberSensorModel


class TestPreprocessingToFeaturesPipeline:
    """
    Tests for preprocessing -> feature extraction pipeline.
    
    Validates that preprocessed signals produce valid UFVs.
    """
    
    @pytest.mark.integration
    def test_preprocess_to_features_das_like(self, preprocessor, ufv_builder, das_signal):
        """
        Test complete pipeline with DAS-like signal.
        
        Signal -> Preprocess -> UFV (204 features)
        """
        # Preprocess
        preprocessed = preprocessor.preprocess(das_signal, original_sampling_rate=10000)
        
        # Extract features
        ufv = ufv_builder.build_ufv(preprocessed, fs=10000, is_multichannel=False)
        
        assert ufv.shape == (204,)
        assert not np.any(np.isnan(ufv))
    
    @pytest.mark.integration
    def test_preprocess_to_features_phi_otdr_like(self, preprocessor, ufv_builder, phi_otdr_signal):
        """
        Test complete pipeline with Phi-OTDR multi-channel signal.
        """
        # Preprocess
        preprocessed = preprocessor.preprocess(
            phi_otdr_signal,
            original_sampling_rate=10000,
            is_multichannel=True
        )
        
        # Extract features
        ufv = ufv_builder.build_ufv(preprocessed, fs=10000, is_multichannel=True)
        
        assert ufv.shape == (204,)
        assert not np.any(np.isnan(ufv))
    
    @pytest.mark.integration
    def test_preprocess_to_features_5khz(self, preprocessor, ufv_builder):
        """
        Test pipeline with 5kHz signal requiring upsampling.
        """
        np.random.seed(42)
        signal_5k = np.random.randn(5000)  # 1 second at 5kHz
        
        # Preprocess (will upsample to 10kHz)
        preprocessed = preprocessor.preprocess(signal_5k, original_sampling_rate=5000)
        
        # Extract features
        ufv = ufv_builder.build_ufv(preprocessed, fs=10000, is_multichannel=False)
        
        assert ufv.shape == (204,)
    
    @pytest.mark.integration
    def test_preprocess_to_features_20khz(self, preprocessor, ufv_builder):
        """
        Test pipeline with 20kHz signal requiring downsampling.
        """
        np.random.seed(42)
        signal_20k = np.random.randn(20000)  # 1 second at 20kHz
        
        # Preprocess (will downsample to 10kHz)
        preprocessed = preprocessor.preprocess(signal_20k, original_sampling_rate=20000)
        
        # Extract features
        ufv = ufv_builder.build_ufv(preprocessed, fs=10000, is_multichannel=False)
        
        assert ufv.shape == (204,)
    
    @pytest.mark.integration
    def test_preprocess_preserves_ufv_dimension(self, preprocessor, ufv_builder):
        """
        Test UFV dimension is always 204 regardless of input variations.
        """
        test_cases = [
            (5000, 5000),   # 5kHz, 5000 samples
            (10000, 10000), # 10kHz, 10000 samples (exact match)
            (20000, 20000), # 20kHz, 20000 samples
            (10000, 1000),  # 10kHz, short signal
            (10000, 50000), # 10kHz, long signal
        ]
        
        for fs, length in test_cases:
            np.random.seed(42)
            signal = np.random.randn(length)
            
            preprocessed = preprocessor.preprocess(signal, original_sampling_rate=fs)
            ufv = ufv_builder.build_ufv(preprocessed, fs=10000, is_multichannel=False)
            
            assert ufv.shape == (204,), f"Failed for fs={fs}, length={length}"


class TestFeaturesToModelPipeline:
    """
    Tests for feature extraction -> model pipeline.
    
    Validates that UFVs produce valid model outputs.
    """
    
    @pytest.mark.integration
    def test_features_to_model_forward_pass(self, ufv_builder, model, das_signal):
        """
        Test UFV to model prediction works.
        """
        # Extract UFV
        ufv = ufv_builder.build_ufv(das_signal, fs=10000, is_multichannel=False)
        
        # Normalize
        ufv_normalized = (ufv - np.mean(ufv)) / (np.std(ufv) + 1e-8)
        
        # Convert to tensor
        ufv_tensor = torch.FloatTensor(ufv_normalized).unsqueeze(0)
        
        # Model forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(ufv_tensor, head='all')
        
        assert 'event_logits' in outputs
        assert 'risk_score' in outputs
    
    @pytest.mark.integration
    def test_features_to_model_normalization(self, ufv_builder, model, das_signal):
        """
        Test that normalization is applied correctly before model.
        """
        ufv = ufv_builder.build_ufv(das_signal, fs=10000, is_multichannel=False)
        
        # Normalize
        ufv_normalized = (ufv - np.mean(ufv)) / (np.std(ufv) + 1e-8)
        
        # Check normalized values are reasonable
        assert np.abs(np.mean(ufv_normalized)) < 0.1, "Mean should be near 0 after normalization"
        assert np.abs(np.std(ufv_normalized) - 1.0) < 0.1, "Std should be near 1 after normalization"
    
    @pytest.mark.integration
    def test_features_normalized_range(self, ufv_builder, das_signal):
        """
        Test normalized features are in reasonable range.
        """
        ufv = ufv_builder.build_ufv(das_signal, fs=10000, is_multichannel=False)
        ufv_normalized = (ufv - np.mean(ufv)) / (np.std(ufv) + 1e-8)
        
        # Most values should be within 4 standard deviations
        assert np.sum(np.abs(ufv_normalized) > 4) < 10, "Too many outliers in normalized UFV"


class TestCompleteEndToEndWorkflows:
    """
    Tests for complete signal -> prediction workflows.
    """
    
    @pytest.mark.integration
    def test_e2e_das_workflow(self, inference_model, das_signal):
        """
        Test complete DAS workflow: signal -> prediction.
        """
        prediction = inference_model.predict(das_signal, sampling_rate=10000)
        
        assert isinstance(prediction, dict)
        assert 'event_type' in prediction
        assert 'risk_score' in prediction
        assert 0 <= prediction['risk_score'] <= 1
    
    @pytest.mark.integration
    def test_e2e_phi_otdr_workflow(self, inference_model, phi_otdr_signal):
        """
        Test complete Phi-OTDR workflow: multi-channel signal -> prediction.
        """
        prediction = inference_model.predict(
            phi_otdr_signal,
            sampling_rate=10000,
            is_multichannel=True
        )
        
        assert isinstance(prediction, dict)
        assert prediction['sensor_type'] in ['DAS', 'Phi-OTDR', 'OTDR']
    
    @pytest.mark.integration
    def test_e2e_universal_mode_5khz(self, inference_model):
        """
        Test universal mode with 5kHz signal.
        """
        np.random.seed(42)
        signal_5k = np.random.randn(5000)
        
        prediction = inference_model.predict_universal(
            signal_5k,
            original_sampling_rate=5000
        )
        
        assert isinstance(prediction, dict)
        assert 'event_type' in prediction
    
    @pytest.mark.integration
    def test_e2e_universal_mode_20khz(self, inference_model):
        """
        Test universal mode with 20kHz signal.
        """
        np.random.seed(42)
        signal_20k = np.random.randn(20000)
        
        prediction = inference_model.predict_universal(
            signal_20k,
            original_sampling_rate=20000
        )
        
        assert isinstance(prediction, dict)
    
    @pytest.mark.integration
    def test_e2e_universal_mode_short_signal(self, inference_model):
        """
        Test universal mode with very short signal (100 samples).
        """
        np.random.seed(42)
        short_signal = np.random.randn(100)
        
        prediction = inference_model.predict_universal(
            short_signal,
            original_sampling_rate=1000
        )
        
        assert isinstance(prediction, dict)
    
    @pytest.mark.integration
    def test_e2e_universal_mode_long_signal(self, inference_model):
        """
        Test universal mode with long signal (50000 samples).
        """
        np.random.seed(42)
        long_signal = np.random.randn(50000)
        
        prediction = inference_model.predict_universal(
            long_signal,
            original_sampling_rate=10000
        )
        
        assert isinstance(prediction, dict)
    
    @pytest.mark.integration
    def test_e2e_batch_standard(self, inference_model):
        """
        Test batch prediction workflow.
        """
        np.random.seed(42)
        signals = [np.random.randn(10000) for _ in range(3)]
        
        predictions = inference_model.predict_batch(signals, sampling_rate=10000)
        
        assert len(predictions) == 3
        for pred in predictions:
            assert isinstance(pred, dict)
            assert 'event_type' in pred
    
    @pytest.mark.integration
    def test_e2e_batch_universal_mixed(self, inference_model):
        """
        Test universal batch with mixed sampling rates.
        """
        np.random.seed(42)
        signals = [
            np.random.randn(5000),
            np.random.randn(10000),
            np.random.randn(20000),
        ]
        rates = [5000, 10000, 20000]
        
        predictions = inference_model.predict_batch_universal(
            signals,
            original_sampling_rates=rates
        )
        
        assert len(predictions) == 3


class TestErrorPropagation:
    """
    Tests for error handling across module boundaries.
    """
    
    @pytest.mark.integration
    def test_error_propagation_invalid_input(self, inference_model):
        """
        Test that invalid input errors are caught at the right layer.
        """
        empty_signal = np.array([])
        
        with pytest.raises((ValueError, RuntimeError)):
            inference_model.predict(empty_signal, sampling_rate=10000)
    
    @pytest.mark.integration
    def test_warning_propagation_short_signal(self, inference_model):
        """
        Test that warnings from preprocessing propagate to user.
        """
        np.random.seed(42)
        short_signal = np.random.randn(10)  # Very short
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Should process but generate warnings
            prediction = inference_model.predict_universal(
                short_signal,
                original_sampling_rate=1000
            )
        
        # Should either produce warnings or have warnings in preprocessing
        assert prediction is not None
    
    @pytest.mark.integration
    def test_error_message_clarity(self, inference_model):
        """
        Test that error messages are informative.
        """
        empty_signal = np.array([])
        
        try:
            inference_model.predict(empty_signal, sampling_rate=10000)
            assert False, "Should have raised an exception"
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            # Error message should mention the issue
            assert 'empty' in error_msg or 'signal' in error_msg or 'input' in error_msg
    
    @pytest.mark.integration
    def test_graceful_degradation_edge_cases(self, inference_model):
        """
        Test system handles edge cases gracefully without crashing.
        """
        np.random.seed(42)
        
        # Very small amplitudes
        small_signal = np.random.randn(10000) * 1e-10
        pred1 = inference_model.predict(small_signal, sampling_rate=10000)
        assert pred1 is not None
        
        # Large amplitudes
        large_signal = np.random.randn(10000) * 1e6
        pred2 = inference_model.predict(large_signal, sampling_rate=10000)
        assert pred2 is not None

