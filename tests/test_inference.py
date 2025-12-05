"""
Test Suite for Inference Module

Tests FiberSensorInference class for model loading and prediction.

This test suite validates the inference pipeline according to:
- Breck et al. (2020): ML infrastructure testing
- Wu et al. (2021): ML for optical fiber sensing

Test categories:
1. Model loading tests
2. Standard prediction tests
3. Universal prediction tests
4. Batch prediction tests
5. Error handling tests

Coverage target: 85%+
Test count: 20 tests
"""

import pytest
import torch
import numpy as np
import os
import warnings

from src.inference import FiberSensorInference


class TestModelLoading:
    """Tests for model initialization and loading."""
    
    @pytest.mark.unit
    def test_init_model_loading(self, inference_model):
        """
        Test that model loads successfully from checkpoint.
        """
        assert inference_model is not None
        assert inference_model.model is not None
    
    @pytest.mark.unit
    def test_init_invalid_path_raises(self):
        """
        Test that invalid model path raises FileNotFoundError.
        """
        with pytest.raises(FileNotFoundError):
            FiberSensorInference(model_path='/nonexistent/path/model.pth')
    
    @pytest.mark.unit
    def test_init_device_detection(self, model_path):
        """
        Test auto-detection of CPU/GPU device.
        """
        if not os.path.exists(model_path):
            pytest.skip("Model file not found")
        
        model = FiberSensorInference(model_path=model_path, device=None)
        
        # Should have a valid device
        assert model.device in ['cpu', 'cuda']
    
    @pytest.mark.unit
    def test_init_explicit_cpu(self, model_path):
        """
        Test explicit CPU device selection.
        """
        if not os.path.exists(model_path):
            pytest.skip("Model file not found")
        
        model = FiberSensorInference(model_path=model_path, device='cpu')
        
        assert model.device == 'cpu'
    
    @pytest.mark.unit
    def test_class_names_loaded(self, inference_model, event_class_names, damage_class_names, sensor_type_names):
        """
        Test that class names are loaded (from checkpoint or defaults).
        """
        assert len(inference_model.event_classes) == 15
        assert len(inference_model.damage_classes) == 4
        assert len(inference_model.sensor_types) == 3


class TestStandardPrediction:
    """Tests for standard predict() method."""
    
    @pytest.mark.unit
    def test_predict_single_channel(self, inference_model, das_signal):
        """
        Test basic prediction on single-channel signal.
        """
        prediction = inference_model.predict(das_signal, sampling_rate=10000)
        
        assert prediction is not None
        assert isinstance(prediction, dict)
    
    @pytest.mark.unit
    def test_predict_multichannel(self, inference_model, phi_otdr_signal):
        """
        Test prediction on multi-channel signal.
        """
        prediction = inference_model.predict(
            phi_otdr_signal,
            sampling_rate=10000,
            is_multichannel=True
        )
        
        assert prediction is not None
    
    @pytest.mark.unit
    def test_predict_output_structure(self, inference_model, das_signal):
        """
        Test prediction output contains all 7 expected keys.
        """
        prediction = inference_model.predict(das_signal, sampling_rate=10000)
        
        expected_keys = [
            'event_type', 'event_confidence',
            'risk_score',
            'damage_type', 'damage_confidence',
            'sensor_type', 'sensor_confidence'
        ]
        
        for key in expected_keys:
            assert key in prediction, f"Missing key: {key}"
    
    @pytest.mark.unit
    def test_predict_confidence_ranges(self, inference_model, das_signal):
        """
        Test confidence scores are in valid range [0, 1].
        """
        prediction = inference_model.predict(das_signal, sampling_rate=10000)
        
        assert 0 <= prediction['event_confidence'] <= 1
        assert 0 <= prediction['damage_confidence'] <= 1
        assert 0 <= prediction['sensor_confidence'] <= 1
    
    @pytest.mark.unit
    def test_predict_risk_score_range(self, inference_model, das_signal):
        """
        Test risk score is in valid range [0, 1].
        """
        prediction = inference_model.predict(das_signal, sampling_rate=10000)
        
        assert 0 <= prediction['risk_score'] <= 1
    
    @pytest.mark.unit
    def test_predict_valid_class_names(self, inference_model, das_signal, event_class_names, damage_class_names, sensor_type_names):
        """
        Test predicted class names are valid strings from known classes.
        """
        prediction = inference_model.predict(das_signal, sampling_rate=10000)
        
        assert prediction['event_type'] in event_class_names
        assert prediction['damage_type'] in damage_class_names
        assert prediction['sensor_type'] in sensor_type_names


class TestUniversalPrediction:
    """Tests for universal predict_universal() method."""
    
    @pytest.mark.unit
    def test_predict_universal_5khz(self, inference_model):
        """
        Test universal prediction with 5kHz signal.
        """
        np.random.seed(42)
        signal_5k = np.random.randn(5000)  # 1 second at 5kHz
        
        prediction = inference_model.predict_universal(
            signal_5k,
            original_sampling_rate=5000
        )
        
        assert prediction is not None
        assert 'event_type' in prediction
    
    @pytest.mark.unit
    def test_predict_universal_20khz(self, inference_model):
        """
        Test universal prediction with 20kHz signal.
        """
        np.random.seed(42)
        signal_20k = np.random.randn(20000)  # 1 second at 20kHz
        
        prediction = inference_model.predict_universal(
            signal_20k,
            original_sampling_rate=20000
        )
        
        assert prediction is not None
    
    @pytest.mark.unit
    def test_predict_universal_100khz(self, inference_model):
        """
        Test universal prediction with high sampling rate (100kHz).
        """
        np.random.seed(42)
        signal_100k = np.random.randn(100000)  # 1 second at 100kHz
        
        prediction = inference_model.predict_universal(
            signal_100k,
            original_sampling_rate=100000
        )
        
        assert prediction is not None
    
    @pytest.mark.unit
    def test_predict_universal_short_signal(self, inference_model):
        """
        Test universal prediction with very short signal (100 samples).
        """
        np.random.seed(42)
        short_signal = np.random.randn(100)
        
        prediction = inference_model.predict_universal(
            short_signal,
            original_sampling_rate=1000
        )
        
        assert prediction is not None
    
    @pytest.mark.unit
    def test_predict_universal_long_signal(self, inference_model):
        """
        Test universal prediction with long signal (50000 samples).
        """
        np.random.seed(42)
        long_signal = np.random.randn(50000)
        
        prediction = inference_model.predict_universal(
            long_signal,
            original_sampling_rate=10000
        )
        
        assert prediction is not None
    
    @pytest.mark.unit
    def test_predict_universal_preprocessing_info(self, inference_model):
        """
        Test universal prediction returns preprocessing info when requested.
        """
        np.random.seed(42)
        signal = np.random.randn(5000)
        
        prediction, info = inference_model.predict_universal(
            signal,
            original_sampling_rate=5000,
            return_preprocessing_info=True
        )
        
        assert info is not None
        assert 'original_shape' in info
        assert 'processed_shape' in info
        assert 'resample_ratio' in info
    
    @pytest.mark.unit
    def test_predict_universal_auto_preprocess_false(self, inference_model, das_signal):
        """
        Test universal prediction falls back to standard predict when auto_preprocess=False.
        """
        prediction = inference_model.predict_universal(
            das_signal,
            original_sampling_rate=10000,
            auto_preprocess=False
        )
        
        assert prediction is not None


class TestBatchPrediction:
    """Tests for batch prediction methods."""
    
    @pytest.mark.unit
    def test_predict_batch_output_count(self, inference_model):
        """
        Test batch prediction returns correct number of predictions.
        """
        np.random.seed(42)
        signals = [np.random.randn(10000) for _ in range(5)]
        
        predictions = inference_model.predict_batch(signals, sampling_rate=10000)
        
        assert len(predictions) == 5
    
    @pytest.mark.unit
    def test_predict_batch_universal_mixed_rates(self, inference_model):
        """
        Test universal batch with different sampling rates per signal.
        """
        np.random.seed(42)
        signals = [
            np.random.randn(5000),   # 5kHz
            np.random.randn(10000),  # 10kHz
            np.random.randn(20000),  # 20kHz
        ]
        rates = [5000, 10000, 20000]
        
        predictions = inference_model.predict_batch_universal(
            signals,
            original_sampling_rates=rates
        )
        
        assert len(predictions) == 3
        for pred in predictions:
            assert 'event_type' in pred


class TestErrorHandling:
    """Tests for error handling in inference."""
    
    @pytest.mark.unit
    def test_validate_empty_signal_raises(self, inference_model):
        """
        Test that empty signal raises ValueError.
        """
        empty_signal = np.array([])
        
        with pytest.raises((ValueError, RuntimeError)):
            inference_model.predict(empty_signal, sampling_rate=10000)
    
    @pytest.mark.unit
    def test_validate_nan_signal_raises(self, inference_model, edge_cases):
        """
        Test that signal with NaN raises ValueError.
        """
        nan_signal = edge_cases['contains_nan']
        
        with pytest.raises((ValueError, RuntimeError)):
            inference_model.predict(nan_signal, sampling_rate=10000)
    
    @pytest.mark.unit
    def test_validate_invalid_sampling_rate(self, inference_model, das_signal):
        """
        Test that invalid sampling rate raises ValueError.
        """
        with pytest.raises((ValueError, RuntimeError)):
            inference_model.predict(das_signal, sampling_rate=0)
        
        with pytest.raises((ValueError, RuntimeError)):
            inference_model.predict(das_signal, sampling_rate=-1000)

