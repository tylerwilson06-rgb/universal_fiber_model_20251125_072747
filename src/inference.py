"""
Inference Module
Easy-to-use interface for model predictions with robust error handling
"""

import torch
import numpy as np
import warnings
import os
from src.model_architecture import UniversalFiberSensorModel
from src.feature_extraction import UniversalFeatureVectorBuilder
from src.signal_preprocessing import UniversalSignalPreprocessor, AdaptiveFeatureAggregator

class FiberSensorInference:
    """Robust inference interface with validation and error handling"""
    
    UFV_DIM = 204
    EXPECTED_SAMPLING_RATE_RANGE = (100, 1000000)  # Reasonable range for fiber sensors
    
    def __init__(self, model_path='models/trained_model.pth', device=None):
        """
        Initialize inference model
        
        Args:
            model_path: path to trained model checkpoint
            device: device to use ('cpu', 'cuda', or None for auto-detection)
        """
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Auto-detected device: {device}")
        
        self.device = device
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with error handling
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Initialize model
            self.model = UniversalFiberSensorModel()
            
            # Try to load state dict (handle different checkpoint formats)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load with strict=False to handle minor architecture differences
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                warnings.warn(f"Missing keys in state dict: {missing_keys[:5]}... (showing first 5)")
            if unexpected_keys:
                warnings.warn(f"Unexpected keys in state dict: {unexpected_keys[:5]}... (showing first 5)")
            
            self.model.eval()
            self.model.to(device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e
        
        # Load UFV builder
        self.ufv_builder = UniversalFeatureVectorBuilder()
        
        # Initialize universal signal preprocessor (for universal mode)
        self.universal_preprocessor = UniversalSignalPreprocessor()
        self.adaptive_aggregator = AdaptiveFeatureAggregator()
        
        # Load class names from checkpoint if available, otherwise use defaults
        if isinstance(checkpoint, dict):
            self.event_classes = checkpoint.get('event_classes', self._default_event_classes())
            self.damage_classes = checkpoint.get('damage_classes', self._default_damage_classes())
            self.sensor_types = checkpoint.get('sensor_types', self._default_sensor_types())
        else:
            self.event_classes = self._default_event_classes()
            self.damage_classes = self._default_damage_classes()
            self.sensor_types = self._default_sensor_types()
        
        # Normalization statistics (should ideally be loaded from training)
        # For now, using per-sample normalization as fallback
        self.normalization_stats = checkpoint.get('normalization_stats', None) if isinstance(checkpoint, dict) else None
    
    def _default_event_classes(self):
        return [
            'car', 'walk', 'running', 'longboard', 'fence',
            'manipulation', 'construction', 'openclose', 'regular',
            'background', 'dig', 'knock', 'water', 'shake', 'walk_phi'
        ]
    
    def _default_damage_classes(self):
        return ['clean', 'reflective', 'non-reflective', 'saturated']
    
    def _default_sensor_types(self):
        return ['DAS', 'Phi-OTDR', 'OTDR']
    
    def _validate_input(self, raw_signal, sampling_rate):
        """Validate input signal and parameters"""
        # Convert to numpy array if needed
        raw_signal = np.asarray(raw_signal, dtype=np.float64)
        
        # Check for empty signal
        if raw_signal.size == 0:
            raise ValueError("Input signal is empty")
        
        # Check for NaN or Inf
        if np.any(np.isnan(raw_signal)):
            raise ValueError("Input signal contains NaN values")
        if np.any(np.isinf(raw_signal)):
            warnings.warn("Input signal contains Inf values. Replacing with finite values.")
            raw_signal = np.nan_to_num(raw_signal, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Validate sampling rate
        if sampling_rate <= 0:
            raise ValueError(f"Invalid sampling rate: {sampling_rate}. Must be positive.")
        if not (self.EXPECTED_SAMPLING_RATE_RANGE[0] <= sampling_rate <= self.EXPECTED_SAMPLING_RATE_RANGE[1]):
            warnings.warn(
                f"Sampling rate {sampling_rate} is outside expected range "
                f"{self.EXPECTED_SAMPLING_RATE_RANGE}. Results may be unreliable."
            )
        
        return raw_signal
    
    def _normalize_ufv(self, ufv):
        """
        Normalize UFV using training statistics if available,
        otherwise use per-sample normalization as fallback
        """
        if self.normalization_stats is not None:
            # Use training-time normalization (per-feature)
            mean = self.normalization_stats.get('mean')
            std = self.normalization_stats.get('std')
            
            if mean is not None and std is not None:
                ufv = (ufv - mean) / (std + 1e-8)
                return ufv
        
        # Fallback: per-sample normalization (current approach)
        # NOTE: This may not match training normalization exactly
        ufv = (ufv - np.mean(ufv)) / (np.std(ufv) + 1e-8)
        return ufv
    
    def predict(self, raw_signal, sampling_rate=10000, is_multichannel=False):
        """
        Make prediction from raw sensor signal
        
        Args:
            raw_signal: numpy array (1D or 2D for multi-channel)
            sampling_rate: Hz (default: 10000)
            is_multichannel: bool, whether signal is multi-channel
        
        Returns:
            dict with predictions and confidence scores
        """
        # Validate input
        raw_signal = self._validate_input(raw_signal, sampling_rate)
        
        # Determine if multi-channel automatically
        if len(raw_signal.shape) == 2 and raw_signal.shape[1] > 1:
            is_multichannel = True
        elif len(raw_signal.shape) > 2:
            raise ValueError(f"Expected 1D or 2D signal, got shape {raw_signal.shape}")
        
        try:
            # Extract UFV
            ufv = self.ufv_builder.build_ufv(raw_signal, sampling_rate, is_multichannel)
            
            # Validate UFV dimension
            if len(ufv) != self.UFV_DIM:
                raise RuntimeError(
                    f"UFV dimension mismatch: expected {self.UFV_DIM}, got {len(ufv)}"
                )
            
            # Normalize
            ufv = self._normalize_ufv(ufv)
            
            # Convert to tensor
            ufv_tensor = torch.FloatTensor(ufv).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(ufv_tensor, head='all')
            
            # Parse outputs with error handling
            event_logits = outputs['event_logits'][0]
            event_idx = event_logits.argmax().item()
            event_conf = torch.softmax(event_logits, dim=0)[event_idx].item()
            
            risk_score = outputs['risk_score'][0][0].item()
            
            damage_logits = outputs['damage_logits'][0]
            damage_idx = damage_logits.argmax().item()
            damage_conf = torch.softmax(damage_logits, dim=0)[damage_idx].item()
            
            sensor_logits = outputs['sensor_logits'][0]
            sensor_idx = sensor_logits.argmax().item()
            sensor_conf = torch.softmax(sensor_logits, dim=0)[sensor_idx].item()
            
            # Validate indices
            if event_idx >= len(self.event_classes):
                warnings.warn(f"Event index {event_idx} out of range. Using index 0.")
                event_idx = 0
            if damage_idx >= len(self.damage_classes):
                warnings.warn(f"Damage index {damage_idx} out of range. Using index 0.")
                damage_idx = 0
            if sensor_idx >= len(self.sensor_types):
                warnings.warn(f"Sensor index {sensor_idx} out of range. Using index 0.")
                sensor_idx = 0
            
            return {
                'event_type': self.event_classes[event_idx],
                'event_confidence': event_conf,
                'risk_score': risk_score,
                'damage_type': self.damage_classes[damage_idx],
                'damage_confidence': damage_conf,
                'sensor_type': self.sensor_types[sensor_idx],
                'sensor_confidence': sensor_conf
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def predict_universal(self, raw_signal, original_sampling_rate=None, 
                         is_multichannel=False, auto_preprocess=True, return_preprocessing_info=False):
        """
        Universal prediction mode - handles ANY sampling rate and signal length automatically.
        
        This is the key feature that makes the model truly universal. It automatically:
        1. Resamples signals to the target rate (10kHz) if needed
        2. Handles variable lengths (padding, truncation, or multi-window averaging)
        3. Preserves signal quality through advanced preprocessing
        
        Args:
            raw_signal: numpy array of ANY length at ANY sampling rate
            original_sampling_rate: Original sampling rate in Hz. If None, attempts to estimate
                                   or uses default (10kHz). For best results, provide actual rate.
            is_multichannel: bool, whether signal is multi-channel
            auto_preprocess: If True (default), automatically preprocesses signal.
                           If False, uses original predict() method (requires proper input).
            return_preprocessing_info: If True, returns preprocessing metadata
        
        Returns:
            dict with predictions and confidence scores
            If return_preprocessing_info=True: (prediction_dict, preprocessing_info_dict)
        
        Example:
            # Works with any sampling rate and length!
            signal_5khz = np.random.randn(5000)  # 5kHz, 1 second
            pred = model.predict_universal(signal_5khz, original_sampling_rate=5000)
            
            signal_20khz = np.random.randn(20000)  # 20kHz, 1 second
            pred = model.predict_universal(signal_20khz, original_sampling_rate=20000)
            
            signal_short = np.random.randn(100)  # Very short signal
            pred = model.predict_universal(signal_short, original_sampling_rate=1000)
        """
        if not auto_preprocess:
            # Fall back to original method (requires proper input)
            if original_sampling_rate is None:
                original_sampling_rate = 10000
            return self.predict(raw_signal, original_sampling_rate, is_multichannel)
        
        # Universal preprocessing mode
        try:
            # Determine if multi-channel automatically
            if len(raw_signal.shape) == 2 and raw_signal.shape[1] > 1:
                is_multichannel = True
            elif len(raw_signal.shape) > 2:
                raise ValueError(f"Expected 1D or 2D signal, got shape {raw_signal.shape}")
            
            # If sampling rate not provided, use default (assumes target rate)
            if original_sampling_rate is None:
                original_sampling_rate = self.universal_preprocessor.TARGET_SAMPLING_RATE
                warnings.warn(
                    "Original sampling rate not provided. Assuming target rate (10kHz). "
                    "For best results, provide actual sampling rate."
                )
            
            # Preprocess signal to standard format
            preprocessed_signal, preprocessing_info = self.universal_preprocessor.preprocess(
                raw_signal,
                original_sampling_rate=original_sampling_rate,
                is_multichannel=is_multichannel,
                return_info=True
            )
            
            # Now use standard prediction with preprocessed signal
            # Signal is now at target rate (10kHz) and target length (10000 samples)
            prediction = self.predict(
                preprocessed_signal,
                sampling_rate=self.universal_preprocessor.TARGET_SAMPLING_RATE,
                is_multichannel=is_multichannel
            )
            
            # Add preprocessing info to prediction if requested
            if return_preprocessing_info:
                prediction['preprocessing_info'] = preprocessing_info
                return prediction, preprocessing_info
            
            return prediction
            
        except Exception as e:
            raise RuntimeError(f"Universal prediction failed: {e}") from e
    
    def predict_batch(self, raw_signals, sampling_rate=10000, is_multichannel=False):
        """
        Predict on a batch of signals (more efficient)
        
        Args:
            raw_signals: list of numpy arrays or 2D/3D array
            sampling_rate: Hz
            is_multichannel: bool
        
        Returns:
            list of prediction dictionaries
        """
        # Handle single signal
        if isinstance(raw_signals, np.ndarray) and len(raw_signals.shape) <= 2:
            return [self.predict(raw_signals, sampling_rate, is_multichannel)]
        
        # Process batch
        ufvs = []
        for signal in raw_signals:
            try:
                signal = self._validate_input(signal, sampling_rate)
                ufv = self.ufv_builder.build_ufv(signal, sampling_rate, is_multichannel)
                ufv = self._normalize_ufv(ufv)
                ufvs.append(ufv)
            except Exception as e:
                warnings.warn(f"Failed to process signal in batch: {e}")
                ufvs.append(np.zeros(self.UFV_DIM))  # Zero padding as fallback
        
        # Batch inference
        ufv_tensor = torch.FloatTensor(np.array(ufvs)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(ufv_tensor, head='all')
        
        # Parse batch outputs
        results = []
        batch_size = ufv_tensor.shape[0]
        
        for i in range(batch_size):
            event_idx = outputs['event_logits'][i].argmax().item()
            event_conf = torch.softmax(outputs['event_logits'][i], dim=0)[event_idx].item()
            
            risk_score = outputs['risk_score'][i][0].item()
            
            damage_idx = outputs['damage_logits'][i].argmax().item()
            damage_conf = torch.softmax(outputs['damage_logits'][i], dim=0)[damage_idx].item()
            
            sensor_idx = outputs['sensor_logits'][i].argmax().item()
            sensor_conf = torch.softmax(outputs['sensor_logits'][i], dim=0)[sensor_idx].item()
            
            results.append({
                'event_type': self.event_classes[min(event_idx, len(self.event_classes)-1)],
                'event_confidence': event_conf,
                'risk_score': risk_score,
                'damage_type': self.damage_classes[min(damage_idx, len(self.damage_classes)-1)],
                'damage_confidence': damage_conf,
                'sensor_type': self.sensor_types[min(sensor_idx, len(self.sensor_types)-1)],
                'sensor_confidence': sensor_conf
            })
        
        return results
    
    def predict_batch_universal(self, raw_signals, original_sampling_rates=None, 
                                is_multichannel=False, auto_preprocess=True):
        """
        Universal batch prediction - handles signals with different sampling rates/lengths.
        
        Args:
            raw_signals: list of numpy arrays (can have different lengths/rates)
            original_sampling_rates: list of sampling rates (one per signal) or single value
                                   If None, assumes target rate for all
            is_multichannel: bool, whether signals are multi-channel
            auto_preprocess: If True, automatically preprocesses each signal
        
        Returns:
            list of prediction dictionaries
        """
        if not isinstance(raw_signals, (list, tuple, np.ndarray)):
            raise ValueError("raw_signals must be a list, tuple, or numpy array")
        
        # Handle single signal
        if isinstance(raw_signals, np.ndarray) and len(raw_signals.shape) <= 2:
            rate = original_sampling_rates if isinstance(original_sampling_rates, (int, float)) else None
            return [self.predict_universal(raw_signals, rate, is_multichannel, auto_preprocess)]
        
        # Process batch
        if original_sampling_rates is None:
            original_sampling_rates = [None] * len(raw_signals)
        elif isinstance(original_sampling_rates, (int, float)):
            original_sampling_rates = [original_sampling_rates] * len(raw_signals)
        
        if len(original_sampling_rates) != len(raw_signals):
            raise ValueError(
                f"Number of sampling rates ({len(original_sampling_rates)}) "
                f"must match number of signals ({len(raw_signals)})"
            )
        
        results = []
        for signal, rate in zip(raw_signals, original_sampling_rates):
            try:
                pred = self.predict_universal(signal, rate, is_multichannel, auto_preprocess)
                results.append(pred)
            except Exception as e:
                warnings.warn(f"Failed to process signal in universal batch: {e}")
                # Return default prediction on error
                results.append({
                    'event_type': 'background',
                    'event_confidence': 0.0,
                    'risk_score': 0.0,
                    'damage_type': 'clean',
                    'damage_confidence': 0.0,
                    'sensor_type': 'DAS',
                    'sensor_confidence': 0.0
                })
        
        return results
