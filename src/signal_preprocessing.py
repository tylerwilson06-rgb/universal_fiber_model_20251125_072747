"""
Universal Signal Preprocessing Module
Handles any sampling rate and signal length automatically for true universality

Based on signal processing best practices:
- Anti-aliasing filters for resampling
- Multi-window averaging for variable lengths
- Quality-preserving transformations
"""

import numpy as np
import warnings
from scipy import signal as scipy_signal
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available. Using scipy for resampling (may be slower).")


class UniversalSignalPreprocessor:
    """
    Universal signal preprocessor that handles any sampling rate and signal length.
    
    This class implements advanced signal processing techniques to ensure the model
    can work with signals from any sensor configuration while maintaining accuracy.
    
    Key Features:
    - Intelligent resampling with anti-aliasing
    - Adaptive length handling (windowing, padding, truncation)
    - Quality preservation and validation
    - Multi-window feature aggregation for long signals
    """
    
    # Standard configuration (matches training data)
    TARGET_SAMPLING_RATE = 10000  # Hz (matches training)
    TARGET_DURATION = 1.0  # seconds
    TARGET_SAMPLES = int(TARGET_SAMPLING_RATE * TARGET_DURATION)  # 10000 samples
    
    # Resampling quality settings
    RESAMPLE_METHOD = 'librosa'  # 'librosa' (best quality) or 'scipy' (faster)
    LIBROSA_QUALITY = 'kaiser_best'  # Highest quality resampling
    
    # Window configuration for long signals
    WINDOW_OVERLAP = 0.5  # 50% overlap for better feature extraction
    MIN_WINDOW_SAMPLES = 512  # Minimum window size
    
    # Quality thresholds
    MIN_SAMPLING_RATE = 100  # Hz - minimum viable rate
    MAX_SAMPLING_RATE = 1000000  # Hz - maximum reasonable rate
    MIN_SIGNAL_LENGTH = 10  # samples - absolute minimum
    
    def __init__(self, target_sampling_rate=None, target_duration=None, 
                 resample_method=None, use_multi_window=True):
        """
        Initialize universal signal preprocessor.
        
        Args:
            target_sampling_rate: Target sampling rate in Hz (default: 10000)
            target_duration: Target duration in seconds (default: 1.0)
            resample_method: 'librosa' (best quality) or 'scipy' (faster)
            use_multi_window: If True, uses multi-window averaging for long signals
        """
        self.target_fs = target_sampling_rate or self.TARGET_SAMPLING_RATE
        self.target_duration = target_duration or self.TARGET_DURATION
        self.target_samples = int(self.target_fs * self.target_duration)
        
        self.resample_method = resample_method or self.RESAMPLE_METHOD
        if self.resample_method == 'librosa' and not LIBROSA_AVAILABLE:
            warnings.warn("librosa not available, falling back to scipy resampling")
            self.resample_method = 'scipy'
        
        self.use_multi_window = use_multi_window
        
        # Statistics for quality monitoring
        self.last_resample_ratio = None
        self.last_length_ratio = None
        self.warnings_issued = []
    
    def preprocess(self, signal, original_sampling_rate=None, 
                   is_multichannel=False, return_info=False):
        """
        Preprocess signal to standard format.
        
        This is the main method that handles:
        1. Resampling to target sampling rate (if needed)
        2. Length normalization (padding, truncation, or windowing)
        3. Quality validation
        
        Args:
            signal: Input signal (1D or 2D numpy array)
            original_sampling_rate: Original sampling rate in Hz
                                   If None, assumes target rate
            is_multichannel: Whether signal is multi-channel
            return_info: If True, returns preprocessing metadata
        
        Returns:
            Preprocessed signal (standardized format)
            If return_info=True: (signal, info_dict)
        """
        # Convert to numpy array
        signal = np.asarray(signal, dtype=np.float64)
        original_shape = signal.shape
        
        # Handle multi-channel signals
        if is_multichannel and len(signal.shape) == 2:
            # Process each channel and average (or process separately)
            processed_channels = []
            for ch in range(signal.shape[1]):
                ch_signal = signal[:, ch]
                processed = self._preprocess_single_channel(
                    ch_signal, original_sampling_rate
                )
                processed_channels.append(processed)
            
            processed_signal = np.column_stack(processed_channels)
        else:
            # Single channel
            signal_1d = signal.flatten()
            processed_signal = self._preprocess_single_channel(
                signal_1d, original_sampling_rate
            )
        
        # Build info dictionary
        info = {
            'original_shape': original_shape,
            'processed_shape': processed_signal.shape,
            'resample_ratio': self.last_resample_ratio,
            'length_ratio': self.last_length_ratio,
            'warnings': self.warnings_issued.copy()
        }
        
        # Reset warnings for next call
        self.warnings_issued = []
        
        if return_info:
            return processed_signal, info
        return processed_signal
    
    def _preprocess_single_channel(self, signal, original_sampling_rate):
        """Preprocess a single-channel signal."""
        original_length = len(signal)
        
        # Step 1: Validate input
        signal = self._validate_and_clean(signal, original_sampling_rate)
        
        # Step 2: Resample if needed
        if original_sampling_rate is not None and original_sampling_rate != self.target_fs:
            signal = self._resample_signal(signal, original_sampling_rate, self.target_fs)
            self.last_resample_ratio = self.target_fs / original_sampling_rate
        else:
            self.last_resample_ratio = 1.0
        
        # Step 3: Handle length
        signal = self._handle_signal_length(signal)
        self.last_length_ratio = len(signal) / original_length if original_length > 0 else 1.0
        
        return signal
    
    def _validate_and_clean(self, signal, sampling_rate):
        """Validate and clean input signal."""
        # Check for empty signal
        if signal.size == 0:
            raise ValueError("Signal is empty")
        
        # Check for NaN/Inf
        if np.any(np.isnan(signal)):
            raise ValueError("Signal contains NaN values")
        if np.any(np.isinf(signal)):
            warnings.warn("Signal contains Inf values. Replacing with finite values.")
            signal = np.nan_to_num(signal, nan=0.0, posinf=1e10, neginf=-1e10)
            self.warnings_issued.append("Inf values replaced")
        
        # Validate sampling rate
        if sampling_rate is not None:
            if sampling_rate < self.MIN_SAMPLING_RATE:
                raise ValueError(
                    f"Sampling rate {sampling_rate} Hz is too low. "
                    f"Minimum: {self.MIN_SAMPLING_RATE} Hz"
                )
            if sampling_rate > self.MAX_SAMPLING_RATE:
                warnings.warn(
                    f"Sampling rate {sampling_rate} Hz is very high. "
                    f"Results may be slower."
                )
                self.warnings_issued.append(f"High sampling rate: {sampling_rate} Hz")
        
        # Validate signal length
        if len(signal) < self.MIN_SIGNAL_LENGTH:
            warnings.warn(
                f"Signal is very short ({len(signal)} samples). "
                f"Results may be unreliable."
            )
            self.warnings_issued.append(f"Very short signal: {len(signal)} samples")
        
        return signal
    
    def _resample_signal(self, signal, original_fs, target_fs):
        """
        Resample signal with anti-aliasing.
        
        Uses high-quality resampling methods to preserve signal characteristics.
        """
        if len(signal) < 2:
            return signal
        
        ratio = target_fs / original_fs
        
        if self.resample_method == 'librosa' and LIBROSA_AVAILABLE:
            # Librosa provides superior quality for audio/signal processing
            # Uses kaiser-windowed sinc interpolation
            resampled = librosa.resample(
                signal,
                orig_sr=original_fs,
                target_sr=target_fs,
                res_type=self.LIBROSA_QUALITY
            )
        else:
            # Scipy FFT-based resampling (good quality, faster)
            num_samples = int(len(signal) * ratio)
            if num_samples < 1:
                num_samples = 1
            
            # Use scipy.signal.resample (FFT-based, includes anti-aliasing)
            resampled = scipy_signal.resample(signal, num_samples)
        
        return resampled
    
    def _handle_signal_length(self, signal):
        """
        Handle variable signal lengths intelligently.
        
        Strategies:
        - Too short: Pad with zeros (or mirror padding)
        - Too long: Multi-window averaging (best) or center truncation
        - Just right: Use as-is
        """
        current_length = len(signal)
        
        if current_length == self.target_samples:
            # Perfect length
            return signal
        
        elif current_length < self.target_samples:
            # Too short: Pad with zeros
            padding = self.target_samples - current_length
            padded = np.pad(signal, (0, padding), mode='constant', constant_values=0)
            
            if current_length < self.MIN_WINDOW_SAMPLES:
                self.warnings_issued.append(
                    f"Short signal padded: {current_length} -> {self.target_samples} samples"
                )
            
            return padded
        
        else:
            # Too long: Use multi-window averaging (best) or center truncation
            if self.use_multi_window and current_length >= 2 * self.target_samples:
                # Multi-window averaging for better feature extraction
                return self._multi_window_average(signal)
            else:
                # Center truncation (preserves most informative segment)
                return self._center_truncate(signal)
    
    def _multi_window_average(self, signal):
        """
        Extract multiple windows and average features.
        
        This approach is more robust for long signals as it:
        1. Captures information from different time segments
        2. Reduces sensitivity to event timing
        3. Provides better feature representation
        
        Returns a single averaged signal of target length.
        """
        num_windows = max(1, int(len(signal) / (self.target_samples * (1 - self.WINDOW_OVERLAP))))
        num_windows = min(num_windows, 10)  # Limit to 10 windows for performance
        
        window_size = self.target_samples
        step_size = int(window_size * (1 - self.WINDOW_OVERLAP))
        
        windows = []
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            
            if end > len(signal):
                # Last window: take from end
                end = len(signal)
                start = end - window_size
                if start < 0:
                    start = 0
            
            window = signal[start:end]
            
            # Pad if necessary (for last window)
            if len(window) < window_size:
                window = np.pad(window, (0, window_size - len(window)), mode='constant')
            
            windows.append(window)
        
        # Average windows (simple average - could use weighted average)
        averaged = np.mean(windows, axis=0)
        
        self.warnings_issued.append(
            f"Long signal processed with {num_windows} windows"
        )
        
        return averaged
    
    def _center_truncate(self, signal):
        """Truncate signal to center segment (most informative)."""
        excess = len(signal) - self.target_samples
        start = excess // 2
        end = start + self.target_samples
        
        truncated = signal[start:end]
        
        self.warnings_issued.append(
            f"Long signal truncated: {len(signal)} -> {self.target_samples} samples"
        )
        
        return truncated
    
    def get_preprocessing_info(self):
        """Get information about last preprocessing operation."""
        return {
            'target_sampling_rate': self.target_fs,
            'target_samples': self.target_samples,
            'resample_method': self.resample_method,
            'use_multi_window': self.use_multi_window,
            'last_resample_ratio': self.last_resample_ratio,
            'last_length_ratio': self.last_length_ratio,
            'warnings': self.warnings_issued
        }


class AdaptiveFeatureAggregator:
    """
    Advanced feature aggregation for variable-length signals.
    
    For very long signals, extracts features from multiple windows
    and aggregates them intelligently.
    """
    
    def __init__(self, window_size=10000, overlap=0.5, aggregation_method='mean'):
        """
        Initialize adaptive feature aggregator.
        
        Args:
            window_size: Size of each window in samples
            overlap: Overlap ratio between windows (0-1)
            aggregation_method: 'mean', 'max', 'weighted_mean'
        """
        self.window_size = window_size
        self.overlap = overlap
        self.aggregation_method = aggregation_method
        self.step_size = int(window_size * (1 - overlap))
    
    def aggregate_features(self, signal, feature_extractor, fs=10000, is_multichannel=False):
        """
        Extract features from multiple windows and aggregate.
        
        Args:
            signal: Input signal
            feature_extractor: Feature extraction function or object
            fs: Sampling rate
            is_multichannel: Whether signal is multi-channel
        
        Returns:
            Aggregated feature vector
        """
        if len(signal) <= self.window_size:
            # Signal fits in one window
            if hasattr(feature_extractor, 'build_ufv'):
                return feature_extractor.build_ufv(signal, fs, is_multichannel)
            else:
                return feature_extractor.extract_all(signal, is_multichannel)
        
        # Extract features from multiple windows
        feature_vectors = []
        weights = []
        
        num_windows = max(1, (len(signal) - self.window_size) // self.step_size + 1)
        num_windows = min(num_windows, 20)  # Limit for performance
        
        for i in range(num_windows):
            start = i * self.step_size
            end = start + self.window_size
            
            if end > len(signal):
                end = len(signal)
                start = end - self.window_size
                if start < 0:
                    start = 0
            
            window = signal[start:end]
            
            # Extract features from window
            if hasattr(feature_extractor, 'build_ufv'):
                features = feature_extractor.build_ufv(window, fs, is_multichannel)
            else:
                features = feature_extractor.extract_all(window, is_multichannel)
            
            feature_vectors.append(features)
            
            # Weight: center windows more heavily
            center_pos = (start + end) / 2
            signal_center = len(signal) / 2
            distance = abs(center_pos - signal_center)
            weight = np.exp(-distance / (len(signal) / 4))  # Gaussian weighting
            weights.append(weight)
        
        # Aggregate features
        feature_vectors = np.array(feature_vectors)
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-10)  # Normalize
        
        if self.aggregation_method == 'mean':
            aggregated = np.mean(feature_vectors, axis=0)
        elif self.aggregation_method == 'max':
            aggregated = np.max(feature_vectors, axis=0)
        elif self.aggregation_method == 'weighted_mean':
            aggregated = np.average(feature_vectors, axis=0, weights=weights)
        else:
            aggregated = np.mean(feature_vectors, axis=0)
        
        return aggregated





