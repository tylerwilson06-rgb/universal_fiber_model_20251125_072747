"""
Feature Extraction Module
Contains all feature extraction methods for Universal Fiber Sensor Model
"""

import numpy as np
import librosa
import pywt
import warnings

class MultiDomainFeatureExtractor:
    """Extract features across 5 domains: MFCC, Wavelet, Spectral, Temporal, Spatial"""
    
    # Expected feature dimensions
    MFCC_DIM = 120  # 40 + 40 + 40
    WAVELET_DIM = 64
    SPECTRAL_DIM = 6
    TEMPORAL_DIM = 6
    SPATIAL_DIM = 4
    STANDARD_FEATURES_DIM = MFCC_DIM + WAVELET_DIM + SPECTRAL_DIM + TEMPORAL_DIM + SPATIAL_DIM  # 200
    
    # Minimum signal length requirements
    MIN_SIGNAL_LENGTH_MFCC = 512
    MIN_SIGNAL_LENGTH_WAVELET = 32  # Minimum for level 4 decomposition
    
    def __init__(self, fs=10000):
        self.fs = fs
    
    def _validate_signal(self, signal_window, min_length=None):
        """Validate and pad signal if necessary"""
        if min_length is None:
            min_length = max(self.MIN_SIGNAL_LENGTH_MFCC, self.MIN_SIGNAL_LENGTH_WAVELET)
        
        signal_window = np.asarray(signal_window, dtype=np.float64).flatten()
        
        # Check for NaN or Inf
        if np.any(np.isnan(signal_window)) or np.any(np.isinf(signal_window)):
            warnings.warn("Signal contains NaN or Inf values. Replacing with zeros.")
            signal_window = np.nan_to_num(signal_window, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for empty or very short signals
        if len(signal_window) == 0:
            raise ValueError("Signal is empty")
        
        # Pad signal if too short
        if len(signal_window) < min_length:
            padding = min_length - len(signal_window)
            signal_window = np.pad(signal_window, (0, padding), mode='constant', constant_values=0)
            warnings.warn(f"Signal was too short ({len(signal_window)-padding} samples). Padded to {len(signal_window)} samples.")
        
        return signal_window
    
    def extract_mfcc_features(self, signal_window):
        """Extract MFCC + deltas + delta-deltas (120 features exactly)"""
        try:
            signal_window = self._validate_signal(signal_window, self.MIN_SIGNAL_LENGTH_MFCC)
            
            n_mels = max(128, int(self.fs / 125))
            n_fft = min(2048, len(signal_window))
            
            # Ensure n_fft is valid (must be <= signal length and even)
            if n_fft > len(signal_window):
                n_fft = len(signal_window)
            if n_fft % 2 == 1:
                n_fft -= 1
            if n_fft < 256:
                n_fft = 256 if len(signal_window) >= 256 else len(signal_window)
                if n_fft % 2 == 1:
                    n_fft -= 1
            
            # Ensure hop_length is valid
            hop_length = max(1, int(0.01 * self.fs))
            if hop_length >= len(signal_window):
                hop_length = max(1, len(signal_window) // 2)
            
            mfcc = librosa.feature.mfcc(
                y=signal_window,
                sr=self.fs,
                n_mfcc=40,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            
            # Handle edge case where mfcc might be empty or wrong shape
            if mfcc.size == 0 or mfcc.shape[0] != 40:
                # Return zero-filled features
                mfcc_mean = np.zeros(40)
                delta_mean = np.zeros(40)
                delta2_mean = np.zeros(40)
            else:
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                
                mfcc_mean = np.mean(mfcc, axis=1)
                delta_mean = np.mean(delta, axis=1)
                delta2_mean = np.mean(delta2, axis=1)
            
            # Ensure exact dimension
            features = np.concatenate([mfcc_mean, delta_mean, delta2_mean])
            if len(features) != self.MFCC_DIM:
                if len(features) < self.MFCC_DIM:
                    features = np.pad(features, (0, self.MFCC_DIM - len(features)), mode='constant')
                else:
                    features = features[:self.MFCC_DIM]
            
            return features
        except Exception as e:
            warnings.warn(f"Error in MFCC extraction: {e}. Returning zero features.")
            return np.zeros(self.MFCC_DIM)
    
    def extract_wavelet_features(self, signal_window):
        """Extract wavelet packet features (64 features exactly)"""
        try:
            signal_window = self._validate_signal(signal_window, self.MIN_SIGNAL_LENGTH_WAVELET)
            
            # Try wavelet packet decomposition
            maxlevel = 4
            min_length_for_level = 2 ** maxlevel
            
            if len(signal_window) < min_length_for_level:
                # Reduce level if signal is too short
                maxlevel = max(1, int(np.log2(len(signal_window))))
            
            try:
                wp = pywt.WaveletPacket(
                    data=signal_window, 
                    wavelet='db4', 
                    mode='symmetric', 
                    maxlevel=maxlevel
                )
                
                features = []
                nodes = wp.get_level(maxlevel, 'natural')
                for node in nodes:
                    coeffs = node.data
                    features.extend([
                        np.sum(coeffs ** 2),
                        np.log(np.sum(coeffs ** 2) + 1e-10),
                        -np.sum(coeffs ** 2 * np.log(np.abs(coeffs) + 1e-10)) if np.any(coeffs != 0) else 0,
                        np.var(coeffs) if len(coeffs) > 0 else 0
                    ])
            except (AttributeError, IndexError, ValueError):
                # Fallback: use simple wavelet decomposition
                coeffs = pywt.wavedec(signal_window, 'db4', level=min(maxlevel, 4))
                features = []
                for coeff in coeffs:
                    if len(coeff) > 0:
                        features.extend([
                            np.sum(coeff ** 2),
                            np.log(np.sum(coeff ** 2) + 1e-10),
                            -np.sum(coeff ** 2 * np.log(np.abs(coeff) + 1e-10)) if np.any(coeff != 0) else 0,
                            np.var(coeff)
                        ])
            
            # Ensure exact 64 features
            features = np.array(features)
            if len(features) < self.WAVELET_DIM:
                features = np.pad(features, (0, self.WAVELET_DIM - len(features)), mode='constant')
            elif len(features) > self.WAVELET_DIM:
                features = features[:self.WAVELET_DIM]
            
            return features
        except Exception as e:
            warnings.warn(f"Error in wavelet extraction: {e}. Returning zero features.")
            return np.zeros(self.WAVELET_DIM)
    
    def extract_spectral_features(self, signal_window):
        """Extract spectral features (6 features exactly)"""
        try:
            signal_window = self._validate_signal(signal_window)
            
            fft = np.fft.rfft(signal_window)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(signal_window), 1/self.fs)
            
            power = magnitude ** 2
            power_sum = np.sum(power)
            
            if power_sum == 0 or len(freqs) == 0:
                return np.zeros(self.SPECTRAL_DIM)
            
            centroid = np.sum(freqs * power) / power_sum
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / power_sum)
            
            cumsum = np.cumsum(power)
            rolloff_idx = np.where(cumsum >= 0.85 * power_sum)[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            
            flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10)
            
            # Handle kurtosis edge cases
            if np.std(magnitude) < 1e-10:
                kurtosis = 0.0
            else:
                kurtosis = np.mean((magnitude - np.mean(magnitude)) ** 4) / (np.std(magnitude) ** 4 + 1e-10)
            
            peak_freq = freqs[np.argmax(magnitude)] if len(magnitude) > 0 else 0.0
            
            return np.array([centroid, bandwidth, rolloff, flatness, kurtosis, peak_freq])
        except Exception as e:
            warnings.warn(f"Error in spectral extraction: {e}. Returning zero features.")
            return np.zeros(self.SPECTRAL_DIM)
    
    def extract_temporal_features(self, signal_window):
        """Extract temporal features (6 features exactly)"""
        try:
            signal_window = self._validate_signal(signal_window)
            
            if len(signal_window) == 0:
                return np.zeros(self.TEMPORAL_DIM)
            
            rms = np.sqrt(np.mean(signal_window ** 2))
            peak = np.max(np.abs(signal_window))
            zcr = np.sum(np.diff(np.sign(signal_window)) != 0) / len(signal_window) if len(signal_window) > 1 else 0
            crest = peak / (rms + 1e-10)
            mad = np.mean(np.abs(signal_window - np.mean(signal_window)))
            
            # Autocorrelation with edge case handling
            if len(signal_window) > 1:
                autocorr = np.correlate(signal_window, signal_window, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                if len(autocorr) > 0 and autocorr[0] > 1e-10:
                    autocorr = autocorr / autocorr[0]
                    lag1_corr = autocorr[1] if len(autocorr) > 1 else 0
                else:
                    lag1_corr = 0
            else:
                lag1_corr = 0
            
            return np.array([rms, peak, zcr, crest, mad, lag1_corr])
        except Exception as e:
            warnings.warn(f"Error in temporal extraction: {e}. Returning zero features.")
            return np.zeros(self.TEMPORAL_DIM)
    
    def extract_spatial_features(self, multichannel_signal):
        """Extract spatial features for multi-channel data (4 features exactly)"""
        try:
            if len(multichannel_signal.shape) < 2:
                return np.zeros(self.SPATIAL_DIM)
            
            num_channels = multichannel_signal.shape[1]
            
            if num_channels < 2:
                return np.zeros(self.SPATIAL_DIM)
            
            # Gradient
            grad = np.mean(np.abs(np.diff(multichannel_signal, axis=1)))
            
            # Correlations
            correlations = []
            for i in range(num_channels - 1):
                try:
                    corr = np.corrcoef(multichannel_signal[:, i], multichannel_signal[:, i+1])[0, 1]
                    if not (np.isnan(corr) or np.isinf(corr)):
                        correlations.append(corr)
                except:
                    pass
            
            mean_corr = np.mean(correlations) if correlations else 0
            std_corr = np.std(correlations) if correlations else 0
            
            # Energy spread
            energy_per_channel = np.sum(multichannel_signal ** 2, axis=0)
            energy_spread = np.std(energy_per_channel)
            
            return np.array([grad, mean_corr, std_corr, energy_spread])
        except Exception as e:
            warnings.warn(f"Error in spatial extraction: {e}. Returning zero features.")
            return np.zeros(self.SPATIAL_DIM)
    
    def extract_all(self, signal_window, is_multichannel=False):
        """Extract all standard features (200 features exactly)"""
        # Handle multi-channel signals - use mean across channels for better representation
        if is_multichannel and len(signal_window.shape) == 2:
            signal_1d = np.mean(signal_window, axis=1)
        else:
            signal_1d = signal_window.flatten()
        
        mfcc_feat = self.extract_mfcc_features(signal_1d)
        wavelet_feat = self.extract_wavelet_features(signal_1d)
        spectral_feat = self.extract_spectral_features(signal_1d)
        temporal_feat = self.extract_temporal_features(signal_1d)
        spatial_feat = self.extract_spatial_features(signal_window) if is_multichannel else np.zeros(self.SPATIAL_DIM)
        
        all_features = np.concatenate([mfcc_feat, wavelet_feat, spectral_feat, temporal_feat, spatial_feat])
        
        # Final dimension check
        if len(all_features) != self.STANDARD_FEATURES_DIM:
            if len(all_features) < self.STANDARD_FEATURES_DIM:
                all_features = np.pad(all_features, (0, self.STANDARD_FEATURES_DIM - len(all_features)), mode='constant')
            else:
                all_features = all_features[:self.STANDARD_FEATURES_DIM]
        
        return all_features


class ProprietaryFeatures:
    """Proprietary fiber-aware features: RBE, DESI, SCR, BSI"""
    
    PROPRIETARY_FEATURES_DIM = 4
    
    def calculate_RBE(self, signal_window):
        """Rayleigh Backscatter Entropy"""
        try:
            signal_window = np.asarray(signal_window, dtype=np.float64).flatten()
            
            if len(signal_window) == 0:
                return 0.0
            
            # Remove NaN/Inf
            signal_window = np.nan_to_num(signal_window, nan=0.0, posinf=0.0, neginf=0.0)
            
            hist, _ = np.histogram(signal_window, bins=50, density=True)
            hist = hist + 1e-10
            entropy = -np.sum(hist * np.log(hist))
            return float(entropy)
        except Exception as e:
            warnings.warn(f"Error in RBE calculation: {e}. Returning 0.0.")
            return 0.0
    
    def calculate_DESI(self, signal_window):
        """Dynamic Event Shape Index"""
        try:
            signal_window = np.asarray(signal_window, dtype=np.float64).flatten()
            
            if len(signal_window) < 16:  # Minimum for level 4 decomposition
                return 0.5
            
            signal_window = np.nan_to_num(signal_window, nan=0.0, posinf=0.0, neginf=0.0)
            
            level = 4
            if len(signal_window) < 2 ** level:
                level = max(1, int(np.log2(len(signal_window))))
            
            coeffs = pywt.wavedec(signal_window, 'db4', level=level)
            
            if len(coeffs) == 0:
                return 0.5
            
            low_scale_energy = np.sum(coeffs[-1] ** 2)
            high_scale_energy = np.sum(coeffs[0] ** 2)
            
            if high_scale_energy < 1e-10:
                return 0.5
            
            return float(low_scale_energy / (high_scale_energy + 1e-10))
        except Exception as e:
            warnings.warn(f"Error in DESI calculation: {e}. Returning 0.5.")
            return 0.5
    
    def calculate_SCR(self, multichannel_signal):
        """Spatial Coherence Ratio"""
        try:
            if len(multichannel_signal.shape) < 2:
                return 0.5
            
            num_channels = multichannel_signal.shape[1]
            
            if num_channels < 2:
                return 0.5
            
            correlations = []
            for i in range(num_channels - 1):
                try:
                    corr = np.corrcoef(multichannel_signal[:, i], multichannel_signal[:, i+1])[0, 1]
                    if not (np.isnan(corr) or np.isinf(corr)):
                        correlations.append(corr)
                except:
                    pass
            
            return float(np.mean(correlations)) if correlations else 0.5
        except Exception as e:
            warnings.warn(f"Error in SCR calculation: {e}. Returning 0.5.")
            return 0.5
    
    def calculate_BSI(self, signal_window):
        """Backscatter Stability Index"""
        try:
            signal_window = np.asarray(signal_window, dtype=np.float64).flatten()
            signal_window = np.nan_to_num(signal_window, nan=0.0, posinf=0.0, neginf=0.0)
            return float(np.var(signal_window))
        except Exception as e:
            warnings.warn(f"Error in BSI calculation: {e}. Returning 0.0.")
            return 0.0
    
    def extract_all(self, signal_window, is_multichannel=False):
        """Extract all proprietary features (4 features exactly)"""
        # For multi-channel, use mean across channels for 1D features
        if is_multichannel and len(signal_window.shape) == 2:
            signal_1d = np.mean(signal_window, axis=1)
        else:
            signal_1d = signal_window.flatten()
        
        rbe = self.calculate_RBE(signal_1d)
        desi = self.calculate_DESI(signal_1d)
        scr = self.calculate_SCR(signal_window) if is_multichannel else 0.5
        bsi = self.calculate_BSI(signal_1d)
        
        return np.array([rbe, desi, scr, bsi], dtype=np.float64)


class UniversalFeatureVectorBuilder:
    """Build complete UFV from any sensor signal"""
    
    UFV_DIM = 204  # 200 standard + 4 proprietary
    
    def __init__(self):
        self.feature_extractor = MultiDomainFeatureExtractor()
        self.proprietary = ProprietaryFeatures()
    
    def build_ufv(self, signal_window, fs=10000, is_multichannel=False):
        """
        Build UFV (204 features exactly)
        
        Args:
            signal_window: numpy array (1D or 2D for multi-channel)
            fs: sampling rate in Hz
            is_multichannel: whether signal is multi-channel
        
        Returns:
            numpy array of exactly 204 features
        """
        # Validate inputs
        signal_window = np.asarray(signal_window)
        
        if signal_window.size == 0:
            raise ValueError("Signal window is empty")
        
        if fs <= 0:
            raise ValueError(f"Invalid sampling rate: {fs}. Must be positive.")
        
        if not isinstance(is_multichannel, bool):
            is_multichannel = len(signal_window.shape) == 2 and signal_window.shape[1] > 1
        
        self.feature_extractor.fs = fs
        
        # Extract features
        standard_features = self.feature_extractor.extract_all(signal_window, is_multichannel)
        proprietary_features = self.proprietary.extract_all(signal_window, is_multichannel)
        
        # Concatenate and validate dimension
        ufv = np.concatenate([standard_features, proprietary_features])
        
        if len(ufv) != self.UFV_DIM:
            raise RuntimeError(
                f"UFV dimension mismatch: expected {self.UFV_DIM}, got {len(ufv)}. "
                f"Standard features: {len(standard_features)}, Proprietary: {len(proprietary_features)}"
            )
        
        return ufv.astype(np.float32)  # Ensure float32 for consistency
